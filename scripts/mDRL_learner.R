#######################################################################################
# Script: mDR-L Learner function
# Date: 20/11/24
# Author: Matt Pryce 
# Notes: mDR-learner function for L time point setting
#######################################################################################

library(caTools)
library(tidyverse)
library(ggplot2)
library(DAAG)
library(glmnet)
library(randomForest)
library(caret)
library(grf)
library(xgboost)
library(reshape2)
library(data.table)
library(SuperLearner)


#######################################
#--- For single time point setting ---#
#######################################

#' @param analysis Type of analysis to be run (Complete case or outcome imputation) 
#' @param data The data frame containing all required information
#' @param id Identification for individuals
#' @param outcome The name of the outcome of interest
#' @param exposure The name of the exposure of interest
#' @param outcome_observed_indicator Indicator identifying when the outcome variable is observed (1=observed, 0=missing)
#' @param splits Number of splitsu for cross-fitting (Variations allowed: 1, 3, 10)
#' @param out_method Statistical technique used to run the outcome models
#' @param out_covariates  List containing the names of the variables to be input into each outcome model
#' @param out_SL_lib Library to be used in super learner if selected
#' @param e_method Statistical technique used to run the propensity score model
#' @param e_covariates  List containing the names of the variables to be input into the propensity score model
#' @param e_SL_lib Library to be used in super learner if selected
#' @param nuisance_estimates_input Indicator for whether nuisance estimates provided
#' @param o_0_pred Variable name for unexposed outcome predictions (if provided)
#' @param o_1_pred Variable name for exposed outcome predictions (if provided)
#' @param e_pred Variable name for propensity score predictions (if provided)
#' @param g_pred Variable name for censoring predictions (if provided)
#' @param g_method Statistical technique used to run the missingness model
#' @param g_covariates List containing the names of the variables to be input into the missingness model, excluding exposure
#' @param g_SL_lib Library to be used in super learner if selected for missingness model
#' @param pse_method Statistical technique used to run the pseudo outcome model
#' @param pse_covariates List containing the names of the variables to be input into the pseudo outcome model
#' @param pse_SL_lib Library to be used in super learner if selected for pseudo outcome model
#' @param imp_covariates Covariates to be used in SL imputation model if SL imputation used
#' @param imp_SL_lib SL libaray for imputation model if SL imputation used
#' @param newdata New data to create predictions for

#' @return A list containing: CATE estimates, a dataset used to train the learner, dataset containing all 
#'         pseudo-outcome predictions (if splits 3) 


mDRL_learner <- function(data,
                         id,
                         outcome,
                         exposure,
                         outcome_observed_indicator = "None",
                         time,
                         splits = c(1,3,10),
                         e_method = c("Parametric","Random forest","Super Learner"),
                         e_covariates,
                         e_SL_lib,
                         out_method = c("Parametric","Random forest","Super Learner"),
                         out_covariates,
                         out_SL_lib,
                         g_method = c("Parametric","Random forest","Super Learner"),
                         g_covariates = c(),
                         g_SL_lib,
                         pse_method = c("Parametric","Random forest","Super Learner"),
                         pse_covariates,
                         pse_SL_lib,
                         newdata
){
  learner <- "mDRL-learner"
  
  #-----------------------#
  #--- Data management ---#
  #-----------------------#
  
  #Checking and keeping appropriate data
  vars <- c(id, outcome,exposure,time, outcome_observed_indicator)
  vars <- append(vars,e_covariates)
  vars <- append(vars,out_covariates)
  vars <- vars[!duplicated(vars)]
  vars <- append(vars,g_covariates)
  vars <- vars[!duplicated(vars)]
  data <- subset(data,select = vars)
  new_data <- newdata
  
  #Standardising names
  names(data)[names(data) == id] <- "ID"
  names(data)[names(data) == time] <- "time"
  names(data)[names(data) == exposure] <- "A"
  names(data)[names(data) == outcome] <- "Y"
  names(data)[names(data) == outcome_observed_indicator] <- "G"
  
  #Checking Y binary or continuous
  Y_comp <- na.omit(data$Y)
  Y_bin <- as.numeric(all(Y_comp %in% 0:1))
  
  #Number of observations read in
  n_og <- dim(data)[1]
  
  #Checking number of time points (including the baseline visit)
  num_tps <- length(unique(data$time))
  
  #Listing the time point values
  tps <- unique(data$time)
  
  #Re-numbering time point
  tps2_list <- 0:(num_tps - 1)
  data$time2 <- 0
  for (i in 0:(num_tps-1)){
    for (j in 1:length(data$ID)){
      if (data$time[j] == tps[i+1]){
        data$time2[j] <- tps2_list[i+1]
      }
    }
  }
  
  #Checking which variables are baseline variables and which are time varying
  covs <- append(e_covariates,out_covariates)
  covs <- append(covs,g_covariates)
  covs <- covs[!duplicated(covs)]
  
  b_covs <- "temp"
  tv_covs <- "temp"
  for (i in covs){
    tv_check <- any(by(data[,i],data$ID,function(x) length(unique(x[!is.na(x)]))) > 1)
    if (tv_check == FALSE){
      b_covs <- append(b_covs,i)
    }
    else if (tv_check == TRUE){
      tv_covs <- append(tv_covs,i)
    }
  }
  b_covs <- b_covs[! b_covs %in% c("temp")]
  tv_covs <- tv_covs[! tv_covs %in% c("temp")]
  tv_covs_c <- append(tv_covs,"G")

  #Removing observations with missing baseline covariates or exposure
  data_time0 <- subset(data,data$time2 == 0)
  for (i in 1:length(b_covs)){
    data_time0 <- data_time0 %>% drop_na(b_covs[i])
    keep_ids <- data_time0$ID
    data <- subset(data, data$ID %in% keep_ids)

    data_time0 <- data_time0 %>% drop_na("A")
    keep_ids <- data_time0$ID
    data <- subset(data, data$ID %in% keep_ids)
  }

  #Number of observations after step
  n2 <- dim(data)[1]

  if (n_og != n2){
    warning("Observations removed due to missing baseline covariate values or missing exposure")
  }

  #Transforming dataset into wide format with one row per individual
  wide_temp <- dcast(setDT(data), ID~time2, value.var=tv_covs_c)

  #Get non tv variables and get one of each (n)
  gen_vars <- c("ID", outcome, exposure)
  b_covs_all <- append(b_covs, gen_vars)
  wide_temp2 <- subset(data,select = b_covs_all)
  wide_temp2 <- wide_temp2[!duplicated(wide_temp2$ID),]

  #Combine to have wide dataset
  data_wide <- as.data.frame(merge(wide_temp2,wide_temp,by="ID"))

  #Checking number of splits is correct
  if (splits != 1 & splits != 2 & splits != 4 & splits != 10){
    return("Error: Number of splits not compatible")
  }

  #Creating splits
  data_wide$s <- rep(1:length(data_wide$Y),1) %% splits

  #Merge splits for long data
  split_sub <- subset(data_wide,select = c("ID","s"))
  data_long <- merge(data,split_sub,by="ID")

  #Checking new data is suitable
  if (is.data.frame(new_data)==FALSE){
    return("Warning: New data is not suitable, original data will be used for predictions")
  }

  #New data N
  if (is.data.frame(new_data)==TRUE){
    temp <- subset(new_data,new_data$time==0)
    new_data_N <- dim(temp)[1]
  }
  else if (is.data.frame(new_data)==FALSE){
    new_data_N <- dim(data_wide)[1]
  }

  

  #------------------------------------------------------#
  #--- Running nuisance models & pseudo outcome model ---#
  #------------------------------------------------------#
  
  #--- Iterating over each split (cross-fitting) ---#
  all_output <- list()
  po_preds <- rep(0,new_data_N)
  for (i in 0:(splits-1)){
    #--- Collecting data for training models ---#
    #Propensity score
    e_data <- subset(data_long,data_long$time2 == 0)
    e_data2 <- subset(e_data,select = append(e_covariates,"s"))
    
    #Censoring
    g_data <- data_long    #Re-organising long data to have censoring record on previous covariate line
    g_data$G2 <- g_data$G
    for (j in 1:length(g_data$ID)){
      if (g_data$time2[j] > 0){
        g_data$G2[j-1] <- g_data$G[j]
      }
    }
    g_data <- subset(g_data, g_data$G != 0 | g_data$G2 != 0)
    g_data <- subset(g_data,g_data$time2 < (num_tps - 1))
    g_data$time_G <- g_data$time2 + 1
    
    if (splits == 1){
      #Propensity score
      e_fit_data_temp <- subset(e_data2,e_data2$s == i)
      e_fit_data <- as.matrix(subset(e_fit_data_temp, select = -c(s)))
      e_fit_data2 <- as.data.frame(subset(e_fit_data_temp, select = -c(s)))
      A_e <- subset(e_data,select = c("A","s"))
      A_e <- subset(A_e,A_e$s == i)
      
      #Censoring
      g_fit_data_c <- g_data
    }
    if (splits == 10){
      #Propensity score
      e_fit_data_temp <- subset(e_data2,e_data2$s != i)
      e_fit_data <- as.matrix(subset(e_fit_data_temp, select = -c(s)))
      e_fit_data2 <- as.data.frame(subset(e_fit_data_temp, select = -c(s)))
      A_e <- subset(e_data,select = c("A","s"))
      A_e <- subset(A_e,A_e$s != i)
      
      #Censoring
      g_fit_data_c <- subset(g_data,g_data$s != i)
    }
    
    
    
    #--- Collecting data for pseudo-outcome predictions ---#
    if (splits == 1){
      #Propesnity score
      e_pred_data_temp <- subset(e_data2,e_data2$s == i)
      e_pred_data <- as.matrix(subset(e_pred_data_temp, select = -c(s)))
      e_pred_data <- subset(e_pred_data_temp, select = -c(s))
      
      #Censoring model
      g_pred_data <- g_data
      
      #Data for outcome repeated regs predictions
      g_out_pred_data <- g_data
      
      #Variables to put in censoring model
      g_vars_all <- append(g_covariates,"time_G")
      g_fit_data <- as.matrix(subset(g_fit_data_c,select = g_vars_all))
      g_pred_data_matrix <- as.matrix(subset(g_pred_data,select = g_vars_all))
      g_out_pred_data_matrix <- as.matrix(subset(g_out_pred_data,select = g_vars_all))
    }
    else if (splits == 10){
      #Propensity score
      e_pred_data_temp <- subset(e_data2,e_data2$s == i)
      e_pred_data <- as.matrix(subset(e_pred_data_temp, select = -c(s)))
      e_pred_data <- subset(e_pred_data_temp, select = -c(s))
      
      #Censoring model
      g_pred_data <- subset(g_data,g_data$s == i)
      
      #Data for outcome repeated regs predictions
      g_out_pred_data <- subset(g_data,g_data$s != i)
      
      #Variables to put in censoring model
      g_vars_all <- append(g_covariates,"time_G")
      g_fit_data <- as.matrix(subset(g_fit_data_c,select = g_vars_all))
      g_pred_data_matrix <- as.matrix(subset(g_pred_data,select = g_vars_all))
      g_out_pred_data_matrix <- as.matrix(subset(g_out_pred_data,select = g_vars_all))
    }


    #--- Collecting data for Outcome models ---#
    #Initializing lists of data
    out_fit_X_data_0 <- list()
    out_fit_X_data_1 <- list()
    out_fit_X_data2_0 <- list()   #Used for pse outcomes to find next censoring indicator
    out_fit_X_data2_1 <- list()   #Used for pse outcomes to find next censoring indicator
    out_pred_X_data <- list()
    out_pred_IDs <- list()
    out_fit_Y_data_0 <- list()
    out_fit_Y_data_1 <- list()

    if (splits == 1){
      #Trimming data to right split for fitting & prediction
      data_wide_o_fit <- subset(data_wide, data_wide$s == i)
      data_wide_o_pred <- subset(data_wide, data_wide$s == i)
    }
    else if (splits == 10){
      #Trimming data to right split for fitting & prediction
      data_wide_o_fit <- subset(data_wide, data_wide$s != i)
      data_wide_o_pred <- subset(data_wide, data_wide$s == i)
    }

    #Splitting data into treated and untreated
    data_wide_o_fit_0 <- subset(data_wide_o_fit,data_wide_o_fit$A == 0)
    data_wide_o_fit_1 <- subset(data_wide_o_fit,data_wide_o_fit$A == 1)

    #Identifying baseline covaraites to put in outcome models
    b_vars_out <- b_covs[b_covs %in% out_covariates]

    #Identifying time varying covariates to put in outcome models
    tv_vars_out <- tv_covs[tv_covs %in% out_covariates]

    #Defining dataset for fitting with only baseline covariates & not being censored at time 1
    #Defining list of variable to keep
    tv_keep_vars <- paste(tv_vars_out,0,sep="_")
    keep_vars <- append(b_vars_out,tv_keep_vars)

    out_fit_X_data_0_s <- subset(data_wide_o_fit_0, data_wide_o_fit_0$G_1 == 1)
    out_fit_X_data_0 <- append(out_fit_X_data_0,list(subset(out_fit_X_data_0_s, select = keep_vars)))
    out_fit_X_data2_0 <- append(out_fit_X_data2_0,list(out_fit_X_data_0_s))
    out_fit_X_data_1_s <- subset(data_wide_o_fit_1, data_wide_o_fit_1$G_1 == 1)
    out_fit_X_data_1 <- append(out_fit_X_data_1,list(subset(out_fit_X_data_1_s, select = keep_vars)))
    out_fit_X_data2_1 <- append(out_fit_X_data2_1,list(out_fit_X_data_1_s))

    #Creating dataset for pse outcome generation
    out_pred_X_data_s <- data_wide_o_pred
    out_pred_X_data <- append(out_pred_X_data,list(subset(out_pred_X_data_s, select = keep_vars)))

    #Storing ID's for predictions
    out_pred_IDs <- append(out_pred_IDs,list(out_pred_X_data_s$ID))

    for (k in 2:(num_tps-1)){
      cen_fit_tp <- c(paste("G",k,sep="_"))
      cen_Y_tp <- c(paste("G",k-1,sep="_"))
      out_fit_X_data_0_s <- subset(data_wide_o_fit_0,data_wide_o_fit_0[,cen_fit_tp]==1)
      out_fit_X_data_1_s <- subset(data_wide_o_fit_1,data_wide_o_fit_1[,cen_fit_tp]==1)
      out_fit_Y_data_0_s <- subset(data_wide_o_fit_0,data_wide_o_fit_0[,cen_Y_tp]==1)
      out_fit_Y_data_1_s <- subset(data_wide_o_fit_1,data_wide_o_fit_1[,cen_Y_tp]==1)
      out_pred_X_data_s <- subset(data_wide_o_pred, data_wide_o_pred[,cen_Y_tp] == 1)

      #Defining list of variable to keep
      keep_vars <- b_vars_out
      for (j in 1:k){
        tv_keep_vars <- paste(tv_vars_out,j-1,sep="_")
        keep_vars <- append(keep_vars,tv_keep_vars)
      }
      out_fit_X_data_0 <- append(out_fit_X_data_0,list(subset(out_fit_X_data_0_s, select = keep_vars)))
      out_fit_X_data2_0 <- append(out_fit_X_data2_0,list(out_fit_X_data_0_s))
      out_fit_X_data_1 <- append(out_fit_X_data_1,list(subset(out_fit_X_data_1_s, select = keep_vars)))
      out_fit_X_data2_1 <- append(out_fit_X_data2_1,list(out_fit_X_data_1_s))
      out_fit_Y_data_0 <- append(out_fit_Y_data_0,list(subset(out_fit_Y_data_0_s, select = keep_vars)))
      out_fit_Y_data_1 <- append(out_fit_Y_data_1,list(subset(out_fit_Y_data_1_s, select = keep_vars)))
      out_pred_X_data <- append(out_pred_X_data,list(subset(out_pred_X_data_s, select = keep_vars)))

      #Storing ID's for predictions
      out_pred_IDs <- append(out_pred_IDs,list(out_pred_X_data_s$ID))
    }

    #Defining first Y values to fit first model
    end_cen_tp <- c(paste("G",(num_tps-1),sep="_"))
    data_wide_o_fit_0_C_end <- subset(data_wide_o_fit_0,data_wide_o_fit_0[,end_cen_tp]==1)
    data_wide_o_fit_1_C_end <- subset(data_wide_o_fit_1,data_wide_o_fit_1[,end_cen_tp]==1)


    #--- Dataset to put predictions in ---#
    if (splits == 1){
      preds <- subset(data_wide,data_wide$s == i)
    }
    else if (splits == 10){
      preds <- subset(data_wide,data_wide$s == i)
    }
    
    
    #-------------------------------#
    #--- Running nuisance models ---#
    #-------------------------------#

    #--- Propensity score model ---#
    if (e_method == "Random forest"){
      #Running propensity score model
      e_mod <- regression_forest(X = e_fit_data, Y = A_e$A)

      #Obtaining predictions from propensity score model
      e_pred <- predict(e_mod, e_pred_data)
      preds <- cbind(preds,e_pred)
      preds <- preds %>% rename(e_pred = predictions)
    }
    else if (e_method == "Parametric"){
      #Running propensity score model
      e_fit_data_all <- as.data.frame(cbind(A = A_e$A, e_fit_data))
      e_mod <- glm(A ~ . , data = e_fit_data_all, family = binomial())

      #Obtaining predictions from propensity score model
      e_pred <- predict(e_mod, e_pred_data,type="response")
      preds <- cbind(preds,e_pred)
    }
    else if (e_method == "Super learner"){
      #Running propensity score model
      A_sums <- table(A_e$A)
      cv_folds <- min(10,A_sums[1],A_sums[2])
      e_mod <- SuperLearner(Y = A_e$A, X = e_fit_data2,
                            method = "method.NNLS",
                            family = binomial(),
                            cvControl = list(V = cv_folds, stratifyCV=FALSE),
                            SL.library = e_SL_lib)

      #Obtaining predictions from propensity score model
      e_pred <- predict(e_mod, e_pred_data)$pred 
      preds <- cbind(preds,e_pred)
    }
    else {
      return("Error: Method to generate propensity score model not compatible")
    }


    #--- Censoring model ---#
    if (g_method == "Super learner"){
      #Running censoring model
      G_sums <- table(g_fit_data_c$G2)
      cv_folds <- min(10,G_sums[1],G_sums[2])
      g_mod <- SuperLearner(Y = as.integer(g_fit_data_c$G2), X = data.frame(g_fit_data),
                            method = "method.NNLS",
                            family = binomial(),
                            # newX = data.frame(g_pred_data_matrix),
                            id = g_fit_data_c$ID,
                            cvControl = list(V = cv_folds, stratifyCV=FALSE),
                            SL.library = g_SL_lib,
                            control = list(saveCVFitLibrary=TRUE))

      #Obtaining conditional predictions from censoring model for the pse model
      g_pred <- predict(g_mod, g_pred_data_matrix)$pred
      g_pred_data <- as.data.frame(cbind(g_pred_data,g_pred_con = g_pred))
      names(g_pred_data)[names(g_pred_data) == "g_pred_con.V1"] <- "g_pred_con"

      #Calculating non conditional g_preds
      g_pred_data$g_pred <- g_pred_data$g_pred_con
      for (k in 1:length(g_pred_data$time2)){
        if (g_pred_data$time_G[k] > 1){
          g_pred_data$g_pred[k] <- g_pred_data$g_pred[k-1]*g_pred_data$g_pred_con[k]
        }
      }

      #Transforming g_preds to wide
      g_pred_wide <- dcast(setDT(g_pred_data), ID~time_G, value.var=c("g_pred","g_pred_con"))

      #Merging with preds data
      preds <- merge(preds,g_pred_wide,by="ID")

      #Setting missing censoring indicators to 1
      for (k in 1:(num_tps-1)){
        var <- paste("G",k,sep="_")
        for (indiv in 1:length(preds$ID)){
          if (is.na(preds[indiv,var]) == 1){
            preds[indiv,var] = 0
          }
        }
      }

      #Obtaining conditional predictions from censoring model for the outcome model pseudo outcomes
      g_out_pred <- predict(g_mod, as.data.frame(g_out_pred_data_matrix))$pred
      g_out_pred_data <- as.data.frame(cbind(g_out_pred_data,g_pred_con = g_out_pred))

      #Creating a list of conditional censoring probs for outcome model pseudo outcomes
      out_pse_cens_0 <- list()
      out_pse_cens_1 <- list()
      for (k in 2:(num_tps-1)){
        temp_0 <- subset(g_out_pred_data, g_out_pred_data$time2 == (k-1) & g_out_pred_data$A == 0)
        temp_1 <- subset(g_out_pred_data, g_out_pred_data$time2 == (k-1) & g_out_pred_data$A == 1)
        out_pse_cens_0 <- append(out_pse_cens_0,list(temp_0))
        out_pse_cens_1 <- append(out_pse_cens_1,list(temp_1))
      }

      #Defining Y's for pse in first outcome model
      out_0_Y <- temp_0$Y
      out_1_Y <- temp_1$Y
    }
    else if (g_method == "Parametric"){
      #Running censoring model
      g_fit_data_all <- as.data.frame(cbind(G = g_fit_data_c$G2, g_fit_data))
      g_mod <- glm(G ~ . , data = g_fit_data_all, family = binomial())

      #Obtaining conditional predictions from censoring model for the pse model
      g_pred <- predict(g_mod, data.frame(g_pred_data_matrix),type = "response")
      g_pred_data <- as.data.frame(cbind(g_pred_data,g_pred_con = g_pred))

      #Calculating non conditional g_preds
      g_pred_data$g_pred <- g_pred_data$g_pred_con
      for (k in 1:length(g_pred_data)){
        if (g_pred_data$time_G[k] > 1){
          g_pred_data$g_pred[k] <- g_pred_data$g_pred[k-1]*g_pred_data$g_pred_con[k]
        }
      }

      #Transforming g_preds to wide
      g_pred_wide <- dcast(setDT(g_pred_data), ID~time_G, value.var=c("g_pred","g_pred_con"))

      #Merging with preds data
      preds <- merge(preds,g_pred_wide,by="ID")

      #Setting missing censoring indicators to 0
      for (k in 1:(num_tps-1)){
        var <- paste("G",k,sep="_")
        for (indiv in 1:length(preds$ID)){
          if (is.na(preds[indiv,var]) == 1){
            preds[indiv,var] = 0
          }
        }
      }

      #Obtaining conditional predictions from censoring model for the outcome model pseudo outcomes
      g_out_pred <- predict(g_mod, data.frame(g_out_pred_data_matrix),type = "response")
      g_out_pred_data <- as.data.frame(cbind(g_out_pred_data,g_pred_con = g_out_pred))

      #Creating a list of conditional censoring probs for outcome model pseudo outcomes
      out_pse_cens_0 <- list()   #UPDATE
      out_pse_cens_1 <- list()
      for (k in 2:(num_tps-1)){
        temp_0 <- subset(g_out_pred_data, g_out_pred_data$time2 == (k-1) & g_out_pred_data$A == 0)
        temp_1 <- subset(g_out_pred_data, g_out_pred_data$time2 == (k-1) & g_out_pred_data$A == 1)
        out_pse_cens_0 <- append(out_pse_cens_0,list(temp_0))
        out_pse_cens_1 <- append(out_pse_cens_1,list(temp_1))
      }

      #Defining Y's for pse in first outcome model
      out_0_Y <- temp_0$Y
      out_1_Y <- temp_1$Y
    }
    else {
      return("Error: Method to generate censoring model not compatible")
    }



    #--- Outcome regression models ---#
    #Initialising list for outcome models
    out_mods_0 <- list()
    out_mods_1 <- list()
    if (out_method == "Super learner"){
      if (Y_bin == 1){
        #Running first outcome models
        out_mod_0 <- SuperLearner(Y = data_wide_o_fit_0_C_end$Y, X = data.frame(out_fit_X_data_0[[num_tps-1]]),
                                  method = "method.NNLS",
                                  family = binomial(),
                                  cvControl = list(V = 5, stratifyCV=FALSE),
                                  SL.library = out_SL_lib,
                                  newX = out_pred_X_data[[num_tps-1]])
        out_mods_0 <- append(out_mods_0,list(out_mod_0))
        out_mod_1 <- SuperLearner(Y = data_wide_o_fit_1_C_end$Y, X = data.frame(out_fit_X_data_1[[num_tps-1]]),
                                  method = "method.NNLS",
                                  family = binomial(),
                                  cvControl = list(V = 5, stratifyCV=FALSE),
                                  SL.library = out_SL_lib,
                                  newX = out_pred_X_data[[num_tps-1]])
        out_mods_1 <- append(out_mods_1,list(out_mod_1))
      }
      else if (Y_bin == 0){
        #Running first outcome models
        out_mod_0 <- SuperLearner(Y = data_wide_o_fit_0_C_end$Y, X = data.frame(out_fit_X_data_0[[num_tps-1]]),
                                  method = "method.NNLS",
                                  family = gaussian(),
                                  cvControl = list(V = 10, stratifyCV=FALSE),
                                  SL.library = out_SL_lib,
                                  newX = out_pred_X_data[[num_tps-1]])
        out_mods_0 <- append(out_mods_0,list(out_mod_0))

        out_mod_1 <- SuperLearner(Y = data_wide_o_fit_1_C_end$Y, X = data.frame(out_fit_X_data_1[[num_tps-1]]),
                                  method = "method.NNLS",
                                  family = gaussian(),
                                  cvControl = list(V = 10, stratifyCV=FALSE),
                                  SL.library = out_SL_lib,
                                  newX = out_pred_X_data[[num_tps-1]])
        out_mods_1 <- append(out_mods_1,list(out_mod_1))
      }

      #Creating predictions for pseudo outcome from first outcome model
      o_pred_0 <- predict(out_mods_0[[1]], out_pred_X_data[[num_tps-1]])$pred
      o_pred_0 <- as.data.frame(cbind(ID = out_pred_IDs[[num_tps-1]],o_pred_0))
      preds <- as.data.frame(merge(preds,o_pred_0,by="ID",all.x=TRUE))
      preds <- preds %>% rename(o_pred_0_m_1 = V2)
      o_pred_1 <- predict(out_mods_1[[1]], out_pred_X_data[[num_tps-1]])$pred
      o_pred_1 <- as.data.frame(cbind(ID = out_pred_IDs[[num_tps-1]],o_pred_1))
      preds <- as.data.frame(merge(preds,o_pred_1,by="ID",all.x=TRUE))
      preds <- preds %>% rename(o_pred_1_m_1 = V2)

      for (j in 1:(num_tps-2)){

        #Obtaining preds from previous outcome model
        o_prev_mod_pred_0 <- predict(out_mods_0[[j]], out_fit_Y_data_0[[num_tps-1-j]])$pred
        o_prev_mod_pred_1 <- predict(out_mods_1[[j]], out_fit_Y_data_1[[num_tps-1-j]])$pred

        #Identifying correct censoring indicator
        cen_tp <- paste("G",num_tps-j,sep="_")
        o_cen_ind_data_0 <- out_fit_X_data2_0[[num_tps-1-j]]
        o_cen_ind_0 <- o_cen_ind_data_0[,cen_tp]
        o_cen_ind_data_1 <- out_fit_X_data2_1[[num_tps-1-j]]
        o_cen_ind_1 <- o_cen_ind_data_1[,cen_tp]

        #Identifying correct censoring predictions and the peoples ID's
        o_cen_preds_0 <- out_pse_cens_0[[num_tps-1-j]]$g_pred_con
        o_cen_preds_1 <- out_pse_cens_1[[num_tps-1-j]]$g_pred_con
        o_cen_ID_0 <- out_pse_cens_0[[num_tps-1-j]]$ID
        o_cen_ID_1 <- out_pse_cens_1[[num_tps-1-j]]$ID

        #Creating sequential pseudo outcome for next model
        o_prev_mod_pred_0_pse <- rep(0,length(o_prev_mod_pred_0))
        for (k in 1:length(o_prev_mod_pred_0)){
          o_prev_mod_pred_0_pse[k] <- (o_cen_ind_0[k]/o_cen_preds_0[k])*(out_0_Y[k] - o_prev_mod_pred_0[k]) + o_prev_mod_pred_0[k]
        }

        o_prev_mod_pred_1_pse <- rep(0,length(o_prev_mod_pred_1))
        for (k in 1:length(o_prev_mod_pred_1)){
          o_prev_mod_pred_1_pse[k] <- (o_cen_ind_1[k]/o_cen_preds_1[k])*(out_1_Y[k] - o_prev_mod_pred_1[k]) + o_prev_mod_pred_1[k]
        }

        #Merging pse with IDs
        pse_id_0 <- cbind(ID = o_cen_ID_0,o_prev_mod_pred_0_pse)
        pse_id_1 <- cbind(ID = o_cen_ID_1,o_prev_mod_pred_1_pse)

        #Merge to previous dataset, draw out Y list (expect missings for people who were censored) - Need to add "if"
        if (num_tps-2-j > 0){
          out_pse_cens_0[[num_tps-2-j]] <- merge(out_pse_cens_0[[num_tps-2-j]],pse_id_0,by="ID",all.x = TRUE)
          out_pse_cens_1[[num_tps-2-j]] <- merge(out_pse_cens_1[[num_tps-2-j]],pse_id_1,by="ID",all.x = TRUE)

          out_0_Y <-  out_pse_cens_0[[num_tps-2-j]]$o_prev_mod_pred_0_pse
          out_1_Y <-  out_pse_cens_1[[num_tps-2-j]]$o_prev_mod_pred_1_pse
        }
        for (k in 1:length(out_0_Y)){
          if (is.na(out_0_Y[k]) == 1){
            out_0_Y[k] = 999
          }
        }

        for (k in 1:length(out_1_Y)){
          if (is.na(out_1_Y[k]) == 1){
            out_1_Y[k] = 999
          }
        }

        #Running next outcome model
        out_mod_0 <- SuperLearner(X = data.frame(out_fit_X_data_0[[num_tps-1-j]]), Y = as.vector(o_prev_mod_pred_0_pse),
                                  SL.library = out_SL_lib,
                                  method = "method.NNLS",
                                  family = gaussian(),
                                  cvControl = list(V = 5),
                                  control = list(saveCVFitLibrary=TRUE))
        out_mods_0 <- append(out_mods_0,list(out_mod_0))
        out_mod_1 <- SuperLearner(X = data.frame(out_fit_X_data_1[[num_tps-1-j]]), Y = as.vector(o_prev_mod_pred_1_pse),
                                  SL.library = out_SL_lib,
                                  method = "method.NNLS",
                                  family = gaussian(),
                                  cvControl = list(V = 5),
                                  control = list(saveCVFitLibrary=TRUE))
        out_mods_1 <- append(out_mods_1,list(out_mod_1))

        #Obtaining preds for pse outcome
        o_pred_0 <- predict(out_mods_0[[j+1]], out_pred_X_data[[num_tps-1-j]])$pred
        o_pred_0 <- as.data.frame(cbind(ID = out_pred_IDs[[num_tps-1-j]],o_pred_0))
        preds <- as.data.frame(merge(preds,o_pred_0,by="ID",all.x=TRUE))
        names(preds)[names(preds) == "V2"] <- paste("o_pred_0_m",j+1,sep="_")
        o_pred_1 <- predict(out_mods_1[[j+1]], out_pred_X_data[[num_tps-1-j]])$pred
        o_pred_1 <- as.data.frame(cbind(ID = out_pred_IDs[[num_tps-1-j]],o_pred_1))
        preds <- as.data.frame(merge(preds,o_pred_1,by="ID",all.x=TRUE))
        names(preds)[names(preds) == "V2"] <- paste("o_pred_1_m",j+1,sep="_")
      }
    }
    else if (out_method == "Parametric"){

      #Running first outcome models
      temp_data0 <- as.data.frame(cbind(Y = data_wide_o_fit_0_C_end$Y, out_fit_X_data_0[[num_tps-1]]))
      temp_data1 <- as.data.frame(cbind(Y = data_wide_o_fit_1_C_end$Y, out_fit_X_data_1[[num_tps-1]]))
      if (Y_bin == 1){
        out_mod_0 <- glm(Y ~ . , data = temp_data0, family = binomial())
        out_mods_0 <- append(out_mods_0,list(out_mod_0))
        out_mod_1 <- glm(Y ~ . , data = temp_data1, family = binomial())
        out_mods_1 <- append(out_mods_1,list(out_mod_1))
      }
      else if (Y_bin == 0){
        out_mod_0 <- lm(Y ~ . , data = temp_data0)
        out_mods_0 <- append(out_mods_0,list(out_mod_0))
        out_mod_1 <- lm(Y ~ . , data = temp_data1)
        out_mods_1 <- append(out_mods_1,list(out_mod_1))
      }

      #Creating predictions for pseudo outcome from first outcome model
      o_pred_0 <- predict(out_mods_0[[1]], out_pred_X_data[[num_tps-1]], type = "response")
      o_pred_0 <- as.data.frame(cbind(ID = out_pred_IDs[[num_tps-1]],o_pred_0))
      preds <- as.data.frame(merge(preds,o_pred_0,by="ID",all.x=TRUE))
      preds <- preds %>% rename(o_pred_0_m_1 = o_pred_0)
      o_pred_1 <- predict(out_mods_1[[1]], out_pred_X_data[[num_tps-1]], type = "response")
      o_pred_1 <- as.data.frame(cbind(ID = out_pred_IDs[[num_tps-1]],o_pred_1))
      preds <- as.data.frame(merge(preds,o_pred_1,by="ID",all.x=TRUE))
      preds <- preds %>% rename(o_pred_1_m_1 = o_pred_1)

      for (j in 1:(num_tps-2)){

        #Obtaining preds from previous outcome model
        o_prev_mod_pred_0 <- predict(out_mods_0[[j]], out_fit_Y_data_0[[num_tps-1-j]], type = "response")
        o_prev_mod_pred_1 <- predict(out_mods_1[[j]], out_fit_Y_data_1[[num_tps-1-j]], type = "response")

        if (j == 1 & Y_bin == 1){
          #Transforming predictions to continuous range with logit transformation
          o_prev_mod_pred_0 <- logit(o_prev_mod_pred_0)
          o_prev_mod_pred_1 <- logit(o_prev_mod_pred_1)
        }

        #Identifying correct censoring indicator
        cen_tp <- paste("G",num_tps-j,sep="_")
        o_cen_ind_data_0 <- out_fit_X_data2_0[[num_tps-1-j]]
        o_cen_ind_0 <- o_cen_ind_data_0[,cen_tp]
        o_cen_ind_data_1 <- out_fit_X_data2_1[[num_tps-1-j]]
        o_cen_ind_1 <- o_cen_ind_data_1[,cen_tp]

        #Identifying correct censoring predictions and the peoples ID's
        o_cen_preds_0 <- out_pse_cens_0[[num_tps-1-j]]$g_pred_con
        o_cen_preds_1 <- out_pse_cens_1[[num_tps-1-j]]$g_pred_con
        o_cen_ID_0 <- out_pse_cens_0[[num_tps-1-j]]$ID
        o_cen_ID_1 <- out_pse_cens_1[[num_tps-1-j]]$ID

        #Creating sequential pseudo outcome for next model
        o_prev_mod_pred_0_pse <- rep(0,length(o_prev_mod_pred_0))
        for (k in 1:length(o_prev_mod_pred_0)){
          o_prev_mod_pred_0_pse[k] <- (o_cen_ind_0[k]/o_cen_preds_0[k])*(out_0_Y[k] - o_prev_mod_pred_0[k]) + o_prev_mod_pred_0[k]
        }

        o_prev_mod_pred_1_pse <- rep(0,length(o_prev_mod_pred_1))
        for (k in 1:length(o_prev_mod_pred_1)){
          o_prev_mod_pred_1_pse[k] <- (o_cen_ind_1[k]/o_cen_preds_1[k])*(out_1_Y[k] - o_prev_mod_pred_1[k]) + o_prev_mod_pred_1[k]
        }

        #Merging pse with IDs
        pse_id_0 <- cbind(ID = o_cen_ID_0,o_prev_mod_pred_0_pse)
        pse_id_1 <- cbind(ID = o_cen_ID_1,o_prev_mod_pred_1_pse)

        #Merge to previous dataset, draw out Y list (expect missings for people who were censored) - Need to add "if"
        if (num_tps-2-j > 0){
          out_pse_cens_0[[num_tps-2-j]] <- merge(out_pse_cens_0[[num_tps-2-j]],pse_id_0,by="ID",all.x = TRUE)
          out_pse_cens_1[[num_tps-2-j]] <- merge(out_pse_cens_1[[num_tps-2-j]],pse_id_1,by="ID",all.x = TRUE)

          out_0_Y <-  out_pse_cens_0[[num_tps-2-j]]$o_prev_mod_pred_0_pse
          out_1_Y <-  out_pse_cens_1[[num_tps-2-j]]$o_prev_mod_pred_1_pse
        }
        for (k in 1:length(out_0_Y)){
          if (is.na(out_0_Y[k]) == 1){
            out_0_Y[k] = 999
          }
        }

        for (k in 1:length(out_1_Y)){
          if (is.na(out_1_Y[k]) == 1){
            out_1_Y[k] = 999
          }
        }

        #Running next outcome model
        temp_data0 <- as.data.frame(cbind(Y = o_prev_mod_pred_0_pse, out_fit_X_data_0[[num_tps-1-j]]))
        out_mod_0 <- lm(Y ~ . , data = temp_data0)
        out_mods_0 <- append(out_mods_0,list(out_mod_0))
        temp_data1 <- as.data.frame(cbind(Y = o_prev_mod_pred_1_pse, out_fit_X_data_1[[num_tps-1-j]]))
        out_mod_1 <- lm(Y ~ . , data = temp_data1)
        out_mods_1 <- append(out_mods_1,list(out_mod_1))

        #Obtaining preds for pse outcome
        o_pred_0 <- predict(out_mods_0[[j+1]], out_pred_X_data[[num_tps-1-j]], type = "response")
        if (Y_bin == 1){
          o_pred_0 <- inv.logit(o_pred_0)
        }
        o_pred_0 <- as.data.frame(cbind(ID = out_pred_IDs[[num_tps-1-j]],o_pred_0))
        preds <- as.data.frame(merge(preds,o_pred_0,by="ID",all.x=TRUE))
        names(preds)[names(preds) == "o_pred_0"] <- paste("o_pred_0_m",j+1,sep="_")
        o_pred_1 <- predict(out_mods_1[[j+1]], out_pred_X_data[[num_tps-1-j]], type = "response")
        if (Y_bin == 1){
          o_pred_1 <- inv.logit(o_pred_1)
        }
        o_pred_1 <- as.data.frame(cbind(ID = out_pred_IDs[[num_tps-1-j]],o_pred_1))
        preds <- as.data.frame(merge(preds,o_pred_1,by="ID",all.x=TRUE))
        names(preds)[names(preds) == "o_pred_1"] <- paste("o_pred_1_m",j+1,sep="_")
      }

    }
    else {
      return("Error: Method to generate outcome regression models not compatible")
    }
    
    
    ##############################################################################################################################
    
    
    #----------------------#
    #--- Psedo-outcomes ---#
    #----------------------#

    #Altering data so missing values so they don't cause issues
    preds <- preds %>% mutate_if(is.numeric, function(x) ifelse(is.na(x), 99, x))

    #--- Calculating pseudo outcome ---#
    #Setting first step
    end_m_0_mod <- paste("o_pred_0_m",(num_tps-1),sep="_")
    end_m_1_mod <- paste("o_pred_1_m",(num_tps-1),sep="_")
    preds$pse_Y <- preds[,end_m_1_mod] -  preds[,end_m_0_mod]

    #Defining Y as m_0
    preds$o_pred_0_m_0 <- preds$Y
    preds$o_pred_1_m_0 <- preds$Y

    #Defining G_0 as 1 and g_pred_0 as 1
    preds$G_0 <- 1
    preds$g_pred_0 <- 1

    #Iterating over each weighted step
    for (j in 0:(num_tps-2)){
      #Defining the censoring indicator to use
      cen_ind <- paste("G",num_tps - 1 - j,sep="_")

      #Defining the g prob to use
      cen_mod <- paste("g_pred",num_tps - 1 - j,sep = "_")

      #Defining the outcome models to use
      pos_0_mod <- paste("o_pred_0_m",j,sep = "_")
      pos_1_mod <- paste("o_pred_1_m",j,sep = "_")
      neg_0_mod <- paste("o_pred_0_m",j + 1,sep = "_")
      neg_1_mod <- paste("o_pred_1_m",j + 1,sep = "_")

      #Adding to the pseudo outcome
      preds$pse_Y <- preds$pse_Y + ((preds$A - preds$e_pred)/(preds$e_pred*(1-preds$e_pred))) * (preds[,cen_ind]/preds[,cen_mod])  * ((preds$A * preds[,pos_1_mod] + (1-preds$A) * preds[,pos_0_mod]) - (preds$A * preds[,neg_1_mod] + (1-preds$A) * preds[,neg_0_mod]))
    }

    #Storing preds data
    if (i == 0){
      preds_all <- preds
    }
    else{
      preds_all <- rbind(preds_all,preds)
    }
  }
  

  if (splits == 1 | splits == 10){
    #--- Pseudo outcome model---#
    #Defining variable to use in model
    #Identifying baseline covaraites to put in pseudo outcome models
    b_vars_pse <- b_covs[b_covs %in% pse_covariates]
    
    #Identifying time varying covariates to put in outcome models
    tv_vars_pse <- tv_covs[tv_covs %in% pse_covariates]
    
    #Defining list of variable to keep
    tv_keep_vars <- paste(tv_vars_pse,0,sep="_")
    
    keep_vars_fit <- append(b_vars_pse,tv_keep_vars)
    keep_vars_pred <- append(b_vars_pse,tv_vars_pse)
    
    pse_fit_data <- subset(preds_all, select = keep_vars_fit)
    
    #Defining dataset to predict with
    pse_pred_data <- subset(new_data, new_data$time == 0)
    pse_pred_data <- subset(pse_pred_data, select = keep_vars_pred)
    
    #Renaming tv variables in the pred data
    for (var in tv_vars_pse){
      names(pse_pred_data)[names(pse_pred_data) == var] <- paste(var,0,sep="_")
    }
    
    #Running pseudo outcome model
    if (pse_method == "Super learner"){
      pse_mod <- SuperLearner(Y = preds_all$pse_Y, X = data.frame(pse_fit_data),
                              SL.library = pse_SL_lib,
                              method = "method.NNLS",
                              family = gaussian(),
                              cvControl = list(V=5),
                              control = list(saveCVFitLibrary=TRUE))
      pse_pred <- predict(pse_mod, pse_pred_data)$pred
    }
    else if (pse_method == "Parametric"){
      mod_data <- cbind(pse = preds_all$pse_Y,pse_fit_data)
      pse_mod <- lm(pse ~ ., data = mod_data)
      
      pse_pred <- predict(pse_mod, pse_pred_data)
    }
    else {
      return("Error: Method to generate pseudo outcome model not compatible")
    }
    
    output <- list(CATE_est = pse_pred)
    
  }
  
  return(output)
  
  #   #--- Gaining CI's ---#
  #   if (rf_CI == TRUE & pse_method == "Random forest"){
  #     pse_n_rows <- nrow(po_data_all)
  #     for (i in 1:num_boot){
  #       # Randomly sample half the rows
  #       set.seed(596967 + i)  # Set seed for reproducibility
  #       random_indices <- sample(1:pse_n_rows, size = ceiling(pse_n_rows/2), replace = FALSE)
  #       half_sample <- po_data_all[random_indices, ]
  #       half_sample <- half_sample[order(half_sample$ID), ]
  #       
  #       #Running final stage model
  #       tuned_parameters <- pse_model$po_mod$tunable.params
  #       tryCatch(
  #         {
  #           pse_model_hs <- nuis_mod(model = "Pseudo outcome - CI",    
  #                                    data = half_sample,       
  #                                    method = pse_method,        
  #                                    covariates = pse_covariates,
  #                                    SL_lib = pse_SL_lib,
  #                                    pred_data = newdata,
  #                                    CI_tuned_params = tuned_parameters)
  #           
  #           half_sample_est <- pse_model_hs
  #         },
  #         #if an error occurs, tell me the error
  #         error=function(e) {
  #           stop(paste("An error occured when fitting the pseudo outcome model in split ",i,sep=""))
  #           print(e)
  #         }
  #       )
  #       
  #       #Creating R and storing 
  #       full_sample_est <- pse_model$po_pred
  #       
  #       R <- full_sample_est - half_sample_est
  #       
  #       if (i == 1){
  #         R_data <- as.data.frame(R)
  #       }
  #       else {
  #         R_data <- cbind(R_data,R)
  #       }
  #     }
  #     
  #     #Gaining variance of R per person
  #     CI_n_rows <- nrow(pse_model$po_pred)
  #     for (i in 1:CI_n_rows){
  #       
  #       sqrt_n <- sqrt(num_boot)
  #       temp <-  sqrt_n * R_data[i,]
  #       var <- apply(temp, MARGIN = 1, FUN = var)
  #       SE <- sqrt(var)
  #       
  #       LCI <- pse_model$po_pred[i,] - (1/sqrt_n)*SE*1.96
  #       UCI <- pse_model$po_pred[i,] + (1/sqrt_n)*SE*1.96
  #       
  #       # temp <-  R_data[i,] 
  #       # var <- apply(temp, MARGIN = 1, FUN = var)
  #       # SE <- sqrt(var)
  #       # 
  #       # LCI <- pse_model$po_pred[i,] - SE*1.96   
  #       # UCI <- pse_model$po_pred[i,] + SE*1.96   
  #       
  #       if (i == 1){
  #         LCI_data <- LCI
  #         UCI_data <- UCI
  #       }
  #       else {
  #         LCI_data <- append(LCI_data,LCI)
  #         UCI_data <- append(UCI_data,UCI)
  #       }
  #     }
  #   }
  #   else if (rf_CI == TRUE & pse_method != "Random forest"){
  #     return("Inappropriate pseudo-outcome regression method for obtaining CI's")
  #   }
  # }
  # 
  # #-----------------------------#
  # #--- Returning information ---#
  # #-----------------------------#
  # if (splits == 1 | splits == 10){
  #   if (rf_CI != TRUE){
  #     output <- list(CATE_est = pse_model$po_pred,
  #                    data = po_data_all)
  #   }
  #   else if (rf_CI == TRUE){
  #     output <- list(CATE_est = pse_model$po_pred,
  #                    CATE_LCI = LCI_data,
  #                    CATE_UCI = UCI_data,
  #                    data = po_data_all)
  #   }
  # }
  # 
  # return(output)
  # return(preds_all)
}



###############################################################

# load("~/PhD/DR_Missing_Paper/Extensions/Results/Ext_output_1.RData")
# check <- model_info_list$i$sim_data_train
# 
# 
# output <- mDRL_learner(data = check, 
#                        id = "ID", 
#                        outcome = "Y", 
#                        exposure = "A",
#                        outcome_observed_indicator = "G_obs", 
#                        time = "time",
#                        splits = 10,
#                        e_covariates = c("X1","X2","X3","X4","X5","X6","X7","X8","X9","X10","X11","X12"),
#                        e_method = "Super learner",
#                        e_SL_lib = c("SL.glm"),  #"SL.randomForest", 
#                        out_covariates = c("X1","X2","X3","X4","X5","X6","X7","X8","X9","X10","X11","X12"),
#                        out_method = "Parametric",
#                        out_SL_lib = c("SL.randomForest","SL.lm"),
#                        g_covariates = c("X1","X2","X3","X4","X5","X6","X7","X8","X9","X10","X11","X12"),
#                        g_method = "Super learner",
#                        g_SL_lib = c("SL.glm"),#"SL.randomForest", 
#                        pse_covariates = c("X1","X2","X3","X4","X5","X6","X7","X8","X9","X10","X11","X12"),
#                        pse_method = "Parametric",
#                        pse_SL_lib = c("SL.lm"),
#                        newdata = check)
