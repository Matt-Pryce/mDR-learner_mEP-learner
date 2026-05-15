# =====================================================================
# Script: DR-Learner function
# Date: 20/03/24 # nolint
# Author: Matt Pryce
# Notes: DR-learner function for single time point setting
# =====================================================================

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
library(hal9001)


#######################################
#--- For single time point setting ---#
#######################################

#' @param analysis Type of analysis (Complete case or outcome imputation)
#' @param data The data frame containing all required information
#' @param id Identification for individuals
#' @param outcome The name of the outcome of interest
#' @param exposure The name of the exposure of interest
#' @param outcome_observed_indicator Indicator identifying when the outcome variable is observed (1=observed, 0=missing) # nolint
#' @param splits Number of splitsu for cross-fitting (Variations allowed: 1, 3, 10) # nolint
#' @param out_method Statistical technique used to run the outcome models
#' @param out_covariates  List containing the names of the variables to be input into each outcome model # nolint
#' @param out_SL_lib Library to be used in super learner if selected
#' @param e_method Statistical technique used to run the propensity score model
#' @param e_covariates  List containing the names of the variables to be input into the propensity score model # nolint
#' @param e_SL_lib Library to be used in super learner if selected
#' @param nuisance_estimates_input Indicator for whether nuisance estimates provided # nolint
#' @param o_0_pred Variable name for unexposed outcome predictions (if provided)
#' @param o_1_pred Variable name for exposed outcome predictions (if provided)
#' @param e_pred Variable name for propensity score predictions (if provided)
#' @param g_pred Variable name for censoring predictions (if provided)
#' @param g_method Statistical technique used to run the missingness model
#' @param g_covariates List containing the names of the variables to be input into the missingness model, excluding exposure # nolint
#' @param g_SL_lib Library to be used in super learner if selected for missingness model # nolint
#' @param pse_method Statistical technique used to run the pseudo outcome model
#' @param pse_covariates List containing the names of the variables to be input into the pseudo outcome model # nolint
#' @param pse_SL_lib Library to be used in super learner if selected for pseudo outcome model # nolint
#' @param imp_covariates Covariates to be used in SL imputation model if SL imputation used # nolint
#' @param imp_SL_lib SL libaray for imputation model if SL imputation used
#' @param newdata New data to create predictions for

#' @return A list containing: CATE estimates, a dataset used to train the learner, dataset containing all  # nolint
#'         pseudo-outcome predictions (if splits 3)


DR_learner <- function(
  analysis = c("Complete case",
               "Available case",
               "SL imputation",
               "mDR-learner"),
  data,
  id,
  outcome,
  exposure,
  outcome_observed_indicator = "None",
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
  nuisance_estimates_input = 0,
  e_pred = NA,
  o_0_pred = NA,
  o_1_pred = NA,
  g_pred = NA,
  pse_method = c("Parametric","Random forest","Super Learner"),
  pse_covariates,
  pse_SL_lib,
  imp_covariates = c(),
  imp_SL_lib,
  CI = FALSE,
  num_boot = 200,
  Para_CI_sim = FALSE,
  newdata
){
  if (analysis == "mDR-learner"){
    learner <- "mDR-learner"  
  }
  else {
    learner <- "DR-learner"
  }
  #-----------------------#
  #--- Data management ---#
  #-----------------------#
  
  clean_data <- data_manage_1tp(data = data,
                                learner = learner,
                                analysis_type = analysis,
                                nuisance_estimates_input = nuisance_estimates_input,
                                id = id,
                                outcome = outcome,
                                exposure = exposure,
                                outcome_observed_indicator = outcome_observed_indicator,
                                e_covariates = e_covariates,
                                out_covariates = out_covariates,
                                g_covariates = g_covariates,
                                pse_covariates = pse_covariates,
                                imp_covariates = imp_covariates,
                                o_0_pred = o_0_pred,
                                o_1_pred = o_1_pred,
                                e_pred = e_pred,
                                g_pred = g_pred,
                                newdata = newdata)
  
  #-------------------------#
  #--- Imputing outcomes ---#
  #-------------------------#
  if (analysis == "SL imputation" & nuisance_estimates_input == 0){
    analysis_data <- nuis_mod(model = "Imputation",
                              data = clean_data$data,
                              covariates = imp_covariates,
                              SL_lib = imp_SL_lib,
                              Y_bin = clean_data$Y_bin,
                              Y_cont = clean_data$Y_cont)
  }
  
  
  #----------------------#
  #--- Non imputation ---#
  #----------------------#
  
  if (analysis == "Complete case"){
    analysis_data <- clean_data$data
    analysis_data <- subset(analysis_data,is.na(analysis_data$Y)==0)
  }
  if (analysis == "Available case" | analysis == "mDR-learner"){
    analysis_data <- clean_data$data
  }


  #------------------------------------------------------#
  #--- Running nuisance models & pseudo outcome model ---#
  #------------------------------------------------------#

  tryCatch(
    {
      #Checking number of splits is correct
      if (splits != 1 & splits != 10){
        stop("Number of splits not compatible")
      }

      #Creating splits
      analysis_data$s <- rep(1:length(analysis_data$Y),1) %% splits
    },
    #if an error occurs, tell me the error
    error=function(e) {
      stop('An error occured when splitting the data')
      print(e)
    }
  )

  #--- Iterating over each split (cross-fitting) ---#
  all_output <- list()

  for (i in 0:(splits-1)){
    if (nuisance_estimates_input == 0){
      #--- Collecting data for training models ---#
      tryCatch(
        {
          #Data for propensity score and outcome models
          e_data <- analysis_data
          o_data <- analysis_data
          if (analysis == "Available case" | analysis == "mDR-learner" | analysis == "IPCW"){
            o_data <- subset(o_data,is.na(o_data$Y) == 0)  
          }
          if (analysis == "mDR-learner"){
            g_data <- analysis_data
          }
          if (splits == 1){
            e_data <- subset(e_data,e_data$s == i)
            o_data <- subset(o_data,o_data$s == i)
            if (analysis == "mDR-learner"){
              g_data <- subset(g_data,g_data$s == i)
            }
          }
          else if (splits == 10){
            e_data <- subset(e_data,e_data$s != i)
            o_data <- subset(o_data,o_data$s != i)
            if (analysis == "mDR-learner"){
              g_data <- subset(g_data,g_data$s != i)
            }
          }
        },
        #if an error occurs, tell me the error
        error=function(e) {
          stop(paste("An error occured when collecting data for training nuisance models in split ",i,sep=""))
          print(e)
        }
      )

      #--- Collecting data to obtain nuisance model predictions for ---#
      tryCatch(
        {
          #Creating dataset for pseudo outcome model
          po_e_data <- subset(analysis_data, select = c(e_covariates,"s"))
          po_e_data <- subset(po_e_data, po_e_data$s == i)
          po_e_data <- subset(po_e_data, select = -c(s))
          po_o_data <- subset(analysis_data,select = c(out_covariates,"s"))
          po_o_data <- subset(po_o_data, po_o_data$s == i)
          po_o_data <- subset(po_o_data, select = -c(s))
          if (analysis == "mDR-learner"){
            po_g_data <- subset(analysis_data,select = c("A",g_covariates,"s"))
            po_g_data <- subset(po_g_data, po_g_data$s == i)
            po_g_data <- subset(po_g_data, select = -c(s))
          }
          po_data <- subset(analysis_data,select = c("ID","Y","A","G",pse_covariates,"s"))
          po_data <- subset(po_data,po_data$s == i)
        },
        #if an error occurs, tell me the error
        error=function(e) {
          stop(paste("An error occured when collecting data for nuisance models predictions in split ",i,sep=""))
          print(e)
        }
      )

      #--- Running nuisance models & obtaining predictions ---#
      #Outcome models
      tryCatch(
        {
          outcome_models <- nuis_mod(model = "Outcome",
                                     data = o_data,
                                     method = out_method,
                                     covariates = out_covariates,
                                     SL_lib = out_SL_lib,
                                     Y_bin = clean_data$Y_bin,
                                     Y_cont = clean_data$Y_cont,
                                     pred_data = po_o_data)
        },
        #if an error occurs, tell me the error
        error=function(e) {
          stop(paste("An error occured in outcome model function in split ",i,sep=""))
          print(e)
        }
      )

      #Propensity score model
      tryCatch(
        {
          PS_model <- nuis_mod(model = "Propensity score",
                                     data = e_data,
                                     method = e_method,
                                     covariates = e_covariates,
                                     SL_lib = e_SL_lib,
                                     pred_data = po_e_data)
        },
        # if an error occurs, tell me the error
        error=function(e) {
          stop(paste("An error occured in propensity score model function in split ",i,sep=""))
          print(e)
        }
      )

      if (analysis == "mDR-learner"){
        #Censoring model
        tryCatch(
          {
            cen_model <- nuis_mod(model = "Censoring",
                                  data = g_data,
                                  method = g_method,
                                  covariates = g_covariates,
                                  SL_lib = g_SL_lib,
                                  pred_data = po_g_data)
          },
          #if an error occurs, tell me the error
          error=function(e) {
            stop(paste("An error occured in censoring model function in split ",i,sep=""))
            print(e)
          }
        )
      }

      #--- Collecting nuisance model predictions ---#
      tryCatch(
        {
          po_data <- cbind(po_data,o_1_pred = outcome_models$o_mod_pred_1)
          po_data <- cbind(po_data,o_0_pred = outcome_models$o_mod_pred_0)
          if (e_method == "Random forest"){
            po_data <- cbind(po_data,e_pred = PS_model$e_pred$predictions)
          }
          else {
            po_data <- cbind(po_data,e_pred = PS_model$e_pred)
          }
          if (analysis == "mDR-learner"){
            if (g_method == "Random forest"){
              po_data <- cbind(po_data,g_pred = cen_model$g_pred$predictions)
            }
            else {
              po_data <- cbind(po_data,g_pred = cen_model$g_pred)
            }
          }
        },
        #if an error occurs, tell me the error
        error=function(e) {
          stop(paste("An error occured when outcome model function in split ",i,sep=""))
          print(e)
        }
      )

    }
    else if (nuisance_estimates_input == 1){
      if (splits == 1 | splits == 10){
        po_data <- analysis_data
        po_data <- subset(po_data,po_data$s == i)
      }
    }


    #--- Calculating pseudo-outcomes ---#
    tryCatch(
      {

        #First setting missing values to 99
        po_data <- po_data %>% mutate_if(is.numeric, function(x) ifelse(is.na(x), 999, x))

        #Calculating pseudo outcome
        if (analysis == "mDR-learner"){
          po_data$pse_Y <- ((po_data$A - po_data$e_pred)/(po_data$e_pred*(1-po_data$e_pred))) *
            (po_data$G/po_data$g_pred) *
            (po_data$Y - (po_data$A*po_data$o_1_pred + (1-po_data$A) * po_data$o_0_pred)) +
            po_data$o_1_pred - po_data$o_0_pred
        }
        else {
          po_data$pse_Y <- ((po_data$A - po_data$e_pred)/(po_data$e_pred*(1-po_data$e_pred))) *
            po_data$G *
            (po_data$Y - (po_data$A*po_data$o_1_pred +(1-po_data$A)*po_data$o_0_pred)) +
            po_data$o_1_pred - po_data$o_0_pred
        }
      },
      #if an error occurs, tell me the error
      error=function(e) {
        stop(paste("An error occured when generating the pseudo outcome in split ",i,sep=""))
        print(e)
      }
    )

    
    #--- Collecting full test data with pseudo-outcomes ---#
    if (i==0){
      po_data_all <- po_data
    }
    else {
      po_data_all <- rbind(po_data_all,po_data)
    }
  }
  
  if (splits == 1 | splits == 10){
    #--- Gaining estimates from all data estimates ---#
    tryCatch(
      {
        pse_model <- nuis_mod(model = "Pseudo outcome",
                              data = po_data_all,
                              method = pse_method,
                              covariates = pse_covariates,
                              SL_lib = pse_SL_lib,
                              pred_data = newdata)
      },
      #if an error occurs, tell me the error
      error=function(e) {
        stop(paste("An error occured when fitting the pseudo outcome model in split ",i,sep=""))
        print(e)
      }
    )


    #--- Gaining CI's ---#
    # Option 1 - Half sample bootstrapping (If using RF in final stage)
    if (CI == TRUE & pse_method == "Random forest"){
      pse_n_rows <- nrow(po_data_all)
      for (i in 1:num_boot){
        # Randomly sample half the rows
        set.seed(596967 + i)  # Set seed for reproducibility
        random_indices <- sample(1:pse_n_rows, size = ceiling(pse_n_rows/2), replace = FALSE)
        half_sample <- po_data_all[random_indices, ]
        half_sample <- half_sample[order(half_sample$ID), ]

        #Running final stage model
        tuned_parameters <- pse_model$po_mod$tunable.params
        tryCatch(
          {
            pse_model_hs <- nuis_mod(model = "Pseudo outcome - CI",
                                     data = half_sample,
                                     method = pse_method,
                                     covariates = pse_covariates,
                                     SL_lib = pse_SL_lib,
                                     pred_data = newdata,
                                     CI_tuned_params = tuned_parameters)

            half_sample_est <- pse_model_hs
          },
          #if an error occurs, tell me the error
          error=function(e) {
            stop(paste("An error occured when fitting half sample pseudo outcome model in split ",i,sep=""))
            print(e)
          }
        )

        #Creating R and storing
        full_sample_est <- pse_model$po_pred

        R <- as.numeric(full_sample_est$predictions - half_sample_est)

        if (i == 1){
          R_data <- as.data.frame(R)
        }
        else {
          R_data <- cbind(R_data,R)
        }
      }

      #Gaining variance of R per person
      CI_n_rows <- nrow(pse_model$po_pred)
      for (i in 1:CI_n_rows){
        temp <- unlist(R_data[i,])
        var <- var(temp)
        SE <- sqrt(var)

        if (i == 1){
          var_list <- var
          SE_list <- SE
        }
        else {
          var_list <- c(var_list,var)
          SE_list <- c(SE_list,SE)
        }
      }

      #Creating normalised matrix
      normalized <- abs(R_data)/(sqrt(var_list))

      #Identifying column maxs
      CI_n_cols <- ncol(normalized)
      for (i in 1:CI_n_cols){
        temp <- unlist(normalized[,i])
        colmax <- max(temp)

        if (i == 1){
          colmax_list <- colmax
        }
        else {
          colmax_list <- c(colmax_list,colmax)
        }
      }

      #Creating S-star
      S_star <- quantile(colmax_list, 0.95)

      LCI <-  pse_model$po_pred$predictions - sqrt(var_list)*S_star
      UCI <-  pse_model$po_pred$predictions + sqrt(var_list)*S_star
    }
    else if (CI == TRUE & pse_method == "Parametric" & Para_CI_sim == FALSE){
      # Extract design matrix from pseudo outcome data
      X <- as.matrix(cbind(1, po_data_all[, pse_covariates]))
      y <- po_data_all$pse_Y
      
      # Calculate Huber-White sandwich estimator covariance matrix
      hw_cov_mat <- sandwich_est(X = X, y = y, model = pse_model$po_mod)
      
      # Create design matrix for new data predictions
      newdata_X <- as.matrix(cbind(1, newdata[, pse_covariates]))
      
      # Calculate standard errors for predictions
      SE_list <- get_se(hw_est = hw_cov_mat, pred.grid = newdata_X)
      
      # Calculate 95% confidence intervals (1.96 * SE for 95% CI)
      LCI <- pse_model$po_pred[,1] - 1.96 * SE_list
      UCI <- pse_model$po_pred[,1] + 1.96 * SE_list
    }
    else if (CI == TRUE & pse_method == "Parametric" & Para_CI_sim == TRUE){
      # # Extract design matrix from pseudo outcome data
      # X <- as.matrix(cbind(1, po_data_all[, pse_covariates]))
      # y <- po_data_all$pse_Y
      
      # # Calculate Huber-White sandwich estimator covariance matrix
      # hw_cov_mat <- sandwich_est(X = X, y = y, model = pse_model$po_mod)
      
      # # Create design matrix for new data predictions
      # newdata_X <- as.matrix(cbind(1, newdata[, pse_covariates]))
      
      # # Calculate standard errors for predictions
      # SE_list <- get_se(hw_est = hw_cov_mat, pred.grid = newdata_X)
      
      # # Calculate 95% confidence intervals (1.96 * SE for 95% CI)
      # LCI <- pse_model$po_pred[,1] - 1.96 * SE_list
      # UCI <- pse_model$po_pred[,1] + 1.96 * SE_list
    }
    else if (CI == TRUE & pse_method == "HAL"){
      X <- as.matrix(cbind(1, po_data_all[, pse_covariates]))
      y <- po_data_all$pse_Y
      newdata_X <- as.matrix(cbind(1, newdata[, pse_covariates]))

      se_dr <- IC_based_se(X = X,
                           Y = y,
                           hal_fit = pse_model$po_mod,
                           eval_points = newdata_X,
                           family = "gaussian")

      CATE_est   <- pse_model$po_pred
      LCI <- CATE_est - 1.96 * se_dr$se
      UCI <- CATE_est + 1.96 * se_dr$se
    }
    else if (CI == TRUE & (pse_method != "Random forest" | pse_method != "Parametric" | pse_method != "HAL")){ # nolint # nolint # nolint # nolint
      return("Inappropriate pseudo-outcome regression 
              method for obtaining CI's")
    }
  }

  #-----------------------------#
  #--- Returning information ---#
  #-----------------------------#
  if (splits == 1 | splits == 10){
    if (CI != TRUE){
      output <- list(CATE_est = pse_model$po_pred,
                     data = po_data_all)
    }
    else if (CI == TRUE & pse_method == "Random forest"){
      output <- list(CATE_est = pse_model$po_pred,
                     CATE_LCI = LCI,
                     CATE_UCI = UCI,
                     data = po_data_all,
                     R_data = R_data,
                     SE_list = SE_list,
                     var_list = var_list,
                     S_star = S_star,
                     normalized = normalized,
                     colmax_list=colmax_list)
    }
    else if (CI == TRUE & pse_method == "Parametric"){
      output <- list(CATE_est = pse_model$po_pred,
                     CATE_LCI = LCI,
                     CATE_UCI = UCI,
                     data = po_data_all,
                     SE_list = SE_list,
                     hw_cov_mat = hw_cov_mat)
    }
    else if (CI == TRUE & pse_method == "HAL"){
      output <- list(CATE_est = pse_model$po_pred,
                     CATE_LCI = LCI,
                     CATE_UCI = UCI,
                     data = po_data_all,
                     se_dr = se_dr)
    }
  }

  return(output)
}


###############################################################

load("C:/Users/Matthew/OneDrive - University College London/Documents/Projects/Missing_outcomes/mDR-learner_mEP-learner/data/ACTG175_data.RData")

#--- Formatting variables ---#
ACTG175_data$treat <- as.numeric(ACTG175_data$treat)
ACTG175_data$r <- as.numeric(ACTG175_data$r)
ACTG175_data$cd496 <- as.numeric(ACTG175_data$cd496)

#--- Defining covariates to be input into each model ---#
out_cov_list <- c("age","wtkg","karnof","cd40","cd80","gender","hemo","homo","symptom","race","drugs","str2")
imp_cov_list <- c("age","wtkg","karnof","cd40","cd80","gender","hemo","homo","symptom","race","drugs","str2")
ps_cov_list <- c("age","wtkg","karnof","cd40","cd80","gender","hemo","homo","symptom","race","drugs","str2")
pse1_cov_list <- c("age")
pse2_cov_list <- c("age","wtkg","karnof","cd40","cd80","gender","hemo","homo","symptom","race","drugs","str2")


#--- Creating learners for SL library's ---#
#LASSO & elastic net
nlambda_seq = c(50,100,250)
alpha_seq <- c(0.5,1)
usemin_seq <- c(FALSE,TRUE)
para_learners = create.Learner("SL.glmnet", tune = list(nlambda = nlambda_seq,alpha = alpha_seq,useMin = usemin_seq))
para_learners

mtry_seq6 <-  floor(sqrt(6) * c(0.5, 1))
min_node_seq <- c(10,20,50)
rf_learners6 = create.Learner("SL.ranger", tune = list(mtry = mtry_seq6, min.node.size = min_node_seq))
rf_learners6

#Nnet (single layer neural nets)
size_seq <- c(1,2,5)
nnet_learners <- create.Learner("SL.nnet",tune = list(size = size_seq))

#SVM (Support vector machine)
nu_seq <- c(1)
type_seq <- c("C-classification")
svm_learners = create.Learner("SL.svm",tune = list(type.class = type_seq))

#KernelKnn
K_seq <- c(5,10,20)
h_seq <- c(0.01,0.05,0.1,0.25)
KernelKnn_learners <- create.Learner("SL.kernelKnn",tune = list(k = K_seq, h = h_seq))

#Boosting
depth_seq <- c(2,4,8)
shrink_seq <- c(0.05,0.1,0.3)
minobs_seq <- c(10,20)
boost_learners = create.Learner("SL.xgboost", tune = list(minobspernode=minobs_seq, max_depth = depth_seq, shrinkage = shrink_seq))
boost_learners


#--- Creating SL libraries ---#
#Outcome models - Reduced
out_lib <- c("SL.mean",
             "SL.lm")#,
            #  "SL.glmnet_8", "SL.glmnet_9",
            #  "SL.glmnet_11", "SL.glmnet_12",
            #  "SL.ranger_1","SL.ranger_2","SL.ranger_3",
            #  "SL.ranger_4","SL.ranger_5","SL.ranger_6",
            #  "SL.nnet_1","SL.nnet_2","SL.nnet_3",
            #  "SL.svm_1",
            #  "SL.kernelKnn_4",
            #  "SL.kernelKnn_10")


#Imputation models - Reduced
imp_lib <- c("SL.mean",
             "SL.glm")#,
            #  "SL.glmnet_8", "SL.glmnet_9",
            #  "SL.glmnet_11", "SL.glmnet_12",
            #  "SL.ranger_1","SL.ranger_2","SL.ranger_3",
            #  "SL.ranger_4","SL.ranger_5","SL.ranger_6",
            #  "SL.nnet_1","SL.nnet_2","SL.nnet_3",
            #  "SL.svm_1",
            #  "SL.kernelKnn_4",
            #  "SL.kernelKnn_10")


#Propensity score models reduced
e_lib <- c("SL.mean",
           "SL.glm")#,
          #  "SL.glmnet_8", "SL.glmnet_9",
          #  "SL.glmnet_11", "SL.glmnet_12",
          #  "SL.ranger_1","SL.ranger_2","SL.ranger_3",
          #  "SL.ranger_4","SL.ranger_5","SL.ranger_6",
          #  "SL.nnet_1","SL.nnet_2","SL.nnet_3",
          #  "SL.svm_1",
          #  "SL.kernelKnn_4","SL.kernelKnn_10")


#Pseudo outcome model - Single covariate - Reduced 
pse_lib <- c("SL.mean",
              "SL.lm")#,
              # "SL.ranger_1","SL.ranger_3","SL.ranger_5",
              # "SL.nnet_1","SL.nnet_2","SL.nnet_3",
              # "SL.svm_1",
              # "SL.kernelKnn_4",
              # "SL.kernelKnn_10")


check <- ACTG175_data
check_test <- ACTG175_data

source("C:/Users/Matthew/OneDrive - University College London/Documents/Projects/Missing_outcomes/mDR-learner_mEP-learner/src/Data_management_1tp.R")
source("C:/Users/Matthew/OneDrive - University College London/Documents/Projects/Missing_outcomes/mDR-learner_mEP-learner/src/nuisance_models.R")
source("C:/Users/Matthew/OneDrive - University College London/Documents/Projects/Missing_outcomes/mDR-learner_mEP-learner/src/CI_functions.r")

#Example
DR_check <- DR_learner(analysis = "mDR-learner",
                       data = ACTG175_data,
                       id = "pidnum",
                       outcome = "cd496",
                       exposure = "treat",
                       outcome_observed_indicator = "r",
                       splits = 1,
                       e_method = "Super learner",
                       e_covariates = ps_cov_list,
                       e_SL_lib = e_lib,
                       out_method = "Super learner",
                       out_covariates = out_cov_list,
                       out_SL_lib = out_lib,
                       g_method = "Super learner",
                       g_covariates = imp_cov_list,
                       g_SL_lib = imp_lib,
                       imp_covariates = imp_cov_list,
                       imp_SL_lib = imp_lib,
                       pse_method = "HAL",
                       pse_covariates = pse1_cov_list,
                       pse_SL_lib = pse_lib,
                       newdata = ACTG175_data,
                       CI = TRUE,
                       num_boot = 100)

# DR_check <- DR_learner(analysis = "Complete case",
#                        data = check,
#                        id = "ID",
#                        outcome = "Y",
#                        exposure = "A",
#                        outcome_observed_indicator = "G_obs",
#                        splits = 3,
#                        nuisance_estimates_input = 1,
#                        o_0_pred = "Y.0_prob_true",
#                        o_1_pred = "Y.1_prob_true",
#                        e_pred = "prop_score",
#                        pse_method = "Parametric",
#                        pse_covariates = c("X1"),
#                        pse_SL_lib = c("SL.lm"),
#                        newdata = check)

# "SL.glmnet_8", "SL.glmnet_9",
# "SL.glmnet_11", "SL.glmnet_12",
# "SL.ranger_1","SL.ranger_2","SL.ranger_3",
# "SL.ranger_4","SL.ranger_5","SL.ranger_6",
# "SL.nnet_1","SL.nnet_2","SL.nnet_3",
# "SL.svm_1",
# "SL.kernelKnn_4","SL.kernelKnn_10"),
