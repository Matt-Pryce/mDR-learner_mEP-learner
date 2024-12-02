#######################################################################################
# Script: DR-Learner function
# Date: 20/03/24
# Author: Matt Pryce 
# Notes: DR-learner function for single time point setting
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


DR_learner <- function(analysis = c("Complete case","Available case","SL imputation","IPCW","mDR-learner"),
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
                       rf_CI = FALSE,
                       num_boot = 200,
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
  
  
  #----------------------------------#
  #--- IPCW weighting on outcomes ---#    Only viable for continous outcomes atm
  #----------------------------------#
  if (analysis == "IPCW" & nuisance_estimates_input == 0){
    analysis_data <- nuis_mod(model = "IPCW",
                              data = clean_data$data,
                              method = g_method,
                              covariates = g_covariates,
                              SL_lib = g_SL_lib,
                              pred_data = clean_data,
                              Y_bin = clean_data$Y_bin,
                              Y_cont = clean_data$Y_cont)
  }
  
  
  #---------------------------#
  #--- Non imputation/IPCW ---#
  #---------------------------#
  
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
          po_e_data <- as.matrix(subset(po_e_data, select = -c(s)))
          po_o_data <- subset(analysis_data,select = c(out_covariates,"s"))
          po_o_data <- subset(po_o_data, po_o_data$s == i)
          po_o_data <- as.matrix(subset(po_o_data, select = -c(s)))
          if (analysis == "mDR-learner"){
            po_g_data <- subset(analysis_data,select = c("A",g_covariates,"s"))
            po_g_data <- subset(po_g_data, po_g_data$s == i)
            po_g_data <- as.matrix(subset(po_g_data, select = -c(s)))
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
    #Gaining estimates from all data estimates
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
    if (rf_CI == TRUE & pse_method == "Random forest"){
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
            stop(paste("An error occured when fitting the pseudo outcome model in split ",i,sep=""))
            print(e)
          }
        )

        #Creating R and storing
        full_sample_est <- pse_model$po_pred

        R <- full_sample_est - half_sample_est

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

        sqrt_n <- sqrt(num_boot)
        temp <-  sqrt_n * R_data[i,]
        var <- apply(temp, MARGIN = 1, FUN = var)
        SE <- sqrt(var)

        LCI <- pse_model$po_pred[i,] - (1/sqrt_n)*SE*1.96
        UCI <- pse_model$po_pred[i,] + (1/sqrt_n)*SE*1.96

        # temp <-  R_data[i,]
        # var <- apply(temp, MARGIN = 1, FUN = var)
        # SE <- sqrt(var)
        #
        # LCI <- pse_model$po_pred[i,] - SE*1.96
        # UCI <- pse_model$po_pred[i,] + SE*1.96

        if (i == 1){
          LCI_data <- LCI
          UCI_data <- UCI
        }
        else {
          LCI_data <- append(LCI_data,LCI)
          UCI_data <- append(UCI_data,UCI)
        }
      }
    }
    else if (rf_CI == TRUE & pse_method != "Random forest"){
      return("Inappropriate pseudo-outcome regression method for obtaining CI's")
    }
  }

  #-----------------------------#
  #--- Returning information ---#
  #-----------------------------#
  if (splits == 1 | splits == 10){
    if (rf_CI != TRUE){
      output <- list(CATE_est = pse_model$po_pred,
                     data = po_data_all)
    }
    else if (rf_CI == TRUE){
      output <- list(CATE_est = pse_model$po_pred,
                     R_data = R_data,
                     CATE_LCI = LCI_data,
                     CATE_UCI = UCI_data,
                     data = po_data_all)
    }
  }

  return(output)
}



###############################################################

load("~/PhD/DR_Missing_Paper/Simulations/Results/Final_22_03_24/Scenario_1/Scenario_1_output1.RData")
check <- model_info_list$i$sim_data_train
check_test <- model_info_list$i$sim_data_test

#Example
DR_check <- DR_learner(analysis = "mDR-learner",
                       data = check,
                       id = "ID",
                       outcome = "Y",
                       exposure = "A",
                       outcome_observed_indicator = "G_obs",
                       splits = 1,
                       e_covariates = c("X1","X2","X3","X4","X5","X6"),
                       e_method = "Super learner",
                       e_SL_lib = c("SL.mean",
                                    "SL.lm"),
                       out_method = "Super learner",
                       out_covariates = c("X1","X2","X3","X4","X5","X6"),
                       out_SL_lib = c("SL.mean",
                                      "SL.lm"),
                       g_covariates = c("X1","X2","X3","X4","X5","X6"),
                       g_method = "Super learner",
                       g_SL_lib = c("SL.mean",
                                    "SL.lm"),
                       imp_covariates = c("X1","X2","X3","X4","X5","X6"),
                       imp_SL_lib = c("SL.mean",
                                      "SL.lm"),
                       pse_method = "Random forest",
                       pse_covariates = c("X1","X2","X3","X4","X5","X6"),
                       pse_SL_lib = c("SL.mean",
                                      "SL.lm"),
                       newdata = check_test,
                       rf_CI = TRUE,
                       num_boot = 10)

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
