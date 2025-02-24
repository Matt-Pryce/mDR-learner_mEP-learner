#######################################################################################
# Script: IPTW-IPCW-Learner function
# Date: 20/02_25
# Author: Matt Pryce 
# Notes: IPTW-IPCW learner function for single time point setting
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
library(mice)

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

#' @param data The data frame containing all required information
#' @param id Identification for individuals
#' @param outcome The name of the outcome of interest
#' @param exposure The name of the exposure of interest
#' @param outcome_observed_indicator Indicator identifying when the outcome variable is observed (1=observed, 0=missing)
#' @param splits Number of splitsu for cross-fitting (Variations allowed: 1, 3, 10)
#' @param e_method Statistical technique used to run the propensity score model
#' @param e_covariates  List containing the names of the variables to be input into the propensity score model
#' @param e_SL_lib Library to be used in super learner if selected
#' @param g_method Statistical technique used to run the missingness model
#' @param g_covariates List containing the names of the variables to be input into the missingness model, excluding exposure
#' @param g_SL_lib Library to be used in super learner if selected for missingness model
#' @param nuisance_estimates_input Indicator for whether nuisance estimates provided
#' @param e_pred Variable name for propensity score predictions (if provided)
#' @param g_pred Variable name for censoring predictions (if provided)
#' @param pse_method Statistical technique used to run the pseudo outcome model
#' @param pse_covariates List containing the names of the variables to be input into the pseudo outcome model
#' @param pse_SL_lib Library to be used in super learner if selected for pseudo outcome model
#' @param newdata New data to create predictions for

#' @return A list containing: CATE estimates, a dataset used to train the learner, dataset containing all 


IPTW_IPCW_learner <- function(data,
                              id,
                              outcome,
                              exposure,
                              outcome_observed_indicator = "None",
                              splits = c(1,10),
                              e_method = c("Parametric","Random forest","Super Learner"),
                              e_covariates,
                              e_SL_lib,
                              g_method = c("Parametric","Random forest","Super Learner"),
                              g_covariates = c(),
                              g_SL_lib,
                              nuisance_estimates_input = 0,
                              e_pred = NA,
                              g_pred = NA,
                              pse_method = c("Parametric","Random forest","Super Learner"),
                              pse_covariates,
                              pse_SL_lib,
                              rf_CI = FALSE,
                              num_boot = 200,
                              newdata
){

  #-----------------------#
  #--- Data management ---#
  #-----------------------#
  
  clean_data <- data_manage_1tp(data = data,
                                learner = "IPTW-IPCW",
                                analysis_type = "IPTW-IPCW",
                                nuisance_estimates_input = nuisance_estimates_input,
                                id = id,
                                outcome = outcome,
                                exposure = exposure,
                                outcome_observed_indicator = outcome_observed_indicator,
                                e_covariates = e_covariates,
                                g_covariates = g_covariates,
                                pse_covariates = pse_covariates,
                                e_pred = e_pred,
                                g_pred = g_pred,
                                newdata = newdata)
 
  analysis_data <- clean_data$data
  

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
          g_data <- analysis_data
          if (splits == 1){
            e_data <- subset(e_data,e_data$s == i)
            g_data <- subset(g_data,g_data$s == i)
          }
          else if (splits == 10){
            e_data <- subset(e_data,e_data$s != i)
            g_data <- subset(g_data,g_data$s != i)
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
          po_g_data <- subset(analysis_data,select = c("A",g_covariates,"s"))
          po_g_data <- subset(po_g_data, po_g_data$s == i)
          po_g_data <- as.matrix(subset(po_g_data, select = -c(s)))
          po_data <- subset(analysis_data,select = c("ID","Y","A","G",pse_covariates,"s"))
          po_data <- subset(po_data,po_data$s == i)
        },
        #if an error occurs, tell me the error
        error=function(e) {
          stop(paste("An error occured when collecting data for nuisance models predictions in split ",i,sep=""))
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

      # Censoring model
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

      #--- Collecting nuisance model predictions ---#
      tryCatch(
        {
          if (e_method == "Random forest"){
            po_data <- cbind(po_data,e_pred = PS_model$e_pred$predictions)
          }
          else {
            po_data <- cbind(po_data,e_pred = PS_model$e_pred)
          }
          if (g_method == "Random forest"){
            po_data <- cbind(po_data,g_pred = cen_model$g_pred$predictions)
          }
          else {
            po_data <- cbind(po_data,g_pred = cen_model$g_pred)
          }
        },
        #if an error occurs, tell me the error
        error=function(e) {
          stop(paste("An error occured when collecting nuisance model predictions in split ",i,sep=""))
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
        #Removing those with missing values 
        po_data <- subset(po_data,is.na(po_data$Y)==0)
        
        #Calculating pseudo outcome
        po_data$pse_Y <- ((po_data$A - po_data$e_pred)/(po_data$e_pred*(1-po_data$e_pred))) *
                         (po_data$G/po_data$g_pred) * po_data$Y 
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


  # #--- Gaining CI's ---#
  # if (rf_CI == TRUE & pse_method == "Random forest"){
  #   pse_n_rows <- nrow(po_data_all)
  #   for (i in 1:num_boot){
  #     # Randomly sample half the rows
  #     set.seed(596967 + i)  # Set seed for reproducibility
  #     random_indices <- sample(1:pse_n_rows, size = ceiling(pse_n_rows/2), replace = FALSE)
  #     half_sample <- po_data_all[random_indices, ]
  #     half_sample <- half_sample[order(half_sample$ID), ]
  #     
  #     #Running final stage model
  #     tuned_parameters <- pse_model$po_mod$tunable.params
  #     tryCatch(
  #       {
  #         pse_model_hs <- nuis_mod(model = "Pseudo outcome - CI",
  #                                  data = half_sample,
  #                                  method = pse_method,
  #                                  covariates = pse_covariates,
  #                                  SL_lib = pse_SL_lib,
  #                                  pred_data = newdata,
  #                                  CI_tuned_params = tuned_parameters)
  #         
  #         half_sample_est <- pse_model_hs
  #       },
  #       #if an error occurs, tell me the error
  #       error=function(e) {
  #         stop(paste("An error occured when fitting the pseudo outcome model in split ",i,sep=""))
  #         print(e)
  #       }
  #     )
  #     
  #     #Creating R and storing
  #     full_sample_est <- pse_model$po_pred
  #     
  #     R <- as.numeric(full_sample_est$predictions - half_sample_est)
  #     
  #     if (i == 1){
  #       R_data <- as.data.frame(R)
  #     }
  #     else {
  #       R_data <- cbind(R_data,R)
  #     }
  #   }
  #   
  #   #Gaining variance of R per person
  #   CI_n_rows <- nrow(pse_model$po_pred)
  #   for (i in 1:CI_n_rows){
  #     temp <- unlist(R_data[i,])
  #     var <- var(temp)
  #     SE <- sqrt(var)
  #     
  #     if (i == 1){
  #       var_list <- var
  #       SE_list <- SE
  #     }
  #     else {
  #       var_list <- c(var_list,var)
  #       SE_list <- c(SE_list,SE)
  #     }
  #   }
  #   
  #   #Creating normalised matrix
  #   normalized <- abs(R_data)/(sqrt(var_list))
  #   
  #   #Identifying column maxs
  #   CI_n_cols <- ncol(normalized)
  #   for (i in 1:CI_n_cols){
  #     temp <- unlist(R_data[,i])
  #     colmax <- max(temp)
  #     
  #     if (i == 1){
  #       colmax_list <- colmax
  #     }
  #     else {
  #       colmax_list <- c(colmax_list,colmax)
  #     }
  #   }
  #   
  #   #Creating S-star
  #   S_star <- quantile(colmax_list, 0.95)
  #   
  #   LCI <-  pse_model$po_pred$predictions - sqrt(var_list)*S_star
  #   UCI <-  pse_model$po_pred$predictions + sqrt(var_list)*S_star
  # }
  # else if (rf_CI == TRUE & pse_method != "Random forest"){
  #   return("Inappropriate pseudo-outcome regression method for obtaining CI's")
  # }
  # 
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
                     CATE_LCI = LCI,
                     CATE_UCI = UCI,
                     data = po_data_all)
    }
  }

  return(output)
}



###############################################################
# 
load("~/PhD/DR_Missing_Paper/Simulations/Results/Final_22_03_24/Scenario_1/Scenario_1_output1.RData")
check <- model_info_list$i$sim_data_train
check_test <- model_info_list$i$sim_data_test

#Example
IPTW_IPCW_check <- IPTW_IPCW_learner(data = check,
                                     id = "ID",
                                     outcome = "Y",
                                     exposure = "A",
                                     outcome_observed_indicator = "G_obs",
                                     splits = 1,
                                     e_covariates = c("X1","X2","X3","X4","X5","X6"),
                                     e_method = "Super learner",
                                     e_SL_lib = c("SL.mean",
                                                  "SL.lm"),
                                     g_covariates = c("X1","X2","X3","X4","X5","X6"),
                                     g_method = "Super learner",
                                     g_SL_lib = c("SL.mean",
                                                  "SL.glm"),
                                     pse_method = "Random forest",
                                     pse_covariates = c("X1","X2","X3","X4","X5","X6"),
                                     pse_SL_lib = c("SL.mean",
                                                    "SL.lm"),
                                     newdata = check_test,
                                     rf_CI = FALSE,
                                     num_boot = 2000)

