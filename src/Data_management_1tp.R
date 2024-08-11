#######################################################################################
# Script: Data management function for 1tp setting
# Date: 20/03/24
# Author: Matt Pryce 
# Notes: - Used when running any models for one time point (T-learner, DR-learner, 
#          mDR-learner, EP-learner)
#        - Checks the input data is correctly formatted. 
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

#' @param data The data frame containing all required information
#' @param leanrer Which learner is data management being run on 
#' @param analysis Type of analysis to be run (If not mDR, complete case or outcome imputation) 
#' @param id Identification for individuals
#' @param outcome The name of the outcome of interest
#' @param exposure The name of the exposure of interest
#' @param outcome_observed_indicator Indicator identifying when the outcome variable is observed (1=observed, 0=missing)
#' @param out_covariates  List containing the names of the variables to be input into each outcome model
#' @param e_covariates  List containing the names of the variables to be input into the propensity score model
#' @param g_covariates List containing the names of the variables to be input into the missingness model, excluding exposure
#' @param imp_covariates Covariates to be used in SL imputation model if SL imputation used
#' @param pse_covariates List containing the names of the variables to be input into the pseudo outcome model
#' @param nuisance_estimates_input Indicator for whether nuisance estimates provided
#' @param o_0_pred Variable name for unexposed outcome predictions (if provided)
#' @param o_1_pred Variable name for exposed outcome predictions (if provided)
#' @param e_pred Variable name for propensity score predictions (if provided)
#' @param g_pred Variable name for censoring predictions (if provided)
#' @param newdata New data to create predictions for

#' @return Cleaned data for training and testing, along with indicators for whether the outcome is binary or continuous 


data_manage_1tp <- function(data,
                            learner,
                            analysis_type = "N/A",
                            nuisance_estimates_input,
                            id,
                            outcome,
                            exposure,
                            outcome_observed_indicator,
                            CATE_est,
                            out_covariates,
                            e_covariates,
                            g_covariates,
                            imp_covariates,
                            pse_covariates,
                            e_pred,
                            o_0_pred,
                            o_1_pred,
                            g_pred,
                            newdata
){

  #---------------------------------------------------------------------#
  #--- Checking if an indicator for observing the outcome is present ---#
  #---------------------------------------------------------------------#
  tryCatch( 
    {
      missing_outcome_data <- 0
      if (outcome_observed_indicator != "None"){
        missing_outcome_data <- 1
      }
    },
    error=function(e) {
      stop('An error occured when identifying indicator for observing the outcome')
      print(e)
    }
  )
    
  
  #---------------------------------------------#
  #--- Checking and keeping appropriate data ---#
  #---------------------------------------------#
  tryCatch( 
    {
      if (missing_outcome_data == 1){
        vars <- c(id, outcome,exposure, outcome_observed_indicator)
      }
      else {
        vars <- c(id, outcome,exposure)
      }
      
      if (nuisance_estimates_input == 0){
        #For all analysis choices and both learners
        vars <- append(vars,out_covariates)
        if (learner == "DR-learner" | learner == "EP-learner"){
          vars <- append(vars,e_covariates)
          vars <- append(vars,pse_covariates)
        }
        if (learner == "mDR-learner" | learner == "mEP-learner"){
          vars <- append(vars,e_covariates)
          vars <- append(vars,g_covariates)
          vars <- append(vars,pse_covariates)
        }
        if (learner == "debiased_MSE"){
          vars <- append(vars,e_covariates)
          vars <- append(vars,g_covariates)
          vars <- append(vars,CATE_est)
        }
        if (learner == "T-learner" | learner == "DR-learner" | learner == "EP-learner"){
          if (analysis_type == "SL imputation"){
            vars <- append(vars,imp_covariates)
          }
          if (analysis_type == "IPCW"){
            vars <- append(vars,g_covariates)
          }
        }
        vars <- vars[!duplicated(vars)]
      }

      if (nuisance_estimates_input == 1){    #Cannot handle IPCW or imputation
        vars <- append(vars,o_0_pred)
        vars <- append(vars,o_1_pred)

        if (learner == "DR-learner" | learner == "EP-learner"){
          vars <- append(vars,pse_covariates)
          vars <- append(vars,e_pred)
        }
        if (learner == "mDR-learner" | learner == "mEP-learner"){
          vars <- append(vars,pse_covariates)
          vars <- append(vars,e_pred)
          vars <- append(vars,g_pred)
        }
        vars <- vars[!duplicated(vars)]
      }
      
      data <- subset(data,select = vars)
    },
    error=function(e) {
      stop('An error occured when selecting the appropriate variables')
      print(e)
    }
  )

  #------------------------#
  #--- Rename variables ---#
  #------------------------#
  tryCatch(
    {
      names(data)[names(data) == id] <- "ID"
      names(data)[names(data) == exposure] <- "A"
      names(data)[names(data) == outcome] <- "Y"
      if (missing_outcome_data == 1){
        names(data)[names(data) == outcome_observed_indicator] <- "G"
      }
      if (learner == "debiased_MSE"){
        names(data)[names(data) == CATE_est] <- "CATE_est"
      }
      if (nuisance_estimates_input == 1){  #Cannot handle IPCW or imputation 
        names(data)[names(data) == o_0_pred] <- "o_0_pred"
        names(data)[names(data) == o_1_pred] <- "o_1_pred"
        if (learner == "DR-learner" | learner == "EP-learner"){
          names(data)[names(data) == e_pred] <- "e_pred"
        }
        if (learner == "mDR-learner" | learner == "mEP-learner"){
          names(data)[names(data) == e_pred] <- "e_pred"
          names(data)[names(data) == g_pred] <- "g_pred"
        }
      }
    },
    error=function(e) {
      stop('An error occured when renaming variables')
      print(e)
    }
  )


  #-----------------------#
  #--- Format new data ---#
  #-----------------------#
  tryCatch(
    {
      if (nuisance_estimates_input == 0 & learner == "T-learner"){
        new_data_vars <- c(id,out_covariates)
        newdata <- subset(newdata,select=new_data_vars)
        names(newdata)[names(newdata) == id] <- "ID"
      }
      if (learner != "T-learner" & learner != "debiased_MSE"){
        new_data_vars <- c(id,pse_covariates)
        newdata <- subset(newdata,select=new_data_vars)
        names(newdata)[names(newdata) == id] <- "ID"
      }
    },
    error=function(e) {
      stop('An error occured when formatting the new data')
      print(e)
    }
  )


  #--------------------#
  #--- Logic checks ---#
  #--------------------#
  tryCatch(
    {
      #Checking if exposure is binary
      if (as.numeric(all(data$A %in% 0:1)) == 0){
        stop("Exposure must be binary")
      }
      #Checking in outcome is binary or continuous
      Y_comp <- na.omit(data$Y)
      Y_bin <- as.numeric(all(Y_comp %in% 0:1))
      Y_cont <- 0
      if (Y_bin == 0 & typeof(Y_comp) == "double"){
        Y_cont <- 1
      }

      if (learner == "debiased_MSE"){
        #Normalizing outcome and CATE estimates with min-max norm when outcome in continuous
        if (Y_bin == 0){
          min_Y <- min(data$Y,na.rm = T)
          max_Y <- max(data$Y,na.rm = T)

          #Outcome
          data$Y_norm <- (data$Y - min_Y)/(max_Y - min_Y)

          #CATE estimates
          data$CATE_est_norm <- (data$CATE_est - min_Y)/(max_Y - min_Y)
        }
      }
    },
    error=function(e) {
      stop('An error occured when formatting the new data')
      print(e)
    }
  )

  #-----------------------------#
  #--- Returning information ---#
  #-----------------------------#
  if (learner != "debiased_MSE"){
    output <- list(data=data,
                   Y_bin = Y_bin,
                   Y_cont = Y_cont,
                   newdata=newdata)
  }
  else {
    output <- list(data=data,
                   Y_bin = Y_bin,
                   Y_cont = Y_cont,
                   max_Y = max_Y,
                   min_Y = min_Y)
  }

  return(output)
}
  

