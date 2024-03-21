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


#' @param analysis Type of analysis to be run (e.g. complete case or outcome imputation) 
#' @param data The data frame containing all required information
#' @param id Identification for individuals
#' @param outcome The name of the outcome of interest
#' @param exposure The name of the exposure of interest
#' @param outcome_observed_indicator Indicator identifying when the outcome variable is observed (1=observed, 0=missing)
#' @param out_method Statistical technique used to run the outcome models
#' @param out_covariates  List containing the names of the variables to be input into each outcome model
#' @param out_SL_lib Library to be used in super learner if selected
#' @param out_SL_strat Indicator for if stratification should be used in the super learner CV (stratifies outcomes - binary only)
#' @param oracle If an oracle version of the T-learner is to be run
#' @param Y.0 If oracle=1, Y.0 value to be used in oracle model
#' @param Y.1 If oracle=1, Y.1 value to be used in oracle model
#' @param imp_covariates Covariates to be used in SL imputation model if SL imputation used
#' @param imp_SL_lib SL libaray for imputation model if SL imputation used
#' @param imp_SL_strat Whether SL CV folds are stratified in the imputation model if SL imputation used  
#' @param newdata New data to create predictions for

#' @return A dataset containing n CATE estimates (UPDATE) 


data_manage_1tp <- function(data,
                            learner,
                            analysis_type = "N/A",
                            nuisance_estimates_input,
                            id,
                            outcome,
                            exposure,
                            outcome_observed_indicator,
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
        if (learner == "DR-learner"){
          vars <- append(vars,e_covariates)
          vars <- append(vars,pse_covariates)
        }
        if (learner == "mDR-learner"){
          vars <- append(vars,e_covariates)
          vars <- append(vars,g_covariates)
          vars <- append(vars,pse_covariates)
        }
        if (analysis_type == "SL imputation"){   #If want to impute youself then set as outcome and run complete case 
          vars <- append(vars,imp_covariates)
        }
        vars <- vars[!duplicated(vars)]
      }
      
      if (nuisance_estimates_input == 1){
        vars <- append(vars,o_0_pred)
        vars <- append(vars,o_1_pred)
        
        if (learner == "DR-learner"){           #Cant run  imputation if imputing nuisance estimates (do as above)
          vars <- append(vars,pse_covariates)
          vars <- append(vars,e_pred)
        }
        if (learner == "mDR-learner"){           #Cant run  imputation if imputing nuisance estimates (do as above)
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
      if (nuisance_estimates_input == 1){
        names(data)[names(data) == o_0_pred] <- "o_0_pred"
        names(data)[names(data) == o_1_pred] <- "o_1_pred"
        if (learner == "DR-learner"){
          names(data)[names(data) == e_pred] <- "e_pred"
        }
        if (learner == "mDR-learner"){
          names(data)[names(data) == e_pred] <- "e_pred"
          names(data)[names(data) == g_pred] <- "e_pred"
        }
      }
    },
    error=function(e) {
      stop('An error occured when renaming variables')
      print(e)
    }
  )
  
  
  #-----------------------#
  #--- Format new data ---#   #If running t-learner can only make predictions of input data 
  #-----------------------#
  tryCatch( 
    {
      if (nuisance_estimates_input == 0 & learner == "T-learner"){
        new_data_vars <- c(id,out_covariates)

      }
      if (learner != "T-learner"){
        new_data_vars <- c(id,pse_covariates)
      }
      newdata <- subset(newdata,select=new_data_vars)
      names(newdata)[names(newdata) == id] <- "ID"
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
    },
    error=function(e) {
      stop('An error occured when formatting the new data')
      print(e)
    }
  )
  
  #-----------------------------#
  #--- Returning information ---#
  #-----------------------------#
  output <- list(data=data,
                 Y_bin = Y_bin,
                 Y_cont = Y_cont,
                 newdata=newdata)
  return(output)
}
  

