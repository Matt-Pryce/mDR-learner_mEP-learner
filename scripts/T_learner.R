#######################################################################################
# Script: T-Learner function
# Date: 20/03/24
# Author: Matt Pryce 
# Notes: T-learner function for single time point setting
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


#######################################
#--- For single time point setting ---#
#######################################

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


T_learner <- function(analysis = c("Complete case","SL imputation"),
                      data,
                      id,
                      outcome,
                      exposure,
                      outcome_observed_indicator = "None",
                      out_method = c("Parametric","Random forest","Super Learner"),
                      out_covariates,
                      out_SL_lib,
                      out_SL_strat = FALSE,
                      imp_covariates = c(),
                      imp_SL_lib,
                      imp_SL_strat = FALSE,
                      nuisance_estimates_input = 0,
                      o_0_pred = NA,
                      o_1_pred = NA,
                      newdata
){
  
  #-----------------------#
  #--- Data management ---#
  #-----------------------#
  
  clean_data <- data_manage_1tp(data = data,
                                learner = "T-learner",
                                analysis_type = analysis,
                                nuisance_estimates_input = nuisance_estimates_input,
                                id = id,
                                outcome = outcome,
                                exposure = exposure,
                                outcome_observed_indicator = outcome_observed_indicator,
                                out_covariates = out_covariates,
                                imp_covariates = imp_covariates,
                                o_0_pred = o_0_pred,
                                o_1_pred = o_1_pred,
                                newdata = newdata
  )
  
  
  #-------------------------#
  #--- Imputing outcomes ---#
  #-------------------------#
  if (analysis == "SL imputation" & nuisance_estimates_input == 0){
    analysis_data <- out_imp_1tp(data = clean_data$data,
                                 id = id,
                                 outcome = outcome,
                                 exposure = exposure,
                                 imp_covariates = imp_covariates,
                                 imp_SL_lib = imp_SL_lib,
                                 imp_SL_strat = FALSE,
                                 Y_bin = clean_data$Y_bin,
                                 Y_cont = clean_data$Y_cont)
  }
  else {
    analysis_data <- clean_data$data
    analysis_data <- subset(analysis_data,analysis_data$G==1) 
  }

  
  
  #------------------------------#
  #--- Running outcome models ---#
  #------------------------------#
  
  if (nuisance_estimates_input == 0){
    outcome_models <- out_mods(data = analysis_data,
                               id = id,
                               outcome = outcome,
                               exposure = exposure,
                               out_method = out_method,   
                               out_covariates = out_covariates,
                               out_SL_lib = out_SL_lib,
                               out_SL_strat = out_SL_strat,
                               Y_bin = clean_data$Y_bin,
                               Y_cont = clean_data$Y_cont,
                               nuisance_estimates_input = nuisance_estimates_input,
                               o_0_pred = o_0_pred,
                               o_1_pred = o_1_pred,
                               pred_data = newdata)
  }
  
  
  #-------------------------------#
  #--- Creating CATE estimates ---#
  #-------------------------------#
  if (nuisance_estimates_input == 0){
    newdata$CATE_est <- outcome_models$o_mod_pred_1 - outcome_models$o_mod_pred_0
  }
  else if (nuisance_estimates_input == 1){
    analysis_data$CATE_est <- analysis_data[["o_1_pred"]] - analysis_data[["o_0_pred"]]
  }
  
  #-----------------------------#
  #--- Returning information ---#
  #-----------------------------#
  if (nuisance_estimates_input == 0){
    output <- list(CATE_est=newdata$CATE_est,
                   outcome_models=outcome_models,
                   newdata=newdata)
  }
  else {
    output <- list(CATE_est=analysis_data$CATE_est,
                   newdata=analysis_data)
  }
  return(output)
}



###############################################################

# #Example 
# T_check <- T_learner(analysis = "SL imputation",
#                      data = check,
#                      id = "ID",
#                      outcome = "Y",
#                      exposure = "A",
#                      outcome_observed_indicator = "G_obs",
#                      out_method = "Super learner",
#                      out_covariates = c("X1","X2","X3"),
#                      out_SL_lib = c("SL.lm"),
#                      out_SL_strat = FALSE,
#                      imp_covariates = c("X3","X5","X6"),
#                      imp_SL_lib = c("SL.lm"),
#                      imp_SL_strat = FALSE,
#                      newdata = check)
# 
# T_check <- T_learner(analysis = "Complete case",
#                      data = check,
#                      id = "ID",
#                      outcome = "Y",
#                      exposure = "A",
#                      outcome_observed_indicator = "G_obs",
#                      nuisance_estimates_input = 1,
#                      o_0_pred = "Y.0_prob_true",
#                      o_1_pred = "Y.1_prob_true",
#                      newdata = check)


############################

#Notes on use:
#   - If want to fit outcome imputation outside function then run Complete case 
#     and specify outcome as imputed outcome 
#   - Won't run imputation if nuisance estimates are input and not estimated,
#     instead treats it as a complete case analysis
#   - If nuisance functions input, T-learner can only make predictions of input data 









