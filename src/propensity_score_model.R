#######################################################################################
# Script: Propensity score model function
# Date: 21/03/24
# Author: Matt Pryce 
# Notes: - Used in each learner to run propensity score model
#        - Runs parametric models, RF or SL 
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


PS_mod <- function(data,
                   id,
                   outcome,
                   exposure,
                   e_method = c("Parametric","Random forest","Super Learner"),
                   e_covariates,
                   e_SL_lib,
                   e_SL_strat = TRUE,
                   nuisance_estimates_input = 0,
                   e_pred = NA,
                   pred_data
){
  
  #----------------------------------------#
  #--- Creating data for outcome models ---#
  #----------------------------------------#
  tryCatch( 
    {
      e_data <- as.data.frame(subset(data,select = c(e_covariates,"A","s")))
      A_e <- e_data$A
    },
    error=function(e) {
      stop('An error occured when creating analysis data')
      print(e)
    }
  )
  
  
  #----------------------#
  #--- Running models ---#
  #----------------------#

  tryCatch(
    {
      if (e_method == "Random forest"){
        e_X <- as.matrix(subset(e_data, select = e_covariates))
        e_mod <- regression_forest(e_X, e_data$A, honesty = FALSE,tune.parameters = "all")
      }
      else if (e_method == "Parametric"){
        e_fit_data <- subset(e_data,select = -c(s))
        e_mod <- glm(A ~ . , data = e_fit_data, family = binomial())
      }
      else if (e_method == "Super learner"){
        A_sums <- table(e_data$A)
        cv_folds <- min(10,A_sums[1],A_sums[2])
        e_mod <- SuperLearner(Y = e_data$A, X = data.frame(subset(e_data, select = e_covariates)),
                              method = "method.NNLS",
                              family = binomial(),
                              cvControl = list(V = cv_folds, stratifyCV=e_SL_strat),
                              SL.library = e_SL_lib)
      }
    },
    error=function(e) {
      stop('An error occured when running propensity score model')
      print(e)
    }
  )


  #-----------------------------------#
  #--- Obtaining model predictions ---#
  #-----------------------------------#
  tryCatch(
    {
      if (e_method == "Random forest"){
        e_pred <- predict(e_mod, pred_data)
      }
      else if (e_method == "Parametric"){
        e_pred <- predict(e_mod, as.data.frame(pred_data), type = "response")
      }
      else if (e_method == "Super learner"){
        e_pred <- predict(e_mod, as.data.frame(pred_data))$pred
      }
    },
    #if an error occurs, tell me the error
    error=function(e) {
      stop('An error occured when drawning predictions from propensity score model')
      print(e)
    }
  )


#-----------------------------#
#--- Returning information ---#
#-----------------------------#
output <- list(e_pred = e_pred,
               e_mod = e_mod)
  
  return(output)
}
