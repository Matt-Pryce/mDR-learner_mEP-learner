#######################################################################################
# Script: Censoring model function
# Date: 22/03/24
# Author: Matt Pryce 
# Notes: - Used in each learner to run censoring model
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


#' @param data The data frame containing all required information
#' @param id Identification for individuals
#' @param outcome The name of the outcome of interest
#' @param exposure The name of the exposure of interest
#' @param g_method Statistical technique used to run the missingness model
#' @param g_covariates List containing the names of the variables to be input into the missingness model, excluding exposure
#' @param g_SL_lib Library to be used in super learner if selected for missingness model
#' @param g_SL_strat If missingness model is estimated using SL, indicates whether stratification should be used in the super learner CV 
#' @param pred_data New data to create predictions for

#' @return Censoring model and predictions from it for pred_data


cen_mod <- function(data,
                   id,
                   outcome,
                   exposure,
                   g_method = c("Parametric","Random forest","Super Learner"),
                   g_covariates,
                   g_SL_lib,
                   g_SL_strat = TRUE,
                   pred_data
){
  
  #----------------------------------------#
  #--- Creating data for outcome models ---#
  #----------------------------------------#
  tryCatch( 
    {
      g_data <- as.data.frame(subset(data,select = c("G",g_covariates,"A","s")))
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
      if (g_method == "Random forest"){
        g_X <- as.matrix(subset(g_data, select = c(g_covariates,"A")))
        g_mod <- regression_forest(g_X, g_data$G, honesty = TRUE,tune.parameters = "all")
      }
      #Running if parametric model selected
      else if (g_method == "Parametric"){
        g_fit_data <- subset(g_data,select = -c(s))
        
        #Running first outcome models
        g_mod <- glm(G ~ . , data = g_fit_data, family = binomial())
      }
      else if (g_method == "Super learner"){
        G_sums <- table(g_data$G)
        cv_folds <- min(10,G_sums[1],G_sums[2])
        g_mod <- SuperLearner(Y = g_data$G, X = data.frame(subset(g_data, select = c(g_covariates,"A"))),
                              method = "method.NNLS",
                              family = binomial(),
                              cvControl = list(V = cv_folds, stratifyCV=g_SL_strat),
                              SL.library = g_SL_lib)
      }
    },
    error=function(e) {
      stop('An error occured when running censoring model')
      print(e)
    }
  )
  
  
  #-----------------------------------#
  #--- Obtaining model predictions ---#
  #-----------------------------------#
  tryCatch(
    {
      if (g_method == "Random forest"){
        g_pred <- predict(g_mod, pred_data)
      }
      else if (g_method == "Parametric"){
        g_pred <- predict(g_mod, as.data.frame(pred_data), type = "response")
      }
      else if (g_method == "Super learner"){
        g_pred <- predict(g_mod, as.data.frame(pred_data))$pred
      }
    },
    #if an error occurs, tell me the error
    error=function(e) {
      stop('An error occured when drawning predictions from censoring model')
      print(e)
    }
  )
  
  
  #-----------------------------#
  #--- Returning information ---#
  #-----------------------------#
  output <- list(g_pred = g_pred,
                 g_mod = g_mod)
  
  return(output)
}
