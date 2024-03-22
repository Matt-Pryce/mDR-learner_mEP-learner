#######################################################################################
# Script: Pseudo-outcome model function
# Date: 21/03/24
# Author: Matt Pryce 
# Notes: - Used in each learner to run pseudo-outcome model
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
#' @param pse_method Statistical technique used to run the pseudo outcome model
#' @param pse_covariates List containing the names of the variables to be input into the pseudo outcome model
#' @param pse_SL_lib Library to be used in super learner if selected for pseudo outcome model
#' @param pred_data New data to create predictions for

#' @return Pseudo-outcome model and predictions from it for pred_data 


Pseudo_mod <- function(data,
                       id,
                       outcome,
                       exposure,
                       pse_method = c("Parametric","Random forest","Super Learner"),
                       pse_covariates,
                       pse_SL_lib,
                       pred_data
){
  
  #----------------------#
  #--- Running models ---#
  #----------------------#
  
  tryCatch(
    {
      #--- Pseudo outcome model---#
      if (pse_method == "Random forest"){
        #Running pseudo outcome model
        po_mod <- regression_forest(X = as.matrix(subset(data,select=c(pse_covariates))), Y = data$pse_Y)#, tune.parameters = "all")
      }
      else if (pse_method == "Parametric"){
        #Creating dataset for pseudo outcome model
        po_fit_data <- subset(data,select = c("pse_Y",pse_covariates))
        po_mod <- lm(pse_Y ~ . , data = po_fit_data)
      }
      else if (pse_method == "Super learner") {
        po_mod <- SuperLearner(Y = data$pse_Y, X = data.frame(subset(data, select = pse_covariates)),
                               method = "method.NNLS",
                               family = gaussian(),
                               cvControl = list(V = 10, stratifyCV=FALSE),
                               SL.library = pse_SL_lib)
      }
    },
    error=function(e) {
      stop('An error occured when running pseudo-outcome model')
      print(e)
    }
  )
  
  
  #-----------------------------------#
  #--- Obtaining model predictions ---#
  #-----------------------------------#
  tryCatch(
    {
      if (pse_method == "Random forest"){
        pse_pred_data <- as.matrix(subset(pred_data, select = c(pse_covariates)))
        po_pred <- predict(po_mod, pse_pred_data)
      }
      else if (pse_method == "Parametric"){
        pse_pred_data <- subset(pred_data, select = c(pse_covariates))
        po_pred <- as.data.frame(predict(po_mod, pse_pred_data, type = "response"))
      }
      else if (pse_method == "Super learner"){
        pse_pred_data <- subset(pred_data, select = c(pse_covariates))
        po_pred <- predict(po_mod, pse_pred_data)$pred
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
  output <- list(po_pred = po_pred,
                 po_mod = po_mod)
  
  return(output)
}
