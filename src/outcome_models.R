#######################################################################################
# Script: Outcome model function
# Date: 21/03/24
# Author: Matt Pryce 
# Notes: - Used in each learner to run outcome models
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


out_mods <- function(data,
                     id,
                     outcome,
                     exposure,
                     out_method = c("Parametric","Random forest","Super Learner"),
                     out_covariates,
                     out_SL_lib,
                     out_SL_strat = FALSE,
                     Y_bin,
                     Y_cont,
                     nuisance_estimates_input = 0,
                     o_0_pred = NA,
                     o_1_pred = NA,
                     pred_data
){
  
  #----------------------------------------#
  #--- Creating data for outcome models ---#
  #----------------------------------------#
  tryCatch( 
    {
      analysis_data0 <- subset(data,data$A==0)
      analysis_data0 <- subset(analysis_data0,select = c("Y",out_covariates))
      analysis_data1 <- subset(data,data$A==1)
      analysis_data1 <- subset(analysis_data1,select = c("Y",out_covariates))
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
      if (out_method == "Random forest"){
        X_S_0 <- as.matrix(subset(analysis_data0, select = out_covariates))
        out_mod_0 <- regression_forest(X_S_0, analysis_data0$Y, honesty = FALSE,tune.parameters = "all")

        X_S_1 <- as.matrix(subset(analysis_data1, select = out_covariates))
        out_mod_1 <- regression_forest(X_S_1, analysis_data1$Y, honesty = FALSE,tune.parameters = "all")
      }
      else if (out_method == "Parametric"){
        if (Y_bin == 1){
          #Running first outcome models
          out_mod_0 <- glm(Y ~ . , data = analysis_data0, family = binomial())
          out_mod_1 <- glm(Y ~ . , data = analysis_data1, family = binomial())
        }
        if (Y_cont == 1){
          #Running first outcome models
          out_mod_0 <- lm(Y ~ . , data = analysis_data0)
          out_mod_1 <- lm(Y ~ . , data = analysis_data1)
        }
      }
      else if (out_method == "Super learner"){
        if (Y_bin == 1){
          out_mod_0 <- SuperLearner(Y = analysis_data0$Y, X = data.frame(subset(analysis_data0, select = out_covariates)),
                                    method = "method.NNLS",
                                    family = binomial(),
                                    cvControl = list(V = 10, stratifyCV=out_SL_strat),
                                    SL.library = out_SL_lib)
          out_mod_1 <- SuperLearner(Y = analysis_data1$Y, X = data.frame(subset(analysis_data1, select = out_covariates)),
                                    method = "method.NNLS",
                                    family = binomial(),
                                    cvControl = list(V = 10, stratifyCV=out_SL_strat),
                                    SL.library = out_SL_lib)
        }
        if (Y_cont == 1){
          out_mod_0 <- SuperLearner(Y = analysis_data0$Y, X = data.frame(subset(analysis_data0, select = out_covariates)),
                                    method = "method.NNLS",
                                    family = gaussian(),
                                    cvControl = list(V = 10, stratifyCV=out_SL_strat),
                                    SL.library = out_SL_lib)
          out_mod_1 <- SuperLearner(Y = analysis_data1$Y, X = data.frame(subset(analysis_data1, select = out_covariates)),
                                    method = "method.NNLS",
                                    family = gaussian(),
                                    cvControl = list(V = 10, stratifyCV=out_SL_strat),
                                    SL.library = out_SL_lib)
        }
      }
    },
    error=function(e) {
      stop('An error occured when running outcome models')
      print(e)
    }
  )


  #-----------------------------------#
  #--- Obtaining model predictions ---#
  #-----------------------------------#
  tryCatch(
    {
      #Obtaining preds from previous outcome model
      if (out_method == "Random forest"){
        o_mod_pred_0 <- predict(out_mod_0, as.matrix(subset(pred_data,select=c(out_covariates))))$pred
        o_mod_pred_1 <- predict(out_mod_1, as.matrix(subset(pred_data,select=c(out_covariates))))$pred
      }
      else if (out_method == "Parametric"){
        o_mod_pred_0 <- predict(out_mod_0, data.frame(pred_data), type = "response")
        o_mod_pred_1 <- predict(out_mod_1, data.frame(pred_data), type = "response")
      }
      else if (out_method == "Super learner"){
        o_mod_pred_0 <- predict(out_mod_0, subset(pred_data,select=c(out_covariates)) )$pred
        o_mod_pred_1 <- predict(out_mod_1, subset(pred_data,select=c(out_covariates)) )$pred
      }
    },
    #if an error occurs, tell me the error
    error=function(e) {
      stop('An error occured when drawning predictions from outcome models')
      print(e)
    }
  )


  #-----------------------------#
  #--- Returning information ---#
  #-----------------------------#
  output <- list(o_mod_pred_1 = o_mod_pred_1,
                 o_mod_pred_0 = o_mod_pred_0,
                 o_mod_0 = out_mod_0,
                 o_mod_1 = out_mod_1)
  
  return(output)
}
