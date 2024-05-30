#######################################################################################
# Script: Imputation function for 1tp setting
# Date: 20/03/24
# Author: Matt Pryce 
# Notes: - Used when running T-learner or DR-learner models for one time point
#        - Creates imputed outcome using SL 
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
#' @param imp_covariates Covariates to be used in SL imputation model if SL imputation used
#' @param imp_SL_lib SL libaray for imputation model if SL imputation used
#' @param Y_bin Indicator for hether outcome is binary
#' @param Y_cont Indicator for hether outcome is continuous 

#' @return A dataset containing n CATE estimates (UPDATE) 


out_imp_1tp <- function(data,
                        imp_covariates,
                        imp_SL_lib,
                        Y_bin,
                        Y_cont
){
  
  #--------------------------------#
  #--- Creating imputation data ---#
  #--------------------------------#
  tryCatch( 
    {
      #Creating data for imputation model
      imp_data <- subset(data,data$G==1)
      imp_data_X <- data.frame(subset(imp_data, select = c(imp_covariates,"A")))
    },
    error=function(e) {
      stop('An error occured when creating imputation data')
      print(e)
    }
  )
  
  
  #--------------------------#
  #--- Running imputation ---#
  #--------------------------#
  tryCatch(
    {
      if (Y_bin == 1){
        imp_mod <- SuperLearner(Y = imp_data$Y, X = imp_data_X,
                                method = "method.NNLS",
                                family = binomial(),
                                cvControl = list(V = 10, stratifyCV=TRUE),
                                SL.library = imp_SL_lib)
      }
      else if (Y_cont == 1){
        imp_mod <- SuperLearner(Y = imp_data$Y, X = imp_data_X,
                                method = "method.NNLS",
                                family = gaussian(),
                                cvControl = list(V = 10, stratifyCV=FALSE),
                                SL.library = imp_SL_lib)
      }
    },
    error=function(e) {
      stop('An error occured when creating imputation data')
      print(e)
    }
  )


  #------------------------------------#
  #--- Obtaining imputation outcome ---#
  #------------------------------------#
  tryCatch(
    {
      #Obtaining predictions from imputation model
      imp_pred_data <- subset(data,select = c(imp_covariates,"A"))
      imp_preds <- predict(imp_mod,imp_pred_data)$pred
      analysis_data <- cbind(data,imp_pred = imp_preds)

      #Imputing predictions for missing outcomes
      for (i in 1:dim(analysis_data)[1]){
        if (is.na(analysis_data$Y[i])==1){
          analysis_data$Y[i] = analysis_data$imp_pred[i]
        }
      }
    },
    error=function(e) {
      stop('An error occured when creating imputation data')
      print(e)
    }
  )

  
  #-----------------------------#
  #--- Returning information ---#
  #-----------------------------#
  return(analysis_data)
}
