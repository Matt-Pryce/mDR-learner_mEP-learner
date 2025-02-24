#######################################################################################
# Script: Nuisance model function
# Date: 21/03/24
# Author: Matt Pryce 
# Notes: - Used to run each nuisance model
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


#' @param model Nuisance model to be run
#' @param data The data frame containing all required information
#' @param method Statistical technique used to run the model
#' @param covariates  List containing the names of the variables to be input into the model
#' @param SL_lib Library to be used in super learner if selected
#' @param Y_bin Indicator for when the outcome is binary
#' @param Y_cont Indicator for when the outcome is continuous 
#' @param pred_data New data to create predictions for

#' @return Outcome models and their predictions for pred_data 


nuis_mod <- function(model,
                     data,
                     method = c("Parametric","Random forest","Super learner"),
                     covariates,
                     SL_lib,
                     Y_bin,
                     Y_cont,
                     pred_data,
                     CI_tuned_params
){
  
  #------------------------------#
  #--- Creating training data ---#
  #------------------------------#

  tryCatch(
    {
      if (model == "Outcome"){
        train_data0 <- subset(data,data$A==0)
        train_data0 <- subset(train_data0,select = c("Y",covariates))
        train_data1 <- subset(data,data$A==1)
        train_data1 <- subset(train_data1,select = c("Y",covariates))
      } 
      else if (model == "Outcome - MSE"){
        train_data0 <- subset(data,data$A==0)
        train_data1 <- subset(data,data$A==1)
        if (Y_bin == 1){
          train_data0 <- subset(train_data0,select = c("Y",covariates))
          train_data1 <- subset(train_data1,select = c("Y",covariates))
        }
        if (Y_cont == 1){
          train_data0 <- subset(train_data0,select = c("Y_norm",covariates))
          train_data1 <- subset(train_data1,select = c("Y_norm",covariates))
        }
      }
      else if (model == "Imputation"){
        train_data <- subset(data,data$G==1)
        train_data_X <- data.frame(subset(train_data, select = c(covariates,"A")))
      }
      else if (model == "Propensity score"){
        train_data <- as.data.frame(subset(data,select = c(covariates,"A","s")))
      }
      else if (model == "Censoring"){
        train_data <- as.data.frame(subset(data,select = c("G",covariates,"A","s")))
      }
      else if (model == "Pseudo outcome" | model == "Pseudo outcome - CI"){
        train_data <- data
      }
    },
    error=function(e) {
      stop('An error occured when creating analysis data')
      print(e)
    }
  )

  #----------------------#
  #--- Running models ---#
  #----------------------#

  if (model == "Imputation"){
    tryCatch(
      {
        if (Y_bin == 1){
          mod <- SuperLearner(Y = train_data$Y, X = train_data_X,
                              method = "method.NNLS",
                              family = binomial(),
                              cvControl = list(V = 10, stratifyCV=TRUE),
                              SL.library = SL_lib)
        }
        else if (Y_cont == 1){
          mod <- SuperLearner(Y = train_data$Y, X = train_data_X,
                              method = "method.NNLS",
                              family = gaussian(),
                              cvControl = list(V = 10, stratifyCV=FALSE),
                              SL.library = SL_lib)
        }
      },
      error=function(e) {
        stop('An error occured when running an imputation model')
        print(e)
      }
    )
  }

  if (model == "Outcome"){
    tryCatch(
      {
        if (method == "Random forest"){
          X_S_0 <- as.matrix(subset(train_data0, select = covariates))
          mod_0 <- regression_forest(X_S_0, train_data0$Y, honesty = FALSE,tune.parameters = "all")

          X_S_1 <- as.matrix(subset(train_data1, select = covariates))
          mod_1 <- regression_forest(X_S_1, train_data1$Y, honesty = FALSE,tune.parameters = "all")
        }
        else if (method == "Parametric"){
          if (Y_bin == 1){
            #Running first outcome models
            mod_0 <- glm(Y ~ . , data = train_data0, family = binomial())
            mod_1 <- glm(Y ~ . , data = train_data1, family = binomial())
          }
          if (Y_cont == 1){
            #Running first outcome models
            mod_0 <- lm(Y ~ . , data = train_data0)
            mod_1 <- lm(Y ~ . , data = train_data1)
          }
        }
        else if (method == "Super learner"){
          if (Y_bin == 1){
            mod_0 <- SuperLearner(Y = train_data0$Y, X = data.frame(subset(train_data0, select = covariates)),
                                      method = "method.NNLS",
                                      family = binomial(),
                                      cvControl = list(V = 10, stratifyCV=TRUE),
                                      SL.library = SL_lib)
            mod_1 <- SuperLearner(Y = train_data1$Y, X = data.frame(subset(train_data1, select = covariates)),
                                      method = "method.NNLS",
                                      family = binomial(),
                                      cvControl = list(V = 10, stratifyCV=TRUE),
                                      SL.library = SL_lib)
          }
          if (Y_cont == 1){
            mod_0 <- SuperLearner(Y = train_data0$Y, X = data.frame(subset(train_data0, select = covariates)),
                                      method = "method.NNLS",
                                      family = gaussian(),
                                      cvControl = list(V = 10, stratifyCV=FALSE),
                                      SL.library = SL_lib)
            mod_1 <- SuperLearner(Y = train_data1$Y, X = data.frame(subset(train_data1, select = covariates)),
                                      method = "method.NNLS",
                                      family = gaussian(),
                                      cvControl = list(V = 10, stratifyCV=FALSE),
                                      SL.library = SL_lib)
          }
        }
      },
      error=function(e) {
        stop('An error occured when running outcome models')
        print(e)
      }
    )
  }

  if (model == "Outcome - MSE"){
    tryCatch(
      {
        if (method == "Random forest"){
          X_S_0 <- as.matrix(subset(train_data0, select = covariates))
          X_S_1 <- as.matrix(subset(train_data1, select = covariates))
          if (Y_bin == 1){
            mod_0 <- regression_forest(X_S_0, train_data0$Y, honesty = FALSE,tune.parameters = "all")
            mod_1 <- regression_forest(X_S_1, train_data1$Y, honesty = FALSE,tune.parameters = "all")
          }
          else {
            mod_0 <- regression_forest(X_S_0, train_data0$Y_norm, honesty = FALSE,tune.parameters = "all")
            mod_1 <- regression_forest(X_S_1, train_data1$Y_norm, honesty = FALSE,tune.parameters = "all")
          }
        }
        else if (method == "Parametric"){
          if (Y_bin == 1){
            #Running first outcome models
            mod_0 <- glm(Y ~ . , data = train_data0, family = binomial())
            mod_1 <- glm(Y ~ . , data = train_data1, family = binomial())
          }
          if (Y_cont == 1){
            #Running first outcome models
            mod_0 <- glm(Y_norm ~ . , data = train_data0, family = quasibinomial)
            mod_1 <- glm(Y_norm ~ . , data = train_data1, family = quasibinomial)
          }
        }
        else if (method == "Super learner"){
          if (Y_bin == 1){
            mod_0 <- SuperLearner(Y = train_data0$Y, X = data.frame(subset(train_data0, select = covariates)),
                                  method = "method.NNLS",
                                  family = binomial(),
                                  cvControl = list(V = 10, stratifyCV=TRUE),
                                  SL.library = SL_lib)
            mod_1 <- SuperLearner(Y = train_data1$Y, X = data.frame(subset(train_data1, select = covariates)),
                                  method = "method.NNLS",
                                  family = binomial(),
                                  cvControl = list(V = 10, stratifyCV=TRUE),
                                  SL.library = SL_lib)
          }
          if (Y_cont == 1){
            mod_0 <- SuperLearner(Y = train_data0$Y_norm, X = data.frame(subset(train_data0, select = covariates)),
                                  method = "method.NNLS",
                                  family = gaussian(),
                                  cvControl = list(V = 10, stratifyCV=FALSE),
                                  SL.library = SL_lib)
            mod_1 <- SuperLearner(Y = train_data1$Y_norm, X = data.frame(subset(train_data1, select = covariates)),
                                  method = "method.NNLS",
                                  family = gaussian(),
                                  cvControl = list(V = 10, stratifyCV=FALSE),
                                  SL.library = SL_lib)
          }
        }
      },
      error=function(e) {
        stop('An error occured when running outcome models')
        print(e)
      }
    )
  }
  
  if (model == "Propensity score" | model == "Censoring" | model == "Pseudo outcome"){
    if (method == "Random forest"){
      if (model == "Propensity score"){
        X <- as.matrix(subset(train_data, select = covariates))
        mod <- regression_forest(X, train_data$A, honesty = FALSE,tune.parameters = "all")
      }
      else if (model == "Censoring"){
        X <- as.matrix(subset(train_data, select = c(covariates,"A")))
        mod <- regression_forest(X, train_data$G, honesty = TRUE,tune.parameters = "all")
      }
      else if (model == "Pseudo outcome"){
        X <- as.matrix(subset(train_data, select = covariates))
        mod <- regression_forest(X, train_data$pse_Y,
                                 # sample.fraction = 0.05,
                                 num.trees = 20000,
                                 tune.parameters = c("sample.fraction","mtry", "min.node.size", "honesty.fraction", "honesty.prune.leaves", "alpha", "imbalance.penalty"),
                                 ci.group.size = 1)
      }
    }
    else if (method == "Parametric"){
      if (model== "Propensity score"){
        fit_data <- subset(train_data,select = -c(s))
        mod <- glm(A ~ . , data = fit_data, family = binomial())
      }
      else if (model == "Censoring"){
        if (model == "Censoring"){
          fit_data <- subset(train_data,select = -c(s))
        }
        else {
          fit_data <- train_data
        }
        mod <- glm(G ~ . , data = fit_data, family = binomial())
      }
      else if (model == "Pseudo outcome"){
        fit_data <- subset(train_data,select = c("pse_Y",covariates))
        mod <- lm(pse_Y ~ . , data = fit_data)
      }
    }
    else if (method == "Super learner"){
      if (model == "Propensity score"){
        sums <- table(train_data$A)
        cv_folds <- min(10,sums[1],sums[2])
        mod <- SuperLearner(Y = train_data$A, X = data.frame(subset(train_data, select = covariates)),
                            method = "method.NNLS",
                            family = binomial(),
                            cvControl = list(V = cv_folds, stratifyCV=TRUE),
                            SL.library = SL_lib)
      }
      else if (model == "Censoring"){
        sums <- table(train_data$G)
        cv_folds <- min(10,sums[1],sums[2])
        mod <- SuperLearner(Y = train_data$G, X = data.frame(subset(train_data, select = c(covariates,"A"))),
                            method = "method.NNLS",
                            family = binomial(),
                            cvControl = list(V = cv_folds, stratifyCV=TRUE),
                            SL.library = SL_lib)
      }
      else if (model == "Pseudo outcome"){
        mod <- SuperLearner(Y = train_data$pse_Y, X = data.frame(subset(train_data, select = covariates)),
                            method = "method.NNLS",
                            family = gaussian(),
                            cvControl = list(V = 10, stratifyCV=FALSE),
                            SL.library = SL_lib)
      }
    }
  }

  
  if (model == "Pseudo outcome - CI"){
    if (method == "Random forest"){
      #Defining tuning parameter values
      tuned_sample.fraction <- CI_tuned_params$sample.fraction
      tuned_mtry <- CI_tuned_params$mtry
      tuned_min.node.size <- CI_tuned_params$min.node.size
      tuned_honesty.fraction <- CI_tuned_params$honesty.fraction
      tuned_honesty.prune.leaves <- CI_tuned_params$honesty.prune.leaves
      tuned_alpha <- CI_tuned_params$alpha
      tuned_imbalance.penalty <- CI_tuned_params$imbalance.penalty
      
      X <- as.matrix(subset(train_data, select = covariates))
      mod <- regression_forest(X, train_data$pse_Y,
                               sample.fraction = tuned_sample.fraction,
                               mtry = tuned_mtry,
                               min.node.size = tuned_min.node.size,
                               honesty.fraction = tuned_honesty.fraction,
                               honesty.prune.leaves = tuned_honesty.prune.leaves,
                               alpha = tuned_alpha,
                               imbalance.penalty = tuned_imbalance.penalty,
                               num.trees = 2000)
    }
    else if (method == "Super learner"){
      #Could be added
    }
  }



  #-----------------------------------#
  #--- Obtaining model predictions ---#
  #-----------------------------------#

  if (model == "Imputation"){
    tryCatch(
      {
        #Obtaining predictions from imputation model
        pred_data <- subset(data,select = c(covariates,"A"))
        preds <- predict(mod,pred_data)$pred
        analysis_data <- cbind(data,imp_pred = preds)

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
  }

  if (model == "Outcome" | model == "Outcome - MSE"){
    tryCatch(
      {
        #Obtaining preds from previous outcome model
        if (method == "Random forest"){
          mod_pred_0 <- predict(mod_0, as.matrix(subset(pred_data,select=c(covariates))))$pred
          mod_pred_1 <- predict(mod_1, as.matrix(subset(pred_data,select=c(covariates))))$pred
        }
        else if (method == "Parametric"){
          mod_pred_0 <- predict(mod_0, data.frame(pred_data), type = "response")
          mod_pred_1 <- predict(mod_1, data.frame(pred_data), type = "response")
        }
        else if (method == "Super learner"){
          pred_data <- subset(pred_data,select = c(covariates))
          mod_pred_0 <- predict(mod_0, pred_data)$pred
          mod_pred_1 <- predict(mod_1, pred_data)$pred
        }
      },
      #if an error occurs, tell me the error
      error=function(e) {
        stop('An error occured when drawning predictions from outcome models')
        print(e)
      }
    )
  }

  if (model == "Propensity score" | model == "Censoring"){
    if (method == "Random forest"){
      pred <- predict(mod, pred_data)$predictions
    }
    else if (method == "Parametric"){
      pred <- predict(mod, as.data.frame(pred_data), type = "response")
    }
    else if (method == "Super learner"){
      pred <- predict(mod, as.data.frame(pred_data))$pred
    }
    
    #Adding in check for estimates violating positivity
    if (model == "Propensity score" & (min(pred) == 0 | min(pred) == 1)){
      stop("Propensity scores violated positivity")
    } 
    if (model == "Censoring" & min(pred) == 0){
      stop("Censoring probabilities violated positivity")
    } 
  }
  
  if (model == "Pseudo outcome" | model == "Pseudo outcome - CI"){
    if (method == "Random forest"){
      pred_data_matrix <- as.matrix(subset(pred_data, select = c(covariates)))
      pred <- predict(mod, pred_data_matrix)
    }
    else if (method == "Parametric"){
      pred_data <- subset(pred_data, select = c(covariates))
      pred <- as.data.frame(predict(mod, pred_data, type = "response"))
    }
    else if (method == "Super learner"){
      pred_data <- subset(pred_data, select = c(covariates))
      pred <- predict(mod, pred_data)$pred
    }
  }



  #-----------------------------#
  #--- Returning information ---#
  #-----------------------------#

  if (model == "Imputation"){
    output <- analysis_data
  }
  else if (model == "Outcome" | model == "Outcome - MSE"){
    output <- list(o_mod_pred_1 = mod_pred_1,
                   o_mod_pred_0 = mod_pred_0,
                   o_mod_0 = mod_0,
                   o_mod_1 = mod_1)
  }
  else if (model == "Propensity score"){
    output <- list(e_pred = pred,
                   e_mod = mod)
  }
  else if (model == "Censoring"){
    output <- list(g_pred = pred,
                   g_mod = mod)
  }
  else if (model == "Pseudo outcome"){
    output <- list(po_pred = pred,
                   po_mod = mod)
  }
  else if (model == "Pseudo outcome - CI"){
    output <- pred$predictions
  }

  return(output)
}


