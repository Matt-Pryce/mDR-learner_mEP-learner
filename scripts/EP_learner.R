#######################################################################################
# Script: EP-Learner function
# Date: 25/03/24
# Author: Matt Pryce 
# Notes: EP-learner function for single time point setting
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
library(Sieve)

source("C:/Users/MatthewPryce/OneDrive - London School of Hygiene and Tropical Medicine/Documents/PhD/DR_Missing_Paper/GitHub_rep/mDR-learner/src/Data_management_1tp.R")
source("C:/Users/MatthewPryce/OneDrive - London School of Hygiene and Tropical Medicine/Documents/PhD/DR_Missing_Paper/GitHub_rep/mDR-learner/src/nuisance_models.R")



#######################################
#--- For single time point setting ---#
#######################################

#' @param analysis Type of analysis to be run 
#' @param data The data frame containing all required information
#' @param id Identification for individuals
#' @param outcome The name of the outcome of interest
#' @param exposure The name of the exposure of interest
#' @param outcome_observed_indicator Indicator identifying when the outcome variable is observed (1=observed, 0=missing)
#' @param splits Number of splitsu for cross-fitting (Variations allowed: 1, 3, 10)
#' @param out_method Statistical technique used to run the outcome models
#' @param out_covariates  List containing the names of the variables to be input into each outcome model
#' @param out_SL_lib Library to be used in super learner if selected
#' @param e_method Statistical technique used to run the propensity score model
#' @param e_covariates  List containing the names of the variables to be input into the propensity score model
#' @param e_SL_lib Library to be used in super learner if selected
#' @param g_method Statistical technique used to run the missingness model
#' @param g_covariates List containing the names of the variables to be input into the missingness model, excluding exposure
#' @param g_SL_lib Library to be used in super learner if selected for missingness model
#' @param nuisance_estimates_input Indicator for whether nuisance estimates provided
#' @param o_0_pred Variable name for unexposed outcome predictions (if provided)
#' @param o_1_pred Variable name for exposed outcome predictions (if provided)
#' @param e_pred Variable name for propensity score predictions (if provided)
#' @param g_pred Variable name for censoring predictions (if provided)
#' @param pse_method Statistical technique used to run the pseudo outcome model
#' @param pse_covariates List containing the names of the variables to be input into the pseudo outcome model
#' @param pse_SL_lib Library to be used in super learner if selected for pseudo outcome model
#' @param sieve_num_basis Dimension of sieve basis
#' @param sieve_interaction_order Interaction order for sieves
#' @param imp_covariates Covariates to be used in SL imputation model if SL imputation used
#' @param imp_SL_lib SL libaray for imputation model if SL imputation used
#' @param newdata New data to create predictions for

#' @return A list containing: CATE estimates, a dataset used to train the learner, dataset containing all 
#'         pseudo-outcome predictions (if splits 4) 


EP_learner <- function(analysis = c("Complete case","Available case","SL imputation","mEP-learner"),
                       data,
                       id,
                       outcome,
                       exposure,
                       outcome_observed_indicator = "None",
                       splits = c(1,10),
                       e_method = c("Parametric","Random forest","Super Learner"),
                       e_covariates,
                       e_SL_lib,
                       out_method = c("Parametric","Random forest","Super Learner"),
                       out_covariates,
                       out_SL_lib,
                       g_method = c("Parametric","Random forest","Super Learner"),
                       g_covariates = c(),
                       g_SL_lib,
                       nuisance_estimates_input = 0,
                       e_pred = NA,
                       o_0_pred = NA,
                       o_1_pred = NA,
                       g_pred = NA,
                       pse_method,
                       pse_covariates,
                       pse_SL_lib,
                       sieve_num_basis = NA,
                       sieve_interaction_order=3,
                       imp_covariates = c(),
                       imp_SL_lib,
                       newdata
){
  
  #-----------------------#
  #--- Data management ---#
  #-----------------------#
  
  if (analysis == "mEP-learner"){
    learner <- "mEP-learner"  
  }
  else {
    learner <- "EP-learner"
  }
   
  clean_data <- data_manage_1tp(data = data,
                                learner = learner,
                                analysis_type = analysis,
                                nuisance_estimates_input = nuisance_estimates_input,
                                id = id,
                                outcome = outcome,
                                exposure = exposure,
                                outcome_observed_indicator = outcome_observed_indicator,
                                e_covariates = e_covariates,
                                out_covariates = out_covariates,
                                g_covariates = g_covariates,
                                pse_covariates = pse_covariates,
                                imp_covariates = imp_covariates,
                                o_0_pred = o_0_pred,
                                o_1_pred = o_1_pred,
                                e_pred = e_pred,
                                g_pred = g_pred,
                                newdata = newdata)
  
  
  
  #-------------------------#
  #--- Imputing outcomes ---#
  #-------------------------#
  if (analysis == "SL imputation" & nuisance_estimates_input == 0){
    analysis_data <- nuis_mod(model = "Imputation",
                              data = clean_data$data,
                              covariates = imp_covariates,
                              SL_lib = imp_SL_lib,
                              Y_bin = clean_data$Y_bin,
                              Y_cont = clean_data$Y_cont)
  }
  if (analysis == "Complete case"){
    analysis_data <- clean_data$data
    analysis_data <- subset(analysis_data,is.na(analysis_data$Y)==0)
  }
  if (analysis == "Available case" | analysis == "mEP-learner"){
    analysis_data <- clean_data$data
  }

  
  
  #-------------------------------#
  #--- Running nuisance models ---#
  #-------------------------------#

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
          #Data for propensity score model
          e_data <- analysis_data
          e_data <- subset(e_data,e_data$s == i)   #Same for all splits

          #Data for outcome models #& censoring models UPDATE TO ALLOW MISSING OUTCOMES
          o_data <- analysis_data
          if (analysis == "Available case" | analysis == "mEP-learner"){
            o_data <- subset(o_data,is.na(o_data$Y) == 0)
          }
          if (analysis == "mEP-learner"){
            g_data <- analysis_data
          }
          if (splits == 1){
            o_data <- subset(o_data,o_data$s == i)
            if (analysis == "mEP-learner"){
              g_data <- subset(g_data,g_data$s == i)
            }
          }
          else if (splits == 10){
            o_data <- subset(o_data,o_data$s != i)
            if (analysis == "mEP-learner"){
              g_data <- subset(g_data,g_data$s != i)
            }
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
          po_o_data <- subset(analysis_data,select = c(out_covariates,"s"))
          po_o_data <- subset(po_o_data, po_o_data$s == i)
          po_o_data <- as.matrix(subset(po_o_data, select = -c(s)))
          if (analysis == "mEP-learner"){
            po_g_data <- subset(analysis_data,select = c("A",g_covariates,"s"))
            po_g_data <- subset(po_g_data, po_g_data$s == i)
            po_g_data <- as.matrix(subset(po_g_data, select = -c(s)))
          }
          if (analysis != "mEP-learner"){
            po_data <- subset(analysis_data,select = c("ID","Y","A",pse_covariates,"s")) 
          }
          else if (analysis == "mEP-learner"){
            po_data <- subset(analysis_data,select = c("ID","Y","A","G",pse_covariates,"s"))
          }
          po_data <- subset(po_data,po_data$s == i)
          
        },
        #if an error occurs, tell me the error
        error=function(e) {
          stop(paste("An error occured when collecting data for nuisance models predictions in split ",i,sep=""))
          print(e)
        }
      )


      #--- Running nuisance models & obtaining predictions ---#
      #Outcome models
      tryCatch(
        {
          outcome_models <- nuis_mod(model = "Outcome",
                                     data = o_data,
                                     method = out_method,
                                     covariates = out_covariates,
                                     SL_lib = out_SL_lib,
                                     Y_bin = clean_data$Y_bin,
                                     Y_cont = clean_data$Y_cont,
                                     pred_data = po_o_data)
        },
        #if an error occurs, tell me the error
        error=function(e) {
          stop(paste("An error occured in outcome model function in split ",i,sep=""))
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
        #if an error occurs, tell me the error
        error=function(e) {
          stop(paste("An error occured in propensity score model function in split ",i,sep=""))
          print(e)
        }
      )

      if (analysis == "mEP-learner"){
        #Censoring model
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
      }


      #--- Collecting nuisance model predictions ---#
      tryCatch(
        {
          po_data <- cbind(po_data,o_1_pred = outcome_models$o_mod_pred_1)
          po_data <- cbind(po_data,o_0_pred = outcome_models$o_mod_pred_0)
          if (e_method == "Random forest"){
            po_data <- cbind(po_data,e_pred = PS_model$e_pred$predictions)
          }
          else {
            po_data <- cbind(po_data,e_pred = PS_model$e_pred)
          }
          if (analysis == "mEP-learner"){
            if (g_method == "Random forest"){
              po_data <- cbind(po_data,g_pred = cen_model$g_pred$predictions)
            }
            else {
              po_data <- cbind(po_data,g_pred = cen_model$g_pred)
            }
          }
        },
        #if an error occurs, tell me the error
        error=function(e) {
          stop(paste("An error occured when outcome model function in split ",i,sep=""))
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

    #--- Collecting full test data ---#
    if (i==0){
      po_data_all <- po_data
    }
    else {
      po_data_all <- rbind(po_data_all,po_data)
    }
  }

  #--------------------------#
  #--- Establishing Sieve ---#
  #--------------------------#

  if (analysis == "Available case" | analysis == "mEP-learner"){
    sieve_data <- subset(po_data_all,is.na(po_data_all$Y)==0)   #Only update observed individuals
  }
  else if (analysis == "Complete case" | analysis == "SL imputation"){
    sieve_data <- po_data_all
  }

  X <- as.matrix(subset(sieve_data, select=pse_covariates))
  basisN <- sieve_num_basis
  if(is.na(basisN)) {
    basisN <- ceiling((nrow(X))^(1/3)*ncol(X))
  }
  basisN <- basisN + 1
  interaction_order <- sieve_interaction_order

  sieve_basis <- Sieve::sieve_preprocess(as.matrix(X),
                                         basisN = basisN,
                                         interaction_order = interaction_order,
                                         type = "cosine")$Phi


  #-----------------#
  #--- TMLE step ---#
  #-----------------#
  sieve_basis_train <- sieve_basis * ((sieve_data$A==1) - (sieve_data$A == 0))
  o_A_pred <- (sieve_data$A == 1) * (sieve_data$o_1_pred) + (sieve_data$A == 0) * (sieve_data$o_0_pred)

  if (analysis == "Complete case" | analysis == "Available case" | analysis == "SL imputation"){
    glmnet_fit <- glmnet::glmnet(x=sieve_basis_train, y=sieve_data$Y,
                                 offset = o_A_pred,
                                 weights = (sieve_data$A == 1)/sieve_data$e_pred + (sieve_data$A == 0)/(1-sieve_data$e_pred),
                                 family = "gaussian",
                                 standardize = FALSE,
                                 lambda = 1e-8,
                                 intercept = FALSE,
                                 alpha = 0)
  }
  else if (analysis == "mEP-learner"){
    glmnet_fit <- glmnet::glmnet(x=sieve_basis_train, y=sieve_data$Y,
                                 offset = o_A_pred,
                                 weights = ((sieve_data$A == 1)*(sieve_data$G == 1))/(sieve_data$e_pred*sieve_data$g_pred) + ((sieve_data$A == 0)*(sieve_data$G == 1))/((1-sieve_data$e_pred)*sieve_data$g_pred),
                                 family = "gaussian",
                                 standardize = FALSE,
                                 lambda = 1e-8,
                                 intercept = FALSE,
                                 alpha = 0)
  }
  beta <- as.matrix(glmnet_fit$beta)


  #------------------------------#
  #--- Updating outcome preds ---#
  #------------------------------#
  
  sieve_data$correction <- as.vector(sieve_basis %*% beta)
  sieve_data$o_1_pred_star <- sieve_data$o_1_pred + sieve_data$correction
  sieve_data$o_0_pred_star <- sieve_data$o_0_pred - sieve_data$correction
  
  if (analysis == "Available case" | analysis == "mEP-learner"){
    #Merging updated outcome preds with full dataset
    sieve_data_sub <- subset(sieve_data,select = c(ID,o_1_pred_star,o_0_pred_star))
    po_data_all <- merge(po_data_all,sieve_data_sub,by="ID",all.x = T)
    for (i in 1:length(po_data_all$Y)){
      if (is.na(po_data_all$o_1_pred_star[i])==1){
        po_data_all$o_1_pred_star[i] <- po_data_all$o_1_pred[i]
        po_data_all$o_0_pred_star[i] <- po_data_all$o_0_pred[i]
      }
    }
  }
  else {
    po_data_all <- sieve_data
  }

  #Construct EP-learner CATE estimate
  po_data_all$pse_Y <- po_data_all$o_1_pred_star  - po_data_all$o_0_pred_star


  #-----------------------------------------#
  #--- Running pseudo outcome regression ---#
  #-----------------------------------------#
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
      stop("An error occured when fitting the pseudo outcome model in split ")
      print(e)
    }
  )


  #-----------------------------#
  #--- Returning information ---#
  #-----------------------------#

  output <- list(CATE_est = pse_model$po_pred,
                 train_data = po_data_all,
                 newdata = newdata)
  
  return(output)
}



###############################################################

#Example
EP_check <- EP_learner(analysis = "mEP-learner",
                       data = check,
                       id = "ID",
                       outcome = "Y",
                       exposure = "A",
                       outcome_observed_indicator = "G_obs",
                       splits = 10,
                       e_covariates = c("X1","X2","X3","X4","X5","X6",
                                        "X7","X8","X9","X10","X11","X12"),
                       e_method = "Super learner",
                       e_SL_lib = c("SL.mean",
                                    "SL.lm",
                                    "SL.glmnet_8", "SL.glmnet_9",
                                    "SL.glmnet_11"),
                       out_method = "Super learner",
                       out_covariates = c("X1","X2","X3","X4","X5","X6",
                                          "X7","X8","X9","X10","X11","X12"),
                       out_SL_lib = c("SL.mean",
                                      "SL.lm",
                                      "SL.glmnet_8", "SL.glmnet_9",
                                      "SL.glmnet_11", "SL.glmnet_12",
                                      "SL.ranger_1","SL.ranger_2","SL.ranger_3",
                                      "SL.ranger_4","SL.ranger_5","SL.ranger_6",
                                      "SL.nnet_1","SL.nnet_2","SL.nnet_3",
                                      "SL.svm_1",
                                      "SL.kernelKnn_4","SL.kernelKnn_10"),
                       g_covariates = c("X1","X2","X3","X4","X5","X6",
                                        "X7","X8","X9","X10","X11","X12"),
                       g_method = "Parametric",
                       g_SL_lib = c("SL.mean",
                                    "SL.lm",
                                    "SL.glmnet_8", "SL.glmnet_9",
                                    "SL.glmnet_11", "SL.glmnet_12",
                                    "SL.ranger_1","SL.ranger_2","SL.ranger_3",
                                    "SL.ranger_4","SL.ranger_5","SL.ranger_6",
                                    "SL.nnet_1","SL.nnet_2","SL.nnet_3",
                                    "SL.svm_1",
                                    "SL.kernelKnn_4","SL.kernelKnn_10"),
                       sieve_interaction_order=3,
                       imp_covariates = c("X1","X2","X3","X4","X5","X6",
                                          "X7","X8","X9","X10","X11","X12"),
                       imp_SL_lib = c("SL.mean",
                                      "SL.lm",
                                      "SL.glmnet_8", "SL.glmnet_9",
                                      "SL.glmnet_11", "SL.glmnet_12",
                                      "SL.ranger_1","SL.ranger_2","SL.ranger_3",
                                      "SL.ranger_4","SL.ranger_5","SL.ranger_6",
                                      "SL.nnet_1","SL.nnet_2","SL.nnet_3",
                                      "SL.svm_1",
                                      "SL.kernelKnn_4","SL.kernelKnn_10"),
                       pse_method = "Super learner",
                       pse_covariates = c("X1","X2","X3","X4","X5","X6",
                                          "X7","X8","X9","X10","X11","X12"),
                       pse_SL_lib = c("SL.mean",
                                      "SL.lm",
                                      "SL.glmnet_8", "SL.glmnet_9",
                                      "SL.glmnet_11", "SL.glmnet_12",
                                      "SL.ranger_1","SL.ranger_2","SL.ranger_3",
                                      "SL.ranger_4","SL.ranger_5","SL.ranger_6",
                                      "SL.nnet_1","SL.nnet_2","SL.nnet_3",
                                      "SL.svm_1",
                                      "SL.kernelKnn_4","SL.kernelKnn_10"),
                       newdata = check)


