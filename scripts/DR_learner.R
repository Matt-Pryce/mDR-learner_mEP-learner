#######################################################################################
# Script: DR-Learner function
# Date: 20/03/24
# Author: Matt Pryce 
# Notes: DR-learner function for single time point setting
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


DR_learner <- function(analysis = c("Complete case","Available case","SL imputation"),
                       data,
                       id,
                       outcome,
                       exposure,
                       outcome_observed_indicator = "None",
                       splits = c(1,3,10),
                       e_method = c("Parametric","Random forest","Super Learner"),
                       e_covariates,
                       e_SL_lib,
                       e_SL_strat = TRUE,
                       out_method = c("Parametric","Random forest","Super Learner"),
                       out_covariates,
                       out_SL_lib,
                       out_SL_strat = FALSE,
                       nuisance_estimates_input = 0,
                       e_pred = NA,
                       o_0_pred = NA,
                       o_1_pred = NA,
                       pse_method = c("Parametric","Random forest","Super Learner"),
                       pse_covariates,
                       pse_SL_lib,
                       imp_covariates = c(),
                       imp_SL_lib,
                       imp_SL_strat = FALSE,
                       newdata
){
  
  #-----------------------#
  #--- Data management ---#
  #-----------------------#
  
  clean_data <- data_manage_1tp(data = data,
                                learner = "DR-learner",
                                analysis_type = analysis,
                                nuisance_estimates_input = nuisance_estimates_input,
                                id = id,
                                outcome = outcome,
                                exposure = exposure,
                                outcome_observed_indicator = outcome_observed_indicator,
                                e_covariates = e_covariates,
                                out_covariates = out_covariates,
                                imp_covariates = imp_covariates,
                                pse_covariates = pse_covariates,
                                o_0_pred = o_0_pred,
                                o_1_pred = o_1_pred,
                                e_pred = e_pred,
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
  if (analysis == "Complete case"){
    analysis_data <- clean_data$data
    analysis_data <- subset(analysis_data,analysis_data$G==1) 
  }
  if (analysis == "Available case"){
    analysis_data <- clean_data$data
  }
  
  
  
  #------------------------------------------------------#
  #--- Running nuisance models & pseudo outcome model ---#
  #------------------------------------------------------#

  tryCatch(
    {
      #Checking number of splits is correct
      if (splits != 1 & splits != 3 & splits != 10){
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
          
          #Data for outcome models
          o_data <- analysis_data
          if (analysis == "Available case"){
            o_data <- subset(o_data,o_data$G == 1)
          }
          if (splits == 3){
            o_data <- subset(o_data,o_data$s == ((i+1) %% 3))
          }
          else if (splits == 1 | splits == 2){
            o_data <- subset(o_data,o_data$s == i)
          }
          else if (splits == 10){
            o_data <- subset(o_data,o_data$s != i)
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
          if (analysis == "Available case"){
            analysis_data <- subset(analysis_data,analysis_data$G==1)
          }
          
          if (splits == 3){
            po_e_data <- subset(analysis_data, select = c(e_covariates,"s"))
            po_e_data <- subset(po_e_data, po_e_data$s == ((i+2) %% 3))
            po_e_data <- as.matrix(subset(po_e_data, select = -c(s)))
            po_o_data <- subset(analysis_data,select = c(out_covariates,"s"))
            po_o_data <- subset(po_o_data, po_o_data$s == ((i+2) %% 3))
            po_o_data <- as.matrix(subset(po_o_data, select = -c(s)))
            po_data <- subset(analysis_data,select = c("ID","Y","A",pse_covariates,"s"))
            po_data <- subset(po_data,po_data$s == ((i+2) %% 3))
          }
          else if (splits == 1 | splits == 10){
            po_e_data <- subset(analysis_data, select = c(e_covariates,"s"))
            po_e_data <- subset(po_e_data, po_e_data$s == i)
            po_e_data <- as.matrix(subset(po_e_data, select = -c(s)))
            po_o_data <- subset(analysis_data,select = c(out_covariates,"s"))
            po_o_data <- subset(po_o_data, po_o_data$s == i)
            po_o_data <- as.matrix(subset(po_o_data, select = -c(s)))
            po_data <- subset(analysis_data,select = c("ID","Y","A",pse_covariates,"s"))
            po_data <- subset(po_data,po_data$s == i)
          }
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
          outcome_models <- out_mods(data = o_data,
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
          PS_model <- PS_mod(data = o_data,
                             id = id,
                             outcome = outcome,
                             exposure = exposure,
                             e_method = e_method,
                             e_covariates = e_covariates,
                             e_SL_lib = e_SL_lib,
                             e_SL_strat = e_SL_strat,
                             nuisance_estimates_input = nuisance_estimates_input,
                             e_pred = e_pred,
                             pred_data = po_e_data
          )
        },
        #if an error occurs, tell me the error
        error=function(e) {
          stop(paste("An error occured in propensity score model function in split ",i,sep=""))
          print(e)
        }
      )
      
      #--- Collecting nuisance model predictions ---#
      tryCatch(
        {
          po_data <- cbind(po_data,o_1_pred = outcome_models$o_mod_pred_1)
          po_data <- cbind(po_data,o_0_pred = outcome_models$o_mod_pred_0)
          po_data <- cbind(po_data,e_pred = PS_model$e_pred)
        },
        #if an error occurs, tell me the error
        error=function(e) {
          stop(paste("An error occured when outcome model function in split ",i,sep=""))
          print(e)
        }
      )

    }
    else if (nuisance_estimates_input == 1){
      
      if (analysis == "Available case"){
        analysis_data <- subset(analysis_data,analysis_data$G==1)
      }
      if (splits == 3){
        po_data <- analysis_data
        po_data <- subset(po_data,po_data$s == ((i+2) %% 3))
      }
      else if (splits == 1 | splits == 10){
        po_data <- analysis_data
        po_data <- subset(po_data,po_data$s == i)
      }
    }
    
    
    #--- Calculating pseudo-outcomes ---#
    tryCatch(
      {
        #First setting missing values to 99
        po_data <- po_data %>% mutate_if(is.numeric, function(x) ifelse(is.na(x), 99, x))
        
        #Calculating pseudo outcome
        po_data$pse_Y <- ((po_data$A - po_data$e_pred)/(po_data$e_pred*(1-po_data$e_pred))) *
          (po_data$Y - (po_data$A*po_data$o_1_pred +(1-po_data$A)*po_data$o_0_pred)) +
          po_data$o_1_pred - po_data$o_0_pred
      },
      #if an error occurs, tell me the error
      error=function(e) {
        stop(paste("An error occured when generating the pseudo outcome in split ",i,sep=""))
        print(e)
      }
    )
    

    #--- Collecting full test data with pseudo-outcomes ---#
    if (i==0){
      po_data_all <- po_data
    }
    else {
      po_data_all <- rbind(po_data_all,po_data)
    }
    
    
    #--- Running pseudo-outcome regression (if 3 split option chosen) ---#
    if (splits == 3){
      tryCatch(
        {
          pse_model <- Pseudo_mod(data = po_data,
                                  id = id,
                                  outcome = outcome,
                                  exposure = exposure,
                                  pse_method = pse_method,
                                  pse_covariates = pse_covariates,
                                  pse_SL_lib = pse_SL_lib,
                                  nuisance_estimates_input = nuisance_estimates_input,
                                  pred_data = newdata)
        },
        #if an error occurs, tell me the error
        error=function(e) {
          stop(paste("An error occured when fitting the pseudo outcome model in split ",i,sep=""))
          print(e)
        }
      )

      #--- Collecting pseudo-outcome regression models/preds ---#
      if (i==0){
        pse_mods <- list(pse_model)
      }
      else {
        pse_mods <- append(pse_mods,list(pse_model))
      }
    }
  }
  
  if (splits == 1 | splits == 10){
    tryCatch(
      {
        pse_model <- Pseudo_mod(data = po_data_all,
                                id = id,
                                outcome = outcome,
                                exposure = exposure,
                                pse_method = pse_method,
                                pse_covariates = pse_covariates,
                                pse_SL_lib = pse_SL_lib,
                                nuisance_estimates_input = nuisance_estimates_input,
                                pred_data = newdata)
      },
      #if an error occurs, tell me the error
      error=function(e) {
        stop(paste("An error occured when fitting the pseudo outcome model in split ",i,sep=""))
        print(e)
      }
    )
  }

  if (nuisance_estimates_input == 1){
    #Add splits to run pseudo-outcome model

  }



  #-----------------------------#
  #--- Returning information ---#
  #-----------------------------#
  if (splits == 1 | splits == 10){
    output <- list(CATE_est = pse_model$po_pred,
                   data = po_data_all)
  }
  else if (splits == 3){
    #Make cate est

    #Calculating average CATE estimate
    po_preds <- as.data.frame(rep(0,dim(newdata)[1]))
    for (i in 1:splits){
      po_preds <- as.data.frame(cbind(po_preds,pse_mods[[i]]$po_pred))
    }
    po_preds <- po_preds[,2:dim(po_preds)[2]]

    avg_CATE_est <- rowMeans(po_preds)

    output <- list(CATE_est = avg_CATE_est,
                   pse_preds = po_preds,
                   data = po_data_all)
  }
  
  return(output)
}



###############################################################

#Example 
DR_check <- DR_learner(analysis = "SL imputation",
                       data = check,
                       id = "ID",
                       outcome = "Y",
                       exposure = "A",
                       outcome_observed_indicator = "G_obs",
                       splits = 3,
                       e_covariates = c("X1","X2","X3"),
                       e_method = "Parametric",
                       e_SL_lib = c("SL.lm"),
                       e_SL_strat = TRUE,
                       out_method = "Parametric",
                       out_covariates = c("X1","X2","X3"),
                       out_SL_lib = c("SL.lm"),
                       out_SL_strat = FALSE,
                       pse_method = "Parametric",
                       pse_covariates = c("X1"),
                       pse_SL_lib = c("SL.lm"),
                       imp_covariates = c("X3","X5","X6"),
                       imp_SL_lib = c("SL.lm"),
                       imp_SL_strat = FALSE,
                       newdata = check)

DR_check <- DR_learner(analysis = "Complete case",
                       data = check,
                       id = "ID",
                       outcome = "Y",
                       exposure = "A",
                       outcome_observed_indicator = "G_obs",
                       splits = 3,
                       nuisance_estimates_input = 1,
                       o_0_pred = "Y.0_prob_true",
                       o_1_pred = "Y.1_prob_true",
                       e_pred = "prop_score",
                       pse_method = "Parametric",
                       pse_covariates = c("X1"),
                       pse_SL_lib = c("SL.lm"),
                       newdata = check)

#Known nuisance functions can only be input for 10 fold cross-fitting 


