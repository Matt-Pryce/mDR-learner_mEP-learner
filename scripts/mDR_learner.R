#######################################################################################
# Script: mDR-Learner function
# Date: 20/03/24
# Author: Matt Pryce 
# Notes: mDR-learner function for single time point setting
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
#' @param newdata New data to create predictions for

#' @return A list containing: CATE estimates, a dataset used to train the learner, dataset containing all 
#'         pseudo-outcome predictions (if splits 4) 


mDR_learner <- function(data,
                        id,
                        outcome,
                        exposure,
                        outcome_observed_indicator = "None",
                        splits = c(1,4,10),
                        e_method = c("Parametric","Random forest","Super Learner"),
                        e_covariates,
                        e_SL_lib,
                        out_method = c("Parametric","Random forest","Super Learner"),
                        out_covariates,
                        out_SL_lib,
                        g_method = c("Parametric","Random forest","Super Learner"),
                        g_covariates,
                        g_SL_lib,
                        nuisance_estimates_input = 0,
                        e_pred = NA,
                        o_0_pred = NA,
                        o_1_pred = NA,
                        g_pred = NA,
                        pse_method = c("Parametric","Random forest","Super Learner"),
                        pse_covariates,
                        pse_SL_lib,
                        newdata
){
  
  #-----------------------#
  #--- Data management ---#
  #-----------------------#
  
  clean_data <- data_manage_1tp(data = data,
                                learner = "mDR-learner",
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
                                o_0_pred = o_0_pred,
                                o_1_pred = o_1_pred,
                                e_pred = e_pred,
                                g_pred = g_pred,
                                newdata = newdata
  )
  
  analysis_data <- clean_data$data
  
  
  
  #------------------------------------------------------#
  #--- Running nuisance models & pseudo outcome model ---#
  #------------------------------------------------------#
  
  tryCatch(
    {
      #Checking number of splits is correct
      if (splits != 1 & splits != 4 & splits != 10){
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
          #Data for propensity score, outcome models & censoring models 
          e_data <- analysis_data
          o_data <- analysis_data
          g_data <- analysis_data
          
          if (splits == 4){
            e_data <- subset(e_data,e_data$s == i)
            o_data <- subset(o_data,o_data$s == ((i+1) %% 4))
            g_data <- subset(g_data,g_data$s == ((i+2) %% 4))
          }
          else if (splits == 1){
            e_data <- subset(e_data,e_data$s == i)
            o_data <- subset(o_data,o_data$s == i)
            g_data <- subset(g_data,g_data$s == i)
          }
          else if (splits == 10){
            e_data <- subset(e_data,e_data$s != i)
            o_data <- subset(o_data,o_data$s != i)
            g_data <- subset(g_data,g_data$s != i)
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
          if (splits == 4){
            po_e_data <- subset(analysis_data, select = c(e_covariates,"s"))
            po_e_data <- subset(po_e_data, po_e_data$s == ((i+3) %% 4))
            po_e_data <- as.matrix(subset(po_e_data, select = -c(s)))
            po_o_data <- subset(analysis_data,select = c(out_covariates,"s"))
            po_o_data <- subset(po_o_data, po_o_data$s == ((i+3) %% 4))
            po_o_data <- as.matrix(subset(po_o_data, select = -c(s)))
            po_g_data <- subset(analysis_data,select = c("A",g_covariates,"s"))
            po_g_data <- subset(po_g_data, po_g_data$s == ((i+3) %% 4))
            po_g_data <- as.matrix(subset(po_g_data, select = -c(s)))
            po_data <- subset(analysis_data,select = c("ID","Y","A","G",pse_covariates,"s"))
            po_data <- subset(po_data,po_data$s == ((i+3) %% 4))
          }
          else if (splits == 1 | splits == 10){
            po_e_data <- subset(analysis_data, select = c(e_covariates,"s"))
            po_e_data <- subset(po_e_data, po_e_data$s == i)
            po_e_data <- as.matrix(subset(po_e_data, select = -c(s)))
            po_o_data <- subset(analysis_data,select = c(out_covariates,"s"))
            po_o_data <- subset(po_o_data, po_o_data$s == i)
            po_o_data <- as.matrix(subset(po_o_data, select = -c(s)))
            po_g_data <- subset(analysis_data,select = c("A",g_covariates,"s"))
            po_g_data <- subset(po_g_data, po_g_data$s == i)
            po_g_data <- as.matrix(subset(po_g_data, select = -c(s)))
            po_data <- subset(analysis_data,select = c("ID","Y","A","G",pse_covariates,"s"))
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
      
      
      #--- Collecting nuisance model predictions ---#
      tryCatch(
        {
          po_data <- cbind(po_data,o_1_pred = outcome_models$o_mod_pred_1)
          po_data <- cbind(po_data,o_0_pred = outcome_models$o_mod_pred_0)
          po_data <- cbind(po_data,e_pred = PS_model$e_pred)
          po_data <- cbind(po_data,g_pred = cen_model$g_pred)
        },
        #if an error occurs, tell me the error
        error=function(e) {
          stop(paste("An error occured when outcome model function in split ",i,sep=""))
          print(e)
        }
      )
      
    }
    else if (nuisance_estimates_input == 1){

      if (splits == 4){
        po_data <- analysis_data
        po_data <- subset(po_data,po_data$s == ((i+3) %% 4))
      }
      else if (splits == 1 | splits == 10){
        po_data <- analysis_data
        po_data <- subset(po_data,po_data$s == i)
      }
    }
    
    
    #--- Calculating pseudo-outcomes ---#
    tryCatch(
      {
        #--- Calculating pseudo outcome ---#
        #First setting missing values to 99
        po_data <- po_data %>% mutate_if(is.numeric, function(x) ifelse(is.na(x), 99, x))
        
        #Calculating outcome
        po_data$pse_Y <- ((po_data$A - po_data$e_pred)/(po_data$e_pred*(1-po_data$e_pred))) *
          (po_data$G/po_data$g_pred) *
          (po_data$Y - (po_data$A*po_data$o_1_pred + (1-po_data$A) * po_data$o_0_pred)) +
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
    
    
    #--- Running pseudo-outcome regression (if 4 split option chosen) ---#
    if (splits == 4){
      tryCatch(
        {
          pse_model <- nuis_mod(model = "Pseudo outcome",
                                data = po_data,
                                method = pse_method,
                                covariates = pse_covariates,
                                SL_lib = pse_SL_lib,
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
        pse_model <- nuis_mod(model = "Pseudo outcome",
                              data = po_data_all,
                              method = pse_method,
                              covariates = pse_covariates,
                              SL_lib = pse_SL_lib,
                              pred_data = newdata)
      },
      #if an error occurs, tell me the error
      error=function(e) {
        stop(paste("An error occured when fitting the pseudo outcome model in split ",i,sep=""))
        print(e)
      }
    )
  }
  
  #-----------------------------#
  #--- Returning information ---#
  #-----------------------------#
  if (splits == 1 | splits == 10){
    output <- list(CATE_est = pse_model$po_pred,
                   data = po_data_all)
  }
  else if (splits == 4){
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
# 
# #Example
mDR_check <- mDR_learner(data = check,
                         id = "ID",
                         outcome = "Y",
                         exposure = "A",
                         outcome_observed_indicator = "G_obs",
                         splits = 4,
                         e_covariates = c("X1","X2","X3"),
                         e_method = "Parametric",
                         e_SL_lib = c("SL.lm"),
                         out_method = "Parametric",
                         out_covariates = c("X1","X2","X3"),
                         out_SL_lib = c("SL.lm"),
                         g_covariates = c("X1","X2","X3"),
                         g_method = "Super learner",
                         g_SL_lib = c("SL.lm"),
                         pse_method = "Parametric",
                         pse_covariates = c("X1"),
                         pse_SL_lib = c("SL.lm"),
                         newdata = check)
# 
# mDR_check <- mDR_learner(data = check,
#                          id = "ID",
#                          outcome = "Y",
#                          exposure = "A",
#                          outcome_observed_indicator = "G_obs",
#                          splits = 10,
#                          nuisance_estimates_input = 1,
#                          o_0_pred = "Y.0_prob_true",
#                          o_1_pred = "Y.1_prob_true",
#                          e_pred = "prop_score",
#                          g_pred = "G_prob",
#                          pse_method = "Parametric",
#                          pse_covariates = c("X1"),
#                          pse_SL_lib = c("SL.lm"),
#                          newdata = check)
