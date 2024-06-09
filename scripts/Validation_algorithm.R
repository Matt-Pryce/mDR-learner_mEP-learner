#######################################################################################
# Script: De-biased MSE metric when outcome data is missing 
# Date (Created): 09/06/24
# Author: Matt Pryce 
# Notes: 
#######################################################################################

#Loading libraries needed
library(caTools)
library(tidyverse)
library(dplyr)
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

source("C:/Users/MatthewPryce/OneDrive - London School of Hygiene and Tropical Medicine/Documents/PhD/DR_Missing_Paper/GitHub_rep/mDR-learner/src/Data_management_1tp.R")
source("C:/Users/MatthewPryce/OneDrive - London School of Hygiene and Tropical Medicine/Documents/PhD/DR_Missing_Paper/GitHub_rep/mDR-learner/src/nuisance_models.R")


#Process
#  1) Load and check & transform input data 
#  2) Estimate nuisance function estimates on 9/10 data, predict for other 1/10
#  3) (Start loop) Construct clever covariates using pi, g, mu0 & m1
#  4) Obtain updated outcome functions using lin reg of outcome and clever cov (End loop)
#  5) Calculate debiased MSE


################################
#--- Input/output variables ---#
################################

#' @param data The data frame containing all required information
#' @param id Identification for individuals
#' @param outcome The name of the outcome of interest
#' @param exposure The name of the exposure of interest
#' @param outcome_observed_indicator The name of the indicator identifying when the outcome variable is observed (1=observed, 0=missing)
#' @param CATE_est A vector of CATE estimates to be evaluated
#' @param splits Number of splits to be used in cross-fitting procedure 
#' @param e_method Statistical technique used to run the propensity score model
#' @param e_covariates List containing the names of the variables to be input into the propensity score model
#' @param e_SL_lib Library to be used in super learner if selected for propensity score model
#' @param out_method Statistical technique used to run the outcome models
#' @param out_covariates  List containing the names of the variables to be input into each outcome model
#' @param out_SL_lib Library to be used in super learner if selected for outcome models 
#' @param g_method Statistical technique used to run the missingness model
#' @param g_covariates List containing the names of the variables to be input into the missingness model, excluding exposure
#' @param g_SL_lib Library to be used in super learner if selected for missingness model
#' @param iterations Number of clever covariate iteratiosn to be applied when implementing TMLE approach   #NOT CURRENT A CHOICE

#' @return An estimate of the debiased MSE 



debiased_MSE <- function(data,
                    id,
                    outcome,
                    exposure,
                    outcome_observed_indicator,
                    CATE_est,
                    splits = 10,
                    e_method = c("Parametric","Random forest","Super Learner"),
                    e_covariates,
                    e_SL_lib,
                    out_method = c("Parametric","Random forest","Super Learner"),
                    out_covariates,
                    out_SL_lib,
                    g_method = c("Parametric","Random forest","Super Learner"),
                    g_covariates,
                    g_SL_lib
){
  
  #--------------------#
  #--- Loading data ---#
  #--------------------#
  
  tryCatch(
    {
      clean_data <- data_manage_1tp(data = data,
                                    learner = "debiased_MSE",
                                    analysis_type = analysis,
                                    nuisance_estimates_input = 0,
                                    id = id,
                                    outcome = outcome,
                                    exposure = exposure,
                                    outcome_observed_indicator = outcome_observed_indicator,
                                    CATE_est = CATE_est,
                                    e_covariates = e_covariates,
                                    out_covariates = out_covariates,
                                    g_covariates = g_covariates,
                                    pse_covariates = pse_covariates)
      
      analysis_data <- clean_data$data
    },
    #if an error occurs, tell me the error
    error=function(e) {
      stop('An error occured when formatting the data')
      print(e)
    }
  )
  
  
  #-------------------------------#
  #--- Running nuisance models ---#
  #-------------------------------#
  tryCatch(
    {
      #Creating splits
      analysis_data$s <- rep(1:length(analysis_data$Y),1) %% splits
    },
    #if an error occurs, tell me the error
    error=function(e) {
      stop('An error occured when splitting the data')
      print(e)
    }
  )

  all_output <- list()

  # Iterating over each split
  for (i in 0:(splits-1)){
    #--- Collecting data for training models ---#
    tryCatch(
      {
        e_data <- analysis_data
        o_data <- analysis_data
        o_data <- subset(o_data,is.na(o_data$Y) == 0)
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
          if (clean_data$Y_bin == 1){
            po_data <- subset(analysis_data,select = c("ID","Y","A","G","CATE_est","s"))
          }
          else {
            po_data <- subset(analysis_data,select = c("ID","Y","Y_norm","A","G","CATE_est","CATE_est_norm","s"))
          }
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
          if (clean_data$Y_bin == 1){
            po_data <- subset(analysis_data,select = c("ID","Y","A","G","CATE_est","s"))
          }
          else {
            po_data <- subset(analysis_data,select = c("ID","Y","Y_norm","A","G","CATE_est","CATE_est_norm","s"))
          }
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
        outcome_models <- nuis_mod(model = "Outcome - MSE",
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
    
    #--- Collecting full test data with pseudo-outcomes ---#
    if (i==0){
      po_data_all <- po_data
    }
    else {
      po_data_all <- rbind(po_data_all,po_data)
    }
  }

  #--- Beginning TMLE iterations ---#
  MSE_data_list <- list()
  MSE_est_list <- list()
  MSE_model_list <- list()

  MSE_conv <- 0
  iter <- 1
  while (MSE_conv == 0){
    #Identifying up to date outcome functions
    if (iter == 1){   #If first iteration, set current outcome estimates as the estimates from the nuisance models
      po_data_all$cur_o_0_pred <- po_data_all$o_0_pred
      po_data_all$cur_o_1_pred <- po_data_all$o_1_pred
    }
    else {  #If iteration > 1, set current outcome estimates as the updated predictions
      po_data_all$cur_o_0_pred <- po_data_all$new_o_0_pred
      po_data_all$cur_o_1_pred <- po_data_all$new_o_1_pred
    }

    #--- Creating clever covariates ---#
    #H_A - Clever covariate based on observed exposure
    if (clean_data$Y_bin == 1){
      po_data_all <- po_data_all %>%  mutate(H_A = case_when(A == 0 ~ -(1/((1-e_pred)*g_pred))*(CATE_est - cur_o_1_pred + cur_o_0_pred),
                                                             A == 1 ~ (1/(e_pred*g_pred))*(CATE_est - cur_o_1_pred + cur_o_0_pred),
                                                             TRUE ~ as.numeric(NA)))
    }
    else {
      po_data_all <- po_data_all %>%  mutate(H_A = case_when(A == 0 ~ -(1/((1-e_pred)*g_pred))*(CATE_est_norm - cur_o_1_pred + cur_o_0_pred),
                                                             A == 1 ~ (1/(e_pred*g_pred))*(CATE_est_norm - cur_o_1_pred + cur_o_0_pred),
                                                             TRUE ~ as.numeric(NA)))
    }

    #H_1 - Clever covariate if exposed
    if (clean_data$Y_bin == 1){
      po_data_all$H_1 <- (1/(po_data_all$e_pred*po_data_all$g_pred))*(po_data_all$CATE_est - po_data_all$cur_o_1_pred + po_data_all$cur_o_0_pred)
    }
    else {
      po_data_all$H_1 <- (1/(po_data_all$e_pred*po_data_all$g_pred))*(po_data_all$CATE_est_norm - po_data_all$cur_o_1_pred + po_data_all$cur_o_0_pred)
    }

    #H_0 - Clever covariate if unexposed
    if (clean_data$Y_bin == 1){
      po_data_all$H_0 <- -(1/((1-po_data_all$e_pred)*po_data_all$g_pred))*(po_data_all$CATE_est - po_data_all$cur_o_1_pred + po_data_all$cur_o_0_pred)
    }
    else {
      po_data_all$H_0 <- -(1/((1-po_data_all$e_pred)*po_data_all$g_pred))*(po_data_all$CATE_est_norm - po_data_all$cur_o_1_pred + po_data_all$cur_o_0_pred)
    }


    #--- Updating outcome functions ---#   
    #Defining outcome prediction based on the observed treatment
    po_data_all <- po_data_all %>%  mutate(cur_o_A_pred = case_when(A == 0 ~ cur_o_0_pred,
                                                                    A == 1 ~ cur_o_1_pred,
                                                                    TRUE ~ as.numeric(NA)))

    #Running model to produce fluctuation parameter
    fluc_mod <- glm(Y_norm ~ -1 + offset(qlogis(cur_o_A_pred)) + H_A, data=po_data_all, family=quasibinomial)

    #Collecting fluctuation parameter
    eps <- coef(fluc_mod)

    #Creating updated outcome estimates
    po_data_all$new_o_0_pred <- plogis(qlogis(po_data_all$cur_o_0_pred) + eps*po_data_all$H_0)
    po_data_all$new_o_1_pred <- plogis(qlogis(po_data_all$cur_o_1_pred) + eps*po_data_all$H_1)

    #Defining outcome prediction based on the observed treatment
    po_data_all <- po_data_all %>%  mutate(new_o_A_pred = case_when(A == 0 ~ new_o_0_pred,
                                                                    A == 1 ~ new_o_1_pred,
                                                                    TRUE ~ as.numeric(NA)))


    #--- Calculate MSE ---#   #Only for transformed version (i.e. Y_cont == 1)
    tryCatch(
      {
        #Summing pseudo outcomes
        MSE <-  mean((((po_data_all$CATE_est_norm - po_data_all$new_o_1_pred + po_data_all$new_o_0_pred)*(clean_data$max_Y-clean_data$min_Y))+clean_data$min_Y)^2)
      },
      #if an error occurs, tell me the error
      error=function(e) {
        stop("An error occured when calculating the MSE")
        print(e)
      }
    )

    #--- Storing models and data ---#
    MSE_data_list <- append(MSE_data_list,list(po_data_all))
    MSE_est_list <- append(MSE_est_list,list(MSE))

    if (iter > 2){
      if (abs((MSE_est_list[[iter]]-MSE_est_list[[iter-1]])/MSE_est_list[[iter-1]])<0.05 &
          abs((MSE_est_list[[iter-1]]-MSE_est_list[[iter-2]])/MSE_est_list[[iter-2]])<0.05){
        MSE_conv <- 1
      }
    }
    else {
      MSE_conv <- 0
    }

    if (iter > 20){
      return("MSE did no converge within 20 iterations")
    }
    
    #Updating iteration
    iter <- iter + 1
    



  }

  #--- Calculating SE & CIs ---#

  #Transforming outcomes back to original scale
  po_data_all$final_o_1_pred <- ((po_data_all$new_o_1_pred*(clean_data$max_Y-clean_data$min_Y))+clean_data$min_Y)
  po_data_all$final_o_0_pred <- ((po_data_all$new_o_0_pred*(clean_data$max_Y-clean_data$min_Y))+clean_data$min_Y)
  po_data_all$final_o_A_pred <- ((po_data_all$new_o_A_pred*(clean_data$max_Y-clean_data$min_Y))+clean_data$min_Y)

  #Set missing values to non-missing so we can calculate IF
  po_data_all <- po_data_all %>% mutate_if(is.numeric, function(x) ifelse(is.na(x), 999, x))

  #Calculating IF
  infl_fn <- (po_data_all$CATE_est - (po_data_all$final_o_1_pred - po_data_all$final_o_0_pred))^2 -
    (2*(((po_data_all$A*po_data_all$G)/(po_data_all$e_pred*po_data_all$g_pred)) +
          (((1-po_data_all$A)*po_data_all$G)/((1-po_data_all$e_pred)*po_data_all$g_pred))) *
       (po_data_all$Y - po_data_all$final_o_A_pred) *
       (po_data_all$CATE_est - po_data_all$final_o_1_pred + po_data_all$final_o_0_pred)) - MSE

  #Obtaining variance
  sample_size <- length(infl_fn)
  varHat.IC <- var(infl_fn, na.rm = TRUE)/sample_size

  #Calculating CI
  MSE_LCI <- MSE - 1.96*sqrt(varHat.IC)
  if (MSE_LCI<0){
    MSE_LCI <- 0
  }
  MSE_UCI <- MSE + 1.96*sqrt(varHat.IC)

  output <- list(MSE=MSE,
                 MSE_LCI=MSE_LCI,
                 MSE_UCI=MSE_UCI,
                 MSE_list=MSE_est_list,
                 Updated_data=MSE_data_list,
                 Final_data=po_data_all)
  return(output)
}


# out_cov_list <- c("age","wtkg","race","gender","hemo","homo","drugs","karnof","symptom","cd40","cd420","cd80","cd820","preanti","oprior")
# imp_cov_list <- c("age","wtkg","race","gender","hemo","homo","drugs","karnof","symptom","cd40","cd420","cd80","cd820","preanti","oprior")
# ps_cov_list <- c("age","wtkg","race","gender","hemo","homo","drugs","karnof","symptom","cd40","cd80","preanti","oprior")
# 
# #Note: To run, load data, create tau, run cov lists and libs
# 
# 
# out_lib <- c("SL.mean",
#              "SL.lm")#,
#              # "SL.glmnet_8", "SL.glmnet_9",
#              # "SL.glmnet_11", "SL.glmnet_12",
#              # "SL.ranger_1","SL.ranger_2","SL.ranger_3",
#              # "SL.ranger_4","SL.ranger_5","SL.ranger_6",
#              # "SL.nnet_1","SL.nnet_2","SL.nnet_3",
#              # "SL.svm_1",
#              # "SL.kernelKnn_4","SL.kernelKnn_10")
# 
# e_lib <- c("SL.mean",
#            "SL.glm")#,
#            # "SL.glmnet_8", "SL.glmnet_9",
#            # "SL.glmnet_11", "SL.glmnet_12",
#            # "SL.ranger_1","SL.ranger_2","SL.ranger_3",
#            # "SL.ranger_4","SL.ranger_5","SL.ranger_6",
#            # "SL.nnet_1","SL.nnet_2","SL.nnet_3",
#            # "SL.svm_1",
#            # "SL.kernelKnn_4","SL.kernelKnn_10")
# 
# g_lib <- c("SL.mean",
#            "SL.glm")#,
#            # "SL.glmnet_8", "SL.glmnet_9",
#            # "SL.glmnet_11", "SL.glmnet_12",
#            # "SL.ranger_1","SL.ranger_2","SL.ranger_3",
#            # "SL.ranger_4","SL.ranger_5","SL.ranger_6",
#            # "SL.nnet_1","SL.nnet_2","SL.nnet_3",
#            # "SL.svm_1",
#            # "SL.kernelKnn_4","SL.kernelKnn_10")
# 
# debiased_MSE_IF_check <- debiased_MSE(data = ACTG175_data,
#                                       id = "pidnum",
#                                       outcome = "cd496",
#                                       exposure = "treat",
#                                       outcome_observed_indicator = "r",
#                                       CATE_est = "tau",
#                                       splits = 10,
#                                       e_method = "Parametric",
#                                       e_covariates = ps_cov_list,
#                                       e_SL_lib = e_lib,
#                                       out_method = "Super learner",
#                                       out_covariates = out_cov_list,
#                                       out_SL_lib = out_lib,
#                                       g_method = "Parametric",
#                                       g_covariates = imp_cov_list,
#                                       g_SL_lib = g_lib)


