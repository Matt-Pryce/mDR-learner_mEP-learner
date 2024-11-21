
######################################################################
# Script: Simulations script for Specification 1
# Author: Matt Pryce
# Date: 12/08/24
# Notes:
######################################################################

#--- Loading libraries needed ---# 
library(base)
library(MASS)
library(tidyverse)
library(stringr)
library(lava)
library(reshape2)
library(data.table)
library(caTools)
library(ggplot2)
library(DAAG)
library(glmnet)
library(randomForest)
library(caret)
library(grf)
library(xgboost)
library(SuperLearner)
library(ranger)
library(KernelKnn)
library(nnet)
library(e1071)
library(Sieve)


#--- Defining which simulation is being run ---#
Sys.setenv(TMPDIR = "H:/HPC/Temp")                #Setting up HPC (will vary depending on the computer it is run on)
taskID <- Sys.getenv("SLURM_ARRAY_TASK_ID")
taskID <- as.numeric(taskID)

#--- Loading parameter info ---#
load("parameters_log.RData")     #Loads stored parameter values for each DGP

#--- Defining parameter values for the specification ---#
scenario <- 1     #This refers to DGP 1/2/3 in the simulations
setting <- 1      #This refers to the sample size 400/800/1600/3200 with 1-4 respectively
n  <- parameters_log$sample_size[setting]
covs  <- parameters_log$num_covs[setting]
ps_spec  <- parameters_log$ps_list[setting]
cen_spec  <- parameters_log$cen_list[setting]
Y0_spec  <- parameters_log$Y0_list[setting]
tau_spec  <- parameters_log$tau_list[setting]


#--- Loading and defining functions to generate data ---#
ps_list <- readRDS("ps_funcs.RData")
ps_func <- ps_list[[ps_spec]]

cen_list <- readRDS("cen_funcs.RData")
cen_func <- cen_list[[cen_spec]]

Y0_list <- readRDS("Y0_funcs.RData")
Y0_func <- Y0_list[[Y0_spec]]

tau_list <- readRDS("tau_funcs.RData")
tau_func <- tau_list[[tau_spec]]


#--- Loading test data ---#
test_data_name <- paste("spec_",scenario,"_test_data.RData",sep="")    #Loads a test dataset of sample size 10,000 for the given DGP
load(test_data_name)


#--- Creating learners for SL library's ---#
#LASSO & elastic net
nlambda_seq = c(50,100,250)
alpha_seq <- c(0.5,1)
usemin_seq <- c(FALSE,TRUE)
para_learners = create.Learner("SL.glmnet", tune = list(nlambda = nlambda_seq,alpha = alpha_seq,useMin = usemin_seq))
para_learners

#Random forest - One covariate
mtry_seq1 <-  1
min_node_seq <- c(10,20,50)
rf_learners1 = create.Learner("SL.ranger", tune = list(mtry = mtry_seq1, min.node.size = min_node_seq))
rf_learners1

mtry_seq6 <-  floor(sqrt(6) * c(0.5, 1))
min_node_seq <- c(10,20,50)
rf_learners6 = create.Learner("SL.ranger", tune = list(mtry = mtry_seq6, min.node.size = min_node_seq))
rf_learners6


#Nnet (single layer neural nets)
size_seq <- c(1,2,5)
nnet_learners <- create.Learner("SL.nnet",tune = list(size = size_seq))

#SVM (Support vector machine)
nu_seq <- c(1)
type_seq <- c("C-classification")
svm_learners = create.Learner("SL.svm",tune = list(type.class = type_seq))

#KernelKnn
K_seq <- c(5,10,20)
h_seq <- c(0.01,0.05,0.1,0.25)
KernelKnn_learners <- create.Learner("SL.kernelKnn",tune = list(k = K_seq, h = h_seq))



#--- Loading data generation and learner functions ---#
#Data generation
source("Data_simulation_functions.R")

#Scripts needed for learners
source("Data_management_1tp.R")
source("nuisance_models.R")

#Learner
source("T_learner.R")
source("DR_learner.R")
source("EP_learner.R")


# #--- Loading sieve basis list ---#
# load(paste("Sieve_basis_list",setting,".RData",sep=""))    #Can be used if Sieve package unavaiable on HPC 


#--- Running loop ---#

# Loop: 1) Set seed
#       2) Generate data
#       3) Run models & store information


start <- Sys.time()

#Setting up temporary list to store model info for the simulation in the spec
model_info_list <- list()

#Number of simulations
sims <- 1

seed <- taskID      #This will vary depending on the iteration run on the HPC 
set.seed(taskID)

#Starting loop
for (i in 1:sims){
  
  #Setting up temporary list to store model info for the simulation in the script
  model_info_sim_list <- list()

  #Identifying scenario
  scen <- scenario

  #-----------------------#
  #--- Simulating data ---#
  #-----------------------#

  #Generate dataset
  sim_data_train <- data_sim_uni_miss(n = n, x = covs, x_form = "Unif",
                                      e = ps_func,
                                      Y_0 = Y0_func,
                                      Tau = tau_func,
                                      G=cen_func,
                                      error = "Normal",
                                      outcome_form = "Countinuous")

  model_info_sim_list <- append(model_info_sim_list,list(sim_data_train = sim_data_train))
  model_info_sim_list <- append(model_info_sim_list,list(sim_data_test = sim_data_test))

  #Creating covariate lists
  cov_list <- NULL
  for (cov in 1:covs){
    cov_list <- append(cov_list,paste("X",cov,sep=""))
  }



  #-----------------------------------------#
  #--- Running T learner (Complete case) ---#
  #-----------------------------------------#
  tryCatch(
    {
      out_lib <- c("SL.mean",
                   "SL.lm",
                   "SL.glmnet_8", "SL.glmnet_9",
                   "SL.glmnet_11", "SL.glmnet_12",
                   "SL.ranger_1","SL.ranger_2","SL.ranger_3",
                   "SL.ranger_4","SL.ranger_5","SL.ranger_6",
                   "SL.nnet_1","SL.nnet_2","SL.nnet_3",
                   "SL.svm_1",
                   "SL.kernelKnn_4","SL.kernelKnn_5","SL.kernelKnn_6",
                   "SL.kernelKnn_10","SL.kernelKnn_11","SL.kernelKnn_12")

      T_learner_CC <- T_learner(analysis = "Complete case",
                                data = sim_data_train,
                                id = "ID",
                                outcome = "Y",
                                exposure = "A",
                                outcome_observed_indicator = "G_obs",
                                out_method = "Super learner",
                                out_covariates = cov_list,
                                out_SL_lib = out_lib,
                                newdata = sim_data_test)

      model_info_sim_list <- append(model_info_sim_list,list(T_CC = T_learner_CC))
    },
    error=function(e) {
      message(paste("An error occured when fitting the T-learner (complete case) for scenario ",scen," with sample size ",n," in simulation ",seed,sep=""))
      print(e)
    }
  )


  #-----------------------------------------#
  #--- Running T learner (SL imputation) ---#
  #-----------------------------------------#
  tryCatch(
    {
      out_lib <- c("SL.mean",
                   "SL.lm",
                   "SL.glmnet_8", "SL.glmnet_9",
                   "SL.glmnet_11", "SL.glmnet_12",
                   "SL.ranger_1","SL.ranger_2","SL.ranger_3",
                   "SL.ranger_4","SL.ranger_5","SL.ranger_6",
                   "SL.nnet_1","SL.nnet_2","SL.nnet_3",
                   "SL.svm_1",
                   "SL.kernelKnn_4","SL.kernelKnn_5","SL.kernelKnn_6",
                   "SL.kernelKnn_10","SL.kernelKnn_11","SL.kernelKnn_12")

      imp_lib <- c("SL.mean",
                   "SL.glm",
                   "SL.glmnet_8", "SL.glmnet_9",
                   "SL.glmnet_11", "SL.glmnet_12",
                   "SL.ranger_1","SL.ranger_2","SL.ranger_3",
                   "SL.ranger_4","SL.ranger_5","SL.ranger_6",
                   "SL.nnet_1","SL.nnet_2","SL.nnet_3",
                   "SL.svm_1",
                   "SL.kernelKnn_4","SL.kernelKnn_5","SL.kernelKnn_6",
                   "SL.kernelKnn_10","SL.kernelKnn_11","SL.kernelKnn_12")

      T_learner_imp <- T_learner(analysis = "SL imputation",
                                 data = sim_data_train,
                                 id = "ID",
                                 outcome = "Y",
                                 exposure = "A",
                                 outcome_observed_indicator = "G_obs",
                                 out_method = "Super learner",
                                 out_covariates = cov_list,
                                 out_SL_lib = out_lib,
                                 imp_covariates = cov_list,
                                 imp_SL_lib = imp_lib,
                                 newdata = sim_data_test)

      model_info_sim_list <- append(model_info_sim_list,list(T_imp = T_learner_imp))
    },
    error=function(e) {
      message(paste("An error occured when fitting the T-learner (SL imputation) for scenario ",scen," with sample size ",n," in simulation ",seed,sep=""))
      print(e)
    }
  )



  #---------------------------------------------------------#
  #--- Running DR learner - 10 CV folds (Available case) ---#
  #---------------------------------------------------------#
  tryCatch(
    {
      out_lib <- c("SL.mean",
                   "SL.lm",
                   "SL.glmnet_8", "SL.glmnet_9",
                   "SL.glmnet_11", "SL.glmnet_12",
                   "SL.ranger_1","SL.ranger_2","SL.ranger_3",
                   "SL.ranger_4","SL.ranger_5","SL.ranger_6",
                   "SL.nnet_1","SL.nnet_2","SL.nnet_3",
                   "SL.svm_1",
                   "SL.kernelKnn_4","SL.kernelKnn_10")
      pse_lib <- out_lib
      e_lib <- c("SL.mean",
                 "SL.glm",
                 "SL.glmnet_8", "SL.glmnet_9",
                 "SL.glmnet_11", "SL.glmnet_12",
                 "SL.ranger_1","SL.ranger_2","SL.ranger_3",
                 "SL.ranger_4","SL.ranger_5","SL.ranger_6",
                 "SL.nnet_1","SL.nnet_2","SL.nnet_3",
                 "SL.svm_1",
                 "SL.kernelKnn_4","SL.kernelKnn_10")

      DR_learner_10_AC <- DR_learner(analysis = "Available case",
                                     data = sim_data_train,
                                     id = "ID",
                                     outcome = "Y",
                                     exposure = "A",
                                     outcome_observed_indicator = "G_obs",
                                     splits = 10,
                                     e_method = "Super learner",
                                     e_covariates = cov_list,
                                     e_SL_lib = e_lib,
                                     out_method = "Super learner",
                                     out_covariates = cov_list,
                                     out_SL_lib = out_lib,
                                     pse_method = "Super learner",
                                     pse_covariates = cov_list,
                                     pse_SL_lib = pse_lib,
                                     newdata = sim_data_test)

      model_info_sim_list <- append(model_info_sim_list,list(DR10_AC = DR_learner_10_AC))
    },
    #if an error occurs, tell me the error
    error=function(e) {
      message(paste("An error occured when fitting the DR-learner - 10 CV folds (Available case) for scenario ",scen," with sample size ",n," in simulation ",seed,sep=""))
      print(e)
    }
  )



  #--------------------------------------------------------#
  #--- Running DR learner - 10 CV folds (SL imputation) ---#
  #--------------------------------------------------------#
  tryCatch(
    {
      out_lib <- c("SL.mean",
                   "SL.lm",
                   "SL.glmnet_8", "SL.glmnet_9",
                   "SL.glmnet_11", "SL.glmnet_12",
                   "SL.ranger_1","SL.ranger_2","SL.ranger_3",
                   "SL.ranger_4","SL.ranger_5","SL.ranger_6",
                   "SL.nnet_1","SL.nnet_2","SL.nnet_3",
                   "SL.svm_1",
                   "SL.kernelKnn_4","SL.kernelKnn_10")
      pse_lib <- out_lib
      e_lib <- c("SL.mean",
                 "SL.glm",
                 "SL.glmnet_8", "SL.glmnet_9",
                 "SL.glmnet_11", "SL.glmnet_12",
                 "SL.ranger_1","SL.ranger_2","SL.ranger_3",
                 "SL.ranger_4","SL.ranger_5","SL.ranger_6",
                 "SL.nnet_1","SL.nnet_2","SL.nnet_3",
                 "SL.svm_1",
                 "SL.kernelKnn_4","SL.kernelKnn_10")
      imp_lib <- c("SL.mean",
                   "SL.glm",
                   "SL.glmnet_8", "SL.glmnet_9",
                   "SL.glmnet_11", "SL.glmnet_12",
                   "SL.ranger_1","SL.ranger_2","SL.ranger_3",
                   "SL.ranger_4","SL.ranger_5","SL.ranger_6",
                   "SL.nnet_1","SL.nnet_2","SL.nnet_3",
                   "SL.svm_1",
                   "SL.kernelKnn_4","SL.kernelKnn_5","SL.kernelKnn_6",
                   "SL.kernelKnn_10","SL.kernelKnn_11","SL.kernelKnn_12")

      DR_learner_10_imp <- DR_learner(analysis = "SL imputation",
                                      data = sim_data_train,
                                      id = "ID",
                                      outcome = "Y",
                                      exposure = "A",
                                      outcome_observed_indicator = "G_obs",
                                      splits = 10,
                                      e_method = "Super learner",
                                      e_covariates = cov_list,
                                      e_SL_lib = e_lib,
                                      out_method = "Super learner",
                                      out_covariates = cov_list,
                                      out_SL_lib = out_lib,
                                      g_covariates = cov_list,
                                      g_method = "Super learner",
                                      g_SL_lib = g_lib,
                                      imp_covariates = cov_list,
                                      imp_SL_lib = imp_lib,
                                      pse_method = "Super learner",
                                      pse_covariates = cov_list,
                                      pse_SL_lib = pse_lib,
                                      newdata = sim_data_test)

      model_info_sim_list <- append(model_info_sim_list,list(DR10_imp = DR_learner_10_imp))
    },
    #if an error occurs, tell me the error
    error=function(e) {
      message(paste("An error occured when fitting the DR-learner - 10 CV folds (SL imputation) for scenario ",scen," with sample size ",n," in simulation ",seed,sep=""))
      print(e)
    }
  )



  #---------------------------------------------------------#
  #--- Running EP learner - 10 CV folds (Available case) ---#
  #---------------------------------------------------------#
  tryCatch(
    {
      out_lib <- c("SL.mean",
                   "SL.lm",
                   "SL.glmnet_8", "SL.glmnet_9",
                   "SL.glmnet_11", "SL.glmnet_12",
                   "SL.ranger_1","SL.ranger_2","SL.ranger_3",
                   "SL.ranger_4","SL.ranger_5","SL.ranger_6",
                   "SL.nnet_1","SL.nnet_2","SL.nnet_3",
                   "SL.svm_1",
                   "SL.kernelKnn_4","SL.kernelKnn_10")
      pse_lib <- out_lib
      e_lib <- c("SL.mean",
                 "SL.glm",
                 "SL.glmnet_8", "SL.glmnet_9",
                 "SL.glmnet_11", "SL.glmnet_12",
                 "SL.ranger_1","SL.ranger_2","SL.ranger_3",
                 "SL.ranger_4","SL.ranger_5","SL.ranger_6",
                 "SL.nnet_1","SL.nnet_2","SL.nnet_3",
                 "SL.svm_1",
                 "SL.kernelKnn_4","SL.kernelKnn_10")

      EP_learner_10_AC <- EP_learner(analysis = "Available case",
                                     data = sim_data_train,
                                     id = "ID",
                                     outcome = "Y",
                                     exposure = "A",
                                     outcome_observed_indicator = "G_obs",
                                     splits = 10,
                                     e_method = "Super learner",
                                     e_covariates = cov_list,
                                     e_SL_lib = e_lib,
                                     out_method = "Super learner",
                                     out_covariates = cov_list,
                                     out_SL_lib = out_lib,
                                     pse_method = "Super learner",
                                     pse_covariates = cov_list,
                                     pse_SL_lib = pse_lib,
                                     newdata = sim_data_test)

      model_info_sim_list <- append(model_info_sim_list,list(EP10_AC = EP_learner_10_AC))
    },
    #if an error occurs, tell me the error
    error=function(e) {
      message(paste("An error occured when fitting the EP-learner - 10 CV folds (Available case) for scenario ",scen," with sample size ",n," in simulation ",seed,sep=""))
      print(e)
    }
  )



  #--------------------------------------------------------#
  #--- Running EP learner - 10 CV folds (SL imputation) ---#
  #--------------------------------------------------------#
  tryCatch(
    {
      out_lib <- c("SL.mean",
                   "SL.lm",
                   "SL.glmnet_8", "SL.glmnet_9",
                   "SL.glmnet_11", "SL.glmnet_12",
                   "SL.ranger_1","SL.ranger_2","SL.ranger_3",
                   "SL.ranger_4","SL.ranger_5","SL.ranger_6",
                   "SL.nnet_1","SL.nnet_2","SL.nnet_3",
                   "SL.svm_1",
                   "SL.kernelKnn_4","SL.kernelKnn_10")
      pse_lib <- out_lib
      e_lib <- c("SL.mean",
                 "SL.glm",
                 "SL.glmnet_8", "SL.glmnet_9",
                 "SL.glmnet_11", "SL.glmnet_12",
                 "SL.ranger_1","SL.ranger_2","SL.ranger_3",
                 "SL.ranger_4","SL.ranger_5","SL.ranger_6",
                 "SL.nnet_1","SL.nnet_2","SL.nnet_3",
                 "SL.svm_1",
                 "SL.kernelKnn_4","SL.kernelKnn_10")
      imp_lib <- c("SL.mean",
                   "SL.glm",
                   "SL.glmnet_8", "SL.glmnet_9",
                   "SL.glmnet_11", "SL.glmnet_12",
                   "SL.ranger_1","SL.ranger_2","SL.ranger_3",
                   "SL.ranger_4","SL.ranger_5","SL.ranger_6",
                   "SL.nnet_1","SL.nnet_2","SL.nnet_3",
                   "SL.svm_1",
                   "SL.kernelKnn_4","SL.kernelKnn_5","SL.kernelKnn_6",
                   "SL.kernelKnn_10","SL.kernelKnn_11","SL.kernelKnn_12")

      EP_learner_10_imp <- EP_learner(analysis = "SL imputation",
                                      data = sim_data_train,
                                      id = "ID",
                                      outcome = "Y",
                                      exposure = "A",
                                      outcome_observed_indicator = "G_obs",
                                      splits = 10,
                                      e_method = "Super learner",
                                      e_covariates = cov_list,
                                      e_SL_lib = e_lib,
                                      out_method = "Super learner",
                                      out_covariates = cov_list,
                                      out_SL_lib = out_lib,
                                      g_covariates = cov_list,
                                      g_method = "Super learner",
                                      g_SL_lib = g_lib,
                                      imp_covariates = cov_list,
                                      imp_SL_lib = imp_lib,
                                      pse_method = "Super learner",
                                      pse_covariates = cov_list,
                                      pse_SL_lib = pse_lib,
                                      newdata = sim_data_test)

      model_info_sim_list <- append(model_info_sim_list,list(EP10_imp = EP_learner_10_imp))
    },
    #if an error occurs, tell me the error
    error=function(e) {
      message(paste("An error occured when fitting the EP-learner - 10 CV folds (SL imputation) for scenario ",scen," with sample size ",n," in simulation ",seed,sep=""))
      print(e)
    }
  )


  #-----------------------------------------#
  #--- Running mDR learner (10 CV folds) ---#
  #-----------------------------------------#
  tryCatch(
    {
      out_lib <- c("SL.mean",
                   "SL.lm",
                   "SL.glmnet_8", "SL.glmnet_9",
                   "SL.glmnet_11", "SL.glmnet_12",
                   "SL.ranger_1","SL.ranger_2","SL.ranger_3",
                   "SL.ranger_4","SL.ranger_5","SL.ranger_6",
                   "SL.nnet_1","SL.nnet_2","SL.nnet_3",
                   "SL.svm_1",
                   "SL.kernelKnn_4","SL.kernelKnn_10")
      pse_lib <- out_lib
      e_lib <- c("SL.mean",
                 "SL.glm",
                 "SL.glmnet_8", "SL.glmnet_9",
                 "SL.glmnet_11", "SL.glmnet_12",
                 "SL.ranger_1","SL.ranger_2","SL.ranger_3",
                 "SL.ranger_4","SL.ranger_5","SL.ranger_6",
                 "SL.nnet_1","SL.nnet_2","SL.nnet_3",
                 "SL.svm_1",
                 "SL.kernelKnn_4","SL.kernelKnn_10")

      g_lib <- c("SL.mean",
                 "SL.glm",
                 "SL.glmnet_8", "SL.glmnet_9",
                 "SL.glmnet_11", "SL.glmnet_12",
                 "SL.ranger_1","SL.ranger_2","SL.ranger_3",
                 "SL.ranger_4","SL.ranger_5","SL.ranger_6",
                 "SL.nnet_1","SL.nnet_2","SL.nnet_3",
                 "SL.svm_1",
                 "SL.kernelKnn_4","SL.kernelKnn_5","SL.kernelKnn_6",
                 "SL.kernelKnn_10","SL.kernelKnn_11","SL.kernelKnn_12")

      mDR_learner_10 <- DR_learner(analysis = "mDR-learner",
                                   data = sim_data_train,
                                   id = "ID",
                                   outcome = "Y",
                                   exposure = "A",
                                   outcome_observed_indicator = "G_obs",
                                   splits = 10,
                                   e_method = "Super learner",
                                   e_covariates = cov_list,
                                   e_SL_lib = e_lib,
                                   out_method = "Super learner",
                                   out_covariates = cov_list,
                                   out_SL_lib = out_lib,
                                   g_method = "Super learner",
                                   g_covariates = cov_list,
                                   g_SL_lib = g_lib,
                                   pse_method = "Super learner",
                                   pse_covariates = cov_list,
                                   pse_SL_lib = pse_lib,
                                   newdata = sim_data_test)

      model_info_sim_list <- append(model_info_sim_list,list(mDR_10 = mDR_learner_10))
    },
    #if an error occurs, tell me the error
    error=function(e) {
      message(paste("An error occured when fitting the mDR-learner (10 CV folds) for scenario ",scen," with sample size ",n," in simulation ",seed,sep=""))
      print(e)
    }
  )



  #-----------------------------------------#
  #--- Running mEP learner (10 CV folds) ---#
  #-----------------------------------------#
  tryCatch(
    {
      out_lib <- c("SL.mean",
                   "SL.lm",
                   "SL.glmnet_8", "SL.glmnet_9",
                   "SL.glmnet_11", "SL.glmnet_12",
                   "SL.ranger_1","SL.ranger_2","SL.ranger_3",
                   "SL.ranger_4","SL.ranger_5","SL.ranger_6",
                   "SL.nnet_1","SL.nnet_2","SL.nnet_3",
                   "SL.svm_1",
                   "SL.kernelKnn_4","SL.kernelKnn_10")
      pse_lib <- out_lib
      e_lib <- c("SL.mean",
                 "SL.glm",
                 "SL.glmnet_8", "SL.glmnet_9",
                 "SL.glmnet_11", "SL.glmnet_12",
                 "SL.ranger_1","SL.ranger_2","SL.ranger_3",
                 "SL.ranger_4","SL.ranger_5","SL.ranger_6",
                 "SL.nnet_1","SL.nnet_2","SL.nnet_3",
                 "SL.svm_1",
                 "SL.kernelKnn_4","SL.kernelKnn_10")

      g_lib <- c("SL.mean",
                 "SL.glm",
                 "SL.glmnet_8", "SL.glmnet_9",
                 "SL.glmnet_11", "SL.glmnet_12",
                 "SL.ranger_1","SL.ranger_2","SL.ranger_3",
                 "SL.ranger_4","SL.ranger_5","SL.ranger_6",
                 "SL.nnet_1","SL.nnet_2","SL.nnet_3",
                 "SL.svm_1",
                 "SL.kernelKnn_4","SL.kernelKnn_5","SL.kernelKnn_6",
                 "SL.kernelKnn_10","SL.kernelKnn_11","SL.kernelKnn_12")

      mEP_learner_10 <- EP_learner(analysis = "mEP-learner",
                                   data = sim_data_train,
                                          id = "ID",
                                          outcome = "Y",
                                          exposure = "A",
                                          outcome_observed_indicator = "G_obs",
                                          splits = 10,
                                          e_method = "Super learner",
                                          e_covariates = cov_list,
                                          e_SL_lib = e_lib,
                                          out_method = "Super learner",
                                          out_covariates = cov_list,
                                          out_SL_lib = out_lib,
                                          g_method = "Super learner",
                                          g_covariates = cov_list,
                                          g_SL_lib = g_lib,
                                          pse_method = "Super learner",
                                          pse_covariates = cov_list,
                                          pse_SL_lib = pse_lib,
                                          newdata = sim_data_test)

      model_info_sim_list <- append(model_info_sim_list,list(mEP_10 = mEP_learner_10))
    },
    #if an error occurs, tell me the error
    error=function(e) {
      message(paste("An error occured when fitting the mEP-learner (10 CV folds) for scenario ",scen," with sample size ",n," in simulation ",seed,sep=""))
      print(e)
    }
  )

  model_info_list <- append(model_info_list,list(i = model_info_sim_list))
}


end <- Sys.time()
end-start


##################


# Save output files
store_name <- paste("scenario_",setting,"_output",taskID,sep = "")
print(store_name)

# Save output
save(model_info_list, file= paste0(store_name,".RData") )



