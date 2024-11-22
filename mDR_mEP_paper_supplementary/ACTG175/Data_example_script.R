
######################################################################
# Script: Data example script for models with cross-fitting
# Author: Matt Pryce
# Date: 29/03/23
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


########################
###   Loading data   ###
########################

#Original data
load("ACTG175_data.RData")


#############################
###   Loading functions   ###
#############################

source("Data_management_1tp.R")
source("nuisance_models.R")

source("EP_learner.R")
source("mDR_learner.R")
source("DR_learner.R")
source("T_learner.R")
source("Validation_algorithm.R")

# load("ACTG175_sieve_basis_list.RData")

##########################
###   Running models   ###
##########################

#Notes:
#  -  Each model within each learner will be run using the superlearner.


##################################################################################################

                       #---------------------------#
                       #---   Data management   ---#
                       #---------------------------#

#Data management 
#  - Do we need to normlaise variables?
#  - Are things coded properly?

#--- Formatting variables ---#
ACTG175_data$treat <- as.numeric(ACTG175_data$treat)
ACTG175_data$r <- as.numeric(ACTG175_data$r)
ACTG175_data$cd496 <- as.numeric(ACTG175_data$cd496)

#--- Defining covariates to be input into each model ---#
out_cov_list <- c("age","wtkg","karnof","cd40","cd80","gender","hemo","homo","symptom","race","drugs","str2")
imp_cov_list <- c("age","wtkg","karnof","cd40","cd80","gender","hemo","homo","symptom","race","drugs","str2")
ps_cov_list <- c("age","wtkg","karnof","cd40","cd80","gender","hemo","homo","symptom","race","drugs","str2")
pse1_cov_list <- c("age")
pse2_cov_list <- c("age","wtkg","karnof","cd40","cd80","gender","hemo","homo","symptom","race","drugs","str2")



##############################################################################################################################

                              #---------------------------------#
                              #---   Defining SL libraries   ---#
                              #---------------------------------#

#--- Creating learners for SL library's ---#
#LASSO & elastic net
nlambda_seq = c(50,100,250)
alpha_seq <- c(0.5,1)
usemin_seq <- c(FALSE,TRUE)
para_learners = create.Learner("SL.glmnet", tune = list(nlambda = nlambda_seq,alpha = alpha_seq,useMin = usemin_seq))
para_learners

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

#Boosting
depth_seq <- c(2,4,8)
shrink_seq <- c(0.05,0.1,0.3)
minobs_seq <- c(10,20)
boost_learners = create.Learner("SL.xgboost", tune = list(minobspernode=minobs_seq, max_depth = depth_seq, shrinkage = shrink_seq))
boost_learners


#--- Creating SL libraries ---#
#Outcome models - Reduced
out_lib <- c("SL.mean",
             "SL.lm",
             "SL.glmnet_8", "SL.glmnet_9",
             "SL.glmnet_11", "SL.glmnet_12",
             "SL.ranger_1","SL.ranger_2","SL.ranger_3",
             "SL.ranger_4","SL.ranger_5","SL.ranger_6",
             "SL.nnet_1","SL.nnet_2","SL.nnet_3",
             "SL.svm_1",
             "SL.kernelKnn_4",
             "SL.kernelKnn_10")


#Imputation models - Reduced
imp_lib <- c("SL.mean",
             "SL.glm",
             "SL.glmnet_8", "SL.glmnet_9",
             "SL.glmnet_11", "SL.glmnet_12",
             "SL.ranger_1","SL.ranger_2","SL.ranger_3",
             "SL.ranger_4","SL.ranger_5","SL.ranger_6",
             "SL.nnet_1","SL.nnet_2","SL.nnet_3",
             "SL.svm_1",
             "SL.kernelKnn_4",
             "SL.kernelKnn_10")


#Propensity score models reduced
e_lib <- c("SL.mean",
           "SL.glm",
           "SL.glmnet_8", "SL.glmnet_9",
           "SL.glmnet_11", "SL.glmnet_12",
           "SL.ranger_1","SL.ranger_2","SL.ranger_3",
           "SL.ranger_4","SL.ranger_5","SL.ranger_6",
           "SL.nnet_1","SL.nnet_2","SL.nnet_3",
           "SL.svm_1",
           "SL.kernelKnn_4","SL.kernelKnn_10")


#Pseudo outcome model - Single covariate - Reduced 
pse_lib <- c("SL.mean",
              "SL.lm",
              "SL.ranger_1","SL.ranger_3","SL.ranger_5",
              "SL.nnet_1","SL.nnet_2","SL.nnet_3",
              "SL.svm_1",
              "SL.kernelKnn_4",
              "SL.kernelKnn_10")


##################################################################################################################


#--------------#
#--- Set up ---#
#--------------#
model_info_list <- list()

taskID <- Sys.getenv("SLURM_ARRAY_TASK_ID")
taskID <- as.numeric(taskID)
set.seed(3854+taskID)

start <- Sys.time()

#---------------------------------------------------------------------#
#--- Running mEP learner - 10 CV folds - Single pseudo outcome set ---#
#---------------------------------------------------------------------#
tryCatch(
  {
    #--- Running model ---#
    mEP_10_pse1 <- EP_learner(analysis = "mEP-learner",
                              data = ACTG175_data,
                              id = "pidnum",
                              outcome = "cd496",
                              exposure = "treat",
                              outcome_observed_indicator = "r",
                              splits = 10,
                              e_method = "Super learner",
                              e_covariates = ps_cov_list,
                              e_SL_lib = e_lib,
                              out_method = "Super learner",
                              out_covariates = out_cov_list,
                              out_SL_lib = out_lib,
                              g_method = "Super learner",
                              g_covariates = imp_cov_list,
                              g_SL_lib = imp_lib,
                              pse_method = "Super learner",
                              pse_covariates = pse1_cov_list,
                              pse_SL_lib = pse_lib,
                              newdata = ACTG175_data)

    model_info_list <- append(model_info_list,list(mEP_pse1 = mEP_10_pse1$CATE_est))

    #--- Collecting CATE estimate ---#
    ACTG175_data <- cbind(ACTG175_data,mEP_pse1_CATE_est=as.vector(mEP_10_pse1$CATE_est))
    names(ACTG175_data)[length(ACTG175_data)] <- "mEP_pse1_CATE_est"

    #--- Running validation ---#
    debiased_MSE_mEP_pse1 <- debiased_MSE(data = ACTG175_data,
                                          id = "pidnum",
                                          outcome = "cd496",
                                          exposure = "treat",
                                          outcome_observed_indicator = "r",
                                          CATE_est = "mEP_pse1_CATE_est",
                                          splits = 10,
                                          e_method = "Super learner",
                                          e_covariates = ps_cov_list,
                                          e_SL_lib = e_lib,
                                          out_method = "Super learner",
                                          out_covariates = out_cov_list,
                                          out_SL_lib = out_lib,
                                          g_method = "Super learner",
                                          g_covariates = imp_cov_list,
                                          g_SL_lib = imp_lib)

    #Storing validation output
    if (length(debiased_MSE_mEP_pse1) == 3){
      model_info_list <- append(model_info_list,list(mEP_MSE_pse1 = debiased_MSE_mEP_pse1))
    }


  },
  #if an error occurs, tell me the error
  error=function(e) {
    message("An error occured when fitting the mEP-learner - 10 CV folds - Single pseudo outcome set")
    print(e)
  }
)


#------------------------------------------------------------------#
#--- Running mEP learner - 10 CV folds - All pseudo outcome set ---#
#------------------------------------------------------------------#
tryCatch(
  {
    #--- Running model ---#
    mEP_10_pse2 <- EP_learner(analysis = "mEP-learner",
                              data = ACTG175_data,
                              id = "pidnum",
                              outcome = "cd496",
                              exposure = "treat",
                              outcome_observed_indicator = "r",
                              splits = 10,
                              e_method = "Super learner",
                              e_covariates = ps_cov_list,
                              e_SL_lib = e_lib,
                              out_method = "Super learner",
                              out_covariates = out_cov_list,
                              out_SL_lib = out_lib,
                              g_method = "Super learner",
                              g_covariates = imp_cov_list,
                              g_SL_lib = imp_lib,
                              pse_method = "Super learner",
                              pse_covariates = pse2_cov_list,
                              pse_SL_lib = pse_lib,
                              newdata = ACTG175_data)

    model_info_list <- append(model_info_list,list(mEP_pse2 = mEP_10_pse2$CATE_est))

    #--- Collecting CATE estimate ---#
    ACTG175_data <- cbind(ACTG175_data,mEP_pse2_CATE_est=as.vector(mEP_10_pse2$CATE_est))
    names(ACTG175_data)[length(ACTG175_data)] <- "mEP_pse2_CATE_est"

    #--- Running validation ---#
    debiased_MSE_mEP_pse2 <- debiased_MSE(data = ACTG175_data,
                                          id = "pidnum",
                                          outcome = "cd496",
                                          exposure = "treat",
                                          outcome_observed_indicator = "r",
                                          CATE_est = "mEP_pse2_CATE_est",
                                          splits = 10,
                                          e_method = "Super learner",
                                          e_covariates = ps_cov_list,
                                          e_SL_lib = e_lib,
                                          out_method = "Super learner",
                                          out_covariates = out_cov_list,
                                          out_SL_lib = out_lib,
                                          g_method = "Super learner",
                                          g_covariates = imp_cov_list,
                                          g_SL_lib = imp_lib)

    #Storing validation output
    if (length(debiased_MSE_mEP_pse2) == 3){
      model_info_list <- append(model_info_list,list(mEP_MSE_pse2 = debiased_MSE_mEP_pse2))
    }


  },
  #if an error occurs, tell me the error
  error=function(e) {
    message("An error occured when fitting the mEP-learner - 10 CV folds - All pseudo outcome set")
    print(e)
  }
)



#-------------------------------------------------------------------------------------#
#--- Running EP learner (Available case) - 10 CV folds - Single pseudo outcome set ---#
#-------------------------------------------------------------------------------------#
tryCatch(
  {
    #--- Running model ---#
    EP_AC_10_pse1 <- EP_learner(analysis = "Available case",
                                data = ACTG175_data,
                                id = "pidnum",
                                outcome = "cd496",
                                exposure = "treat",
                                outcome_observed_indicator = "r",
                                splits = 10,
                                e_method = "Super learner",
                                e_covariates = ps_cov_list,
                                e_SL_lib = e_lib,
                                out_method = "Super learner",
                                out_covariates = out_cov_list,
                                out_SL_lib = out_lib,
                                g_method = "Super learner",
                                g_covariates = imp_cov_list,
                                g_SL_lib = imp_lib,
                                pse_method = "Super learner",
                                pse_covariates = pse1_cov_list,
                                pse_SL_lib = pse_lib,
                                newdata = ACTG175_data)

    model_info_list <- append(model_info_list,list(EP_AC_pse1 = EP_AC_10_pse1$CATE_est))

    #--- Collecting CATE estimate ---#
    ACTG175_data <- cbind(ACTG175_data,EP_AC_pse1_CATE_est=as.vector(EP_AC_10_pse1$CATE_est))
    names(ACTG175_data)[length(ACTG175_data)] <- "EP_AC_pse1_CATE_est"

    #--- Running validation ---#
    debiased_MSE_EP_AC_pse1 <- debiased_MSE(data = ACTG175_data,
                                            id = "pidnum",
                                            outcome = "cd496",
                                            exposure = "treat",
                                            outcome_observed_indicator = "r",
                                            CATE_est = "EP_AC_pse1_CATE_est",
                                            splits = 10,
                                            e_method = "Super learner",
                                            e_covariates = ps_cov_list,
                                            e_SL_lib = e_lib,
                                            out_method = "Super learner",
                                            out_covariates = out_cov_list,
                                            out_SL_lib = out_lib,
                                            g_method = "Super learner",
                                            g_covariates = imp_cov_list,
                                            g_SL_lib = imp_lib)

    #Storing validation output
    if (length(debiased_MSE_EP_AC_pse1) == 3){
      model_info_list <- append(model_info_list,list(EP_MSE_pse1 = debiased_MSE_EP_AC_pse1))
    }


  },
  #if an error occurs, tell me the error
  error=function(e) {
    message("An error occured when fitting the EP-learner AC - 10 CV folds - Single pseudo outcome set")
    print(e)
  }
)


#----------------------------------------------------------------------------------#
#--- Running EP learner (Available case) - 10 CV folds - All pseudo outcome set ---#
#----------------------------------------------------------------------------------#
tryCatch(
  {
    #--- Running model ---#
    EP_AC_10_pse2 <- EP_learner(analysis = "Available case",
                                data = ACTG175_data,
                                id = "pidnum",
                                outcome = "cd496",
                                exposure = "treat",
                                outcome_observed_indicator = "r",
                                splits = 10,
                                e_method = "Super learner",
                                e_covariates = ps_cov_list,
                                e_SL_lib = e_lib,
                                out_method = "Super learner",
                                out_covariates = out_cov_list,
                                out_SL_lib = out_lib,
                                g_method = "Super learner",
                                g_covariates = imp_cov_list,
                                g_SL_lib = imp_lib,
                                pse_method = "Super learner",
                                pse_covariates = pse2_cov_list,
                                pse_SL_lib = pse_lib,
                                newdata = ACTG175_data)

    model_info_list <- append(model_info_list,list(EP_AC_pse2 = EP_AC_10_pse2$CATE_est))

    #--- Collecting CATE estimate ---#
    ACTG175_data <- cbind(ACTG175_data,EP_AC_pse2_CATE_est=as.vector(EP_AC_10_pse2$CATE_est))
    names(ACTG175_data)[length(ACTG175_data)] <- "EP_AC_pse2_CATE_est"

    #--- Running validation ---#
    debiased_MSE_EP_AC_pse2 <- debiased_MSE(data = ACTG175_data,
                                          id = "pidnum",
                                          outcome = "cd496",
                                          exposure = "treat",
                                          outcome_observed_indicator = "r",
                                          CATE_est = "EP_AC_pse2_CATE_est",
                                          splits = 10,
                                          e_method = "Super learner",
                                          e_covariates = ps_cov_list,
                                          e_SL_lib = e_lib,
                                          out_method = "Super learner",
                                          out_covariates = out_cov_list,
                                          out_SL_lib = out_lib,
                                          g_method = "Super learner",
                                          g_covariates = imp_cov_list,
                                          g_SL_lib = imp_lib)

    #Storing validation output
    if (length(debiased_MSE_EP_AC_pse2) == 3){
      model_info_list <- append(model_info_list,list(EP_MSE_pse2 = debiased_MSE_EP_AC_pse2))
    }


  },
  #if an error occurs, tell me the error
  error=function(e) {
    message("An error occured when fitting the EP-learner AC - 10 CV folds - All pseudo outcome set")
    print(e)
  }
)




#---------------------------------------------------------------------#
#--- Running mDR learner - 10 CV folds - Single pseudo outcome set ---#
#---------------------------------------------------------------------#
tryCatch(
  {
    #--- Running model ---#
    mDR_10_pse1 <- mDR_learner(data = ACTG175_data,
                               id = "pidnum",
                               outcome = "cd496",
                               exposure = "treat",
                               outcome_observed_indicator = "r",
                               splits = 10,
                               e_method = "Super learner",
                               e_covariates = ps_cov_list,
                               e_SL_lib = e_lib,
                               out_method = "Super learner",
                               out_covariates = out_cov_list,
                               out_SL_lib = out_lib,
                               g_method = "Super learner",
                               g_covariates = imp_cov_list,
                               g_SL_lib = imp_lib,
                               pse_method = "Super learner",
                               pse_covariates = pse1_cov_list,
                               pse_SL_lib = pse_lib,
                               newdata = ACTG175_data)

    model_info_list <- append(model_info_list,list(mDR_pse1 = mDR_10_pse1$CATE_est))

    #--- Collecting CATE estimate ---#
    ACTG175_data <- cbind(ACTG175_data,mDR_pse1_CATE_est=as.vector(mDR_10_pse1$CATE_est))
    names(ACTG175_data)[length(ACTG175_data)] <- "mDR_pse1_CATE_est"

    #--- Running validation ---#
    debiased_MSE_mDR_pse1 <- debiased_MSE(data = ACTG175_data,
                                          id = "pidnum",
                                          outcome = "cd496",
                                          exposure = "treat",
                                          outcome_observed_indicator = "r",
                                          CATE_est = "mDR_pse1_CATE_est",
                                          splits = 10,
                                          e_method = "Super learner",
                                          e_covariates = ps_cov_list,
                                          e_SL_lib = e_lib,
                                          out_method = "Super learner",
                                          out_covariates = out_cov_list,
                                          out_SL_lib = out_lib,
                                          g_method = "Super learner",
                                          g_covariates = imp_cov_list,
                                          g_SL_lib = imp_lib)

    #Storing validation output
    if (length(debiased_MSE_mDR_pse1) == 3){
      model_info_list <- append(model_info_list,list(mDR_MSE_pse1 = debiased_MSE_mDR_pse1))
    }


  },
  #if an error occurs, tell me the error
  error=function(e) {
    message("An error occured when fitting the extended DR-learner - 10 CV folds - Single pseudo outcome set")
    print(e)
  }
)

#------------------------------------------------------------------#
#--- Running mDR learner - 10 CV folds - All pseudo outcome set ---#
#------------------------------------------------------------------#
tryCatch(
  {
    #--- Running model ---#
    mDR_10_pse2 <- mDR_learner(data = ACTG175_data,
                               id = "pidnum",
                               outcome = "cd496",
                               exposure = "treat",
                               outcome_observed_indicator = "r",
                               splits = 10,
                               e_method = "Super learner",
                               e_covariates = ps_cov_list,
                               e_SL_lib = e_lib,
                               out_method = "Super learner",
                               out_covariates = out_cov_list,
                               out_SL_lib = out_lib,
                               g_method = "Super learner",
                               g_covariates = imp_cov_list,
                               g_SL_lib = imp_lib,
                               pse_method = "Super learner",
                               pse_covariates = pse2_cov_list,
                               pse_SL_lib = pse_lib,
                               newdata = ACTG175_data)

    model_info_list <- append(model_info_list,list(mDR_pse2 = mDR_10_pse2$CATE_est))

    #--- Collecting CATE estimate ---#
    ACTG175_data <- cbind(ACTG175_data,mDR_pse2_CATE_est=as.vector(mDR_10_pse2$CATE_est))
    names(ACTG175_data)[length(ACTG175_data)] <- "mDR_pse2_CATE_est"

    #--- Running validation ---#
    debiased_MSE_mDR_pse2 <- debiased_MSE(data = ACTG175_data,
                                          id = "pidnum",
                                          outcome = "cd496",
                                          exposure = "treat",
                                          outcome_observed_indicator = "r",
                                          CATE_est = "mDR_pse2_CATE_est",
                                          splits = 10,
                                          e_method = "Super learner",
                                          e_covariates = ps_cov_list,
                                          e_SL_lib = e_lib,
                                          out_method = "Super learner",
                                          out_covariates = out_cov_list,
                                          out_SL_lib = out_lib,
                                          g_method = "Super learner",
                                          g_covariates = imp_cov_list,
                                          g_SL_lib = imp_lib)

    #Storing validation output
    if (length(debiased_MSE_mDR_pse2) == 3){
      model_info_list <- append(model_info_list,list(mDR_MSE_pse2 = debiased_MSE_mDR_pse2))
    }


  },
  #if an error occurs, tell me the error
  error=function(e) {
    message("An error occured when fitting the extended DR-learner - 10 CV folds - Single pseudo outcome set")
    print(e)
  }
)



#--------------------------------------------------------------------#
#--- Running DR learner - 10 CV folds - Single pseudo outcome set ---#
#--------------------------------------------------------------------#
tryCatch(
  {
    #--- Running model ---#
    DR_10_pse1 <- DR_learner(analysis = "Available case",
                             data = ACTG175_data,
                             id = "pidnum",
                             outcome = "cd496",
                             exposure = "treat",
                             outcome_observed_indicator = "r",
                             splits = 10,
                             e_method = "Super learner",
                             e_covariates = ps_cov_list,
                             e_SL_lib = e_lib,
                             out_method = "Super learner",
                             out_covariates = out_cov_list,
                             out_SL_lib = out_lib,
                             pse_method = "Super learner",
                             pse_covariates = pse1_cov_list,
                             pse_SL_lib = pse_lib,
                             newdata = ACTG175_data)
    
    model_info_list <- append(model_info_list,list(DR_pse1 = DR_10_pse1$CATE_est))
    
    #--- Collecting CATE estimate ---#
    ACTG175_data <- cbind(ACTG175_data,DR_pse1_CATE_est=as.vector(DR_10_pse1$CATE_est))
    names(ACTG175_data)[length(ACTG175_data)] <- "DR_pse1_CATE_est"
    
    #--- Running validation ---#
    debiased_MSE_DR_pse1 <- debiased_MSE(data = ACTG175_data,
                                         id = "pidnum",
                                         outcome = "cd496",
                                         exposure = "treat",
                                         outcome_observed_indicator = "r",
                                         CATE_est = "DR_pse1_CATE_est",
                                         splits = 10,
                                         e_method = "Super learner",
                                         e_covariates = ps_cov_list,
                                         e_SL_lib = e_lib,
                                         out_method = "Super learner",
                                         out_covariates = out_cov_list,
                                         out_SL_lib = out_lib,
                                         g_method = "Super learner",
                                         g_covariates = imp_cov_list,
                                         g_SL_lib = imp_lib)
    
    #Storing validation output
    if (length(debiased_MSE_DR_pse1) == 3){
      model_info_list <- append(model_info_list,list(DR_MSE_pse1 = debiased_MSE_DR_pse1))
    }
    
    
  },
  #if an error occurs, tell me the error
  error=function(e) {
    message("An error occured when fitting the DR-learner - 10 CV folds - Single pseudo outcome set")
    print(e)
  }
)

#-----------------------------------------------------------------#
#--- Running DR learner - 10 CV folds - All pseudo outcome set ---#
#-----------------------------------------------------------------#
tryCatch(
  {
    #--- Running model ---#
    DR_10_pse2 <- DR_learner(analysis = "Available case",
                             data = ACTG175_data,
                             id = "pidnum",
                             outcome = "cd496",
                             exposure = "treat",
                             outcome_observed_indicator = "r",
                             splits = 10,
                             e_method = "Super learner",
                             e_covariates = ps_cov_list,
                             e_SL_lib = e_lib,
                             out_method = "Super learner",
                             out_covariates = out_cov_list,
                             out_SL_lib = out_lib,
                             pse_method = "Super learner",
                             pse_covariates = pse2_cov_list,
                             pse_SL_lib = pse_lib,
                             newdata = ACTG175_data)
    
    model_info_list <- append(model_info_list,list(DR_pse2 = DR_10_pse2$CATE_est))
    
    #--- Collecting CATE estimate ---#
    ACTG175_data <- cbind(ACTG175_data,DR_pse2_CATE_est=as.vector(DR_10_pse2$CATE_est))
    names(ACTG175_data)[length(ACTG175_data)] <- "DR_pse2_CATE_est"
    
    #--- Running validation ---#
    debiased_MSE_DR_pse2 <- debiased_MSE(data = ACTG175_data,
                                          id = "pidnum",
                                          outcome = "cd496",
                                          exposure = "treat",
                                          outcome_observed_indicator = "r",
                                          CATE_est = "DR_pse2_CATE_est",
                                          splits = 10,
                                          e_method = "Super learner",
                                          e_covariates = ps_cov_list,
                                          e_SL_lib = e_lib,
                                          out_method = "Super learner",
                                          out_covariates = out_cov_list,
                                          out_SL_lib = out_lib,
                                          g_method = "Super learner",
                                          g_covariates = imp_cov_list,
                                          g_SL_lib = imp_lib)
    
    #Storing validation output
    if (length(debiased_MSE_DR_pse2) == 3){
      model_info_list <- append(model_info_list,list(DR_MSE_pse2 = debiased_MSE_DR_pse2))
    }
    
    
  },
  #if an error occurs, tell me the error
  error=function(e) {
    message("An error occured when fitting the extended DR-learner - 10 CV folds - Single pseudo outcome set")
    print(e)
  }
)


#--------------------------------------------------#
#--- Running T learner - All pseudo outcome set ---#
#--------------------------------------------------#
tryCatch(
  {
    #--- Running model ---#
    T_CC <- T_learner(analysis = "Complete case",
                      data = ACTG175_data,
                      id = "pidnum",
                      outcome = "cd496",
                      exposure = "treat",
                      outcome_observed_indicator = "r",
                      out_method = "Super learner",
                      out_covariates = out_cov_list,
                      out_SL_lib = out_lib,
                      newdata = ACTG175_data)


    model_info_list <- append(model_info_list,list(T_CC_pse2 = T_CC$CATE_est))

    #--- Collecting CATE estimate ---#
    ACTG175_data <- cbind(ACTG175_data,T_pse2_CATE_est=as.vector(T_CC$CATE_est))
    names(ACTG175_data)[length(ACTG175_data)] <- "T_pse2_CATE_est"

    #--- Running validation ---#
    debiased_MSE_T_pse2 <- debiased_MSE(data = ACTG175_data,
                                         id = "pidnum",
                                         outcome = "cd496",
                                         exposure = "treat",
                                         outcome_observed_indicator = "r",
                                         CATE_est = "T_pse2_CATE_est",
                                         splits = 10,
                                         e_method = "Super learner",
                                         e_covariates = ps_cov_list,
                                         e_SL_lib = e_lib,
                                         out_method = "Super learner",
                                         out_covariates = out_cov_list,
                                         out_SL_lib = out_lib,
                                         g_method = "Super learner",
                                         g_covariates = imp_cov_list,
                                         g_SL_lib = imp_lib)
    
    #Storing validation output
    if (length(debiased_MSE_T_pse2) == 3){
      model_info_list <- append(model_info_list,list(T_MSE_pse2 = debiased_MSE_T_pse2))
    }


  },
  #if an error occurs, tell me the error
  error=function(e) {
    message("An error occured when fitting the T-learner")
    print(e)
  }
)


end <- Sys.time()
end-start


##################


# Save output files
store_name <- paste("Data_example_output",taskID,sep = "_")
print(store_name)

# Save output
save(model_info_list, file= paste0(store_name,".RData") )



