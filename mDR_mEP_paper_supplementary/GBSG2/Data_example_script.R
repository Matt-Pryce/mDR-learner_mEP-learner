
######################################################################
# Script: Data example script for models with cross-fitting
# Author: Matt Pryce
# Date: 11/03/25
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
load("GBSG2.RData")


#############################
###   Loading functions   ###
#############################

source("Data_management_1tp.R")
source("nuisance_models.R")

source("EP_learner.R")
source("DR_learner.R")
source("T_learner.R")
source("IPTW_IPCW_learner.R")


##########################
###   Running models   ###
##########################

#Notes:
#  -  Each model within each learner will be run using the superlearner.


##################################################################################################

#---------------------------#
#---   Data management   ---#
#---------------------------#

#--- Formatting variables ---#
GBSG2$horTh <- as.numeric(GBSG2$horTh)
GBSG2$r <- as.numeric(GBSG2$r)
GBSG2$event_3yr <- as.numeric(GBSG2$event_3yr)
GBSG2$ID2 <- c(1:686)
GBSG2$age <- as.numeric(GBSG2$age)
GBSG2$tsize <- as.numeric(GBSG2$tsize)
GBSG2$pnodes <- as.numeric(GBSG2$pnodes)
GBSG2$progrec <- as.numeric(GBSG2$progrec)
GBSG2$estrec <- as.numeric(GBSG2$estrec)
GBSG2 <- GBSG2 %>% mutate(menostat = if_else(menostat == "Pre", 0, 1))
GBSG2 <- GBSG2 %>% mutate(tgrade = recode(tgrade, "I" = 1, "II" = 2, "III" = 3))

#--- Defining covariates to be input into each model ---#
out_cov_list <- c("age","menostat","tsize","tgrade","pnodes","progrec","estrec")
imp_cov_list <- c("age","menostat","tsize","tgrade","pnodes","progrec","estrec")
ps_cov_list <- c("age","menostat","tsize","tgrade","pnodes","progrec","estrec")
pse1_cov_list <- c("progrec")
pse2_cov_list <- c("age","menostat","tsize","tgrade","pnodes","progrec","estrec")



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
             "SL.glm",
             "SL.glmnet_8", "SL.glmnet_9",
             "SL.glmnet_11", "SL.glmnet_12",
             "SL.ranger_1","SL.ranger_2","SL.ranger_3",
             "SL.ranger_4","SL.ranger_5","SL.ranger_6")

#Imputation models - Reduced
imp_lib <- c("SL.mean",
             "SL.glm",
             "SL.glmnet_8", "SL.glmnet_9",
             "SL.glmnet_11", "SL.glmnet_12",
             "SL.ranger_1","SL.ranger_2","SL.ranger_3",
             "SL.ranger_4","SL.ranger_5","SL.ranger_6")


#Propensity score models reduced
e_lib <- c("SL.mean",
           "SL.glm",
           "SL.glmnet_8", "SL.glmnet_9",
           "SL.glmnet_11", "SL.glmnet_12",
           "SL.ranger_1","SL.ranger_2","SL.ranger_3",
           "SL.ranger_4","SL.ranger_5","SL.ranger_6")


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
                              data = GBSG2,
                              id = "ID2",
                              outcome = "event_3yr",
                              exposure = "trt",
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
                              newdata = GBSG2)
    
    model_info_list <- append(model_info_list,list(mEP_pse1_est = mEP_10_pse1$CATE_est))
    
    #--- Collecting CATE estimate ---#
    GBSG2 <- cbind(GBSG2,mEP_pse1_CATE_est=as.vector(mEP_10_pse1$CATE_est))
    names(GBSG2)[length(GBSG2)] <- "mEP_pse1_CATE_est"
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
                              data = GBSG2,
                              id = "ID2",
                              outcome = "event_3yr",
                              exposure = "trt",
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
                              newdata = GBSG2)
    
    model_info_list <- append(model_info_list,list(mEP_pse2_est = mEP_10_pse2$CATE_est))
    
    #--- Collecting CATE estimate ---#
    GBSG2 <- cbind(GBSG2,mEP_pse2_CATE_est=as.vector(mEP_10_pse2$CATE_est))
    names(GBSG2)[length(GBSG2)] <- "mEP_pse2_CATE_est"
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
                                data = GBSG2,
                                id = "ID2",
                                outcome = "event_3yr",
                                exposure = "trt",
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
                                newdata = GBSG2)
    
    model_info_list <- append(model_info_list,list(EP_AC_pse1_est = EP_AC_10_pse1$CATE_est))
    
    #--- Collecting CATE estimate ---#
    GBSG2 <- cbind(GBSG2,EP_AC_pse1_CATE_est=as.vector(EP_AC_10_pse1$CATE_est))
    names(GBSG2)[length(GBSG2)] <- "EP_AC_pse1_CATE_est"
    
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
                                data = GBSG2,
                                id = "ID2",
                                outcome = "event_3yr",
                                exposure = "trt",
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
                                newdata = GBSG2)
    
    model_info_list <- append(model_info_list,list(EP_AC_pse2_est = EP_AC_10_pse2$CATE_est))
    
    #--- Collecting CATE estimate ---#
    GBSG2 <- cbind(GBSG2,EP_AC_pse2_CATE_est=as.vector(EP_AC_10_pse2$CATE_est))
    names(GBSG2)[length(GBSG2)] <- "EP_AC_pse2_CATE_est"
  },
  #if an error occurs, tell me the error
  error=function(e) {
    message("An error occured when fitting the EP-learner AC - 10 CV folds - All pseudo outcome set")
    print(e)
  }
)



#------------------------------------------------------------------------------------#
#--- Running EP learner (SL imputation) - 10 CV folds - Single pseudo outcome set ---#
#------------------------------------------------------------------------------------#
tryCatch(
  {
    #--- Running model ---#
    EP_SL_10_pse1 <- EP_learner(analysis = "SL imputation",
                                data = GBSG2,
                                id = "ID2",
                                outcome = "event_3yr",
                                exposure = "trt",
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
                                imp_covariates = imp_cov_list,
                                imp_SL_lib = imp_lib,
                                pse_method = "Super learner",
                                pse_covariates = pse1_cov_list,
                                pse_SL_lib = pse_lib,
                                newdata = GBSG2)
    
    model_info_list <- append(model_info_list,list(EP_SL_pse1_est = EP_SL_10_pse1$CATE_est))
    
    #--- Collecting CATE estimate ---#
    GBSG2 <- cbind(GBSG2,EP_SL_pse1_CATE_est=as.vector(EP_SL_10_pse1$CATE_est))
    names(GBSG2)[length(GBSG2)] <- "EP_SL_pse1_CATE_est"
    
  },
  #if an error occurs, tell me the error
  error=function(e) {
    message("An error occured when fitting the EP-learner SL - 10 CV folds - Single pseudo outcome set")
    print(e)
  }
)


#----------------------------------------------------------------------------------#
#--- Running EP learner (SL imputation) - 10 CV folds - All pseudo outcome set ---#
#----------------------------------------------------------------------------------#
tryCatch(
  {
    #--- Running model ---#
    EP_SL_10_pse2 <- EP_learner(analysis = "SL imputation",
                                data = GBSG2,
                                id = "ID2",
                                outcome = "event_3yr",
                                exposure = "trt",
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
                                imp_covariates = imp_cov_list,
                                imp_SL_lib = imp_lib,
                                pse_method = "Super learner",
                                pse_covariates = pse2_cov_list,
                                pse_SL_lib = pse_lib,
                                newdata = GBSG2)
    
    model_info_list <- append(model_info_list,list(EP_SL_pse2_est = EP_SL_10_pse2$CATE_est))
    
    #--- Collecting CATE estimate ---#
    GBSG2 <- cbind(GBSG2,EP_SL_pse2_CATE_est=as.vector(EP_SL_10_pse2$CATE_est))
    names(GBSG2)[length(GBSG2)] <- "EP_SL_pse2_CATE_est"
  },
  #if an error occurs, tell me the error
  error=function(e) {
    message("An error occured when fitting the EP-learner SL - 10 CV folds - All pseudo outcome set")
    print(e)
  }
)


#---------------------------------------------------------------------#
#--- Running mDR learner - 10 CV folds - Single pseudo outcome set ---#
#---------------------------------------------------------------------#
tryCatch(
  {
    #--- Running model ---#
    mDR_10_pse1 <- DR_learner(analysis = "mDR-learner",
                              data = GBSG2,
                              id = "ID2",
                              outcome = "event_3yr",
                              exposure = "trt",
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
                              newdata = GBSG2)
    
    model_info_list <- append(model_info_list,list(mDR_pse1_est = mDR_10_pse1$CATE_est))
    
    #--- Collecting CATE estimate ---#
    GBSG2 <- cbind(GBSG2,mDR_pse1_CATE_est=as.vector(mDR_10_pse1$CATE_est))
    names(GBSG2)[length(GBSG2)] <- "mDR_pse1_CATE_est"
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
    mDR_10_pse2 <- DR_learner(analysis = "mDR-learner",
                              data = GBSG2,
                              id = "ID2",
                              outcome = "event_3yr",
                              exposure = "trt",
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
                              newdata = GBSG2)
    
    model_info_list <- append(model_info_list,list(mDR_pse2_est = mDR_10_pse2$CATE_est))
    
    #--- Collecting CATE estimate ---#
    GBSG2 <- cbind(GBSG2,mDR_pse2_CATE_est=as.vector(mDR_10_pse2$CATE_est))
    names(GBSG2)[length(GBSG2)] <- "mDR_pse2_CATE_est"
  },
  #if an error occurs, tell me the error
  error=function(e) {
    message("An error occured when fitting the extended DR-learner - 10 CV folds - Single pseudo outcome set")
    print(e)
  }
)



#-------------------------------------------------------------------------------------#
#--- Running DR learner (Available case) - 10 CV folds - Single pseudo outcome set ---#
#-------------------------------------------------------------------------------------#
tryCatch(
  {
    #--- Running model ---#
    DR_AC_10_pse1 <- DR_learner(analysis = "Available case",
                                data = GBSG2,
                                id = "ID2",
                                outcome = "event_3yr",
                                exposure = "trt",
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
                                newdata = GBSG2)
    
    model_info_list <- append(model_info_list,list(DR_AC_pse1_est = DR_AC_10_pse1$CATE_est))
    
    #--- Collecting CATE estimate ---#
    GBSG2 <- cbind(GBSG2,DR_AC_pse1_CATE_est=as.vector(DR_AC_10_pse1$CATE_est))
    names(GBSG2)[length(GBSG2)] <- "DR_AC_pse1_CATE_est"
  },
  #if an error occurs, tell me the error
  error=function(e) {
    message("An error occured when fitting the DR-learner - 10 CV folds - Single pseudo outcome set")
    print(e)
  }
)

#----------------------------------------------------------------------------------#
#--- Running DR learner (Available case) - 10 CV folds - All pseudo outcome set ---#
#----------------------------------------------------------------------------------#
tryCatch(
  {
    #--- Running model ---#
    DR_AC_10_pse2 <- DR_learner(analysis = "Available case",
                                data = GBSG2,
                                id = "ID2",
                                outcome = "event_3yr",
                                exposure = "trt",
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
                                newdata = GBSG2)
    
    model_info_list <- append(model_info_list,list(DR_AC_pse2_est = DR_AC_10_pse2$CATE_est))
    
    #--- Collecting CATE estimate ---#
    GBSG2 <- cbind(GBSG2,DR_AC_pse2_CATE_est=as.vector(DR_AC_10_pse2$CATE_est))
    names(GBSG2)[length(GBSG2)] <- "DR_AC_pse2_CATE_est"
  },
  #if an error occurs, tell me the error
  error=function(e) {
    message("An error occured when fitting the extended DR-learner - 10 CV folds - Single pseudo outcome set")
    print(e)
  }
)


#-----------------------------------------------------------------------------------------#
#--- Running DR learner (with SL imputation) - 10 CV folds - Single pseudo outcome set ---#
#-----------------------------------------------------------------------------------------#
tryCatch(
  {
    #--- Running model ---#
    DR_SL_10_pse1 <- DR_learner(analysis = "SL imputation",
                                data = GBSG2,
                                id = "ID2",
                                outcome = "event_3yr",
                                exposure = "trt",
                                outcome_observed_indicator = "r",
                                splits = 10,
                                e_method = "Super learner",
                                e_covariates = ps_cov_list,
                                e_SL_lib = e_lib,
                                out_method = "Super learner",
                                out_covariates = out_cov_list,
                                out_SL_lib = out_lib,
                                imp_covariates = imp_cov_list,
                                imp_SL_lib = imp_lib,
                                pse_method = "Super learner",
                                pse_covariates = pse1_cov_list,
                                pse_SL_lib = pse_lib,
                                newdata = GBSG2)
    
    model_info_list <- append(model_info_list,list(DR_SL_pse1_est = DR_SL_10_pse1$CATE_est))
    
    #--- Collecting CATE estimate ---#
    GBSG2 <- cbind(GBSG2,DR_SL_pse1_CATE_est=as.vector(DR_SL_10_pse1$CATE_est))
    names(GBSG2)[length(GBSG2)] <- "DR_SL_pse1_CATE_est"
    
  },
  #if an error occurs, tell me the error
  error=function(e) {
    message("An error occured when fitting the DR-learner SL - 10 CV folds - Single pseudo outcome set")
    print(e)
  }
)

#-----------------------------------------------------------------#
#--- Running DR learner - 10 CV folds - All pseudo outcome set ---#
#-----------------------------------------------------------------#
tryCatch(
  {
    #--- Running model ---#
    DR_SL_10_pse2 <- DR_learner(analysis = "SL imputation",
                                data = GBSG2,
                                id = "ID2",
                                outcome = "event_3yr",
                                exposure = "trt",
                                outcome_observed_indicator = "r",
                                splits = 10,
                                e_method = "Super learner",
                                e_covariates = ps_cov_list,
                                e_SL_lib = e_lib,
                                out_method = "Super learner",
                                out_covariates = out_cov_list,
                                out_SL_lib = out_lib,
                                imp_covariates = imp_cov_list,
                                imp_SL_lib = imp_lib,
                                pse_method = "Super learner",
                                pse_covariates = pse2_cov_list,
                                pse_SL_lib = pse_lib,
                                newdata = GBSG2)
    
    model_info_list <- append(model_info_list,list(DR_SL_pse2_est = DR_SL_10_pse2$CATE_est))
    
    #--- Collecting CATE estimate ---#
    GBSG2 <- cbind(GBSG2,DR_SL_pse2_CATE_est=as.vector(DR_SL_10_pse2$CATE_est))
    names(GBSG2)[length(GBSG2)] <- "DR_SL_pse2_CATE_est"
  },
  #if an error occurs, tell me the error
  error=function(e) {
    message("An error occured when fitting the extended DR-learner SL - 10 CV folds - All pseudo outcome set")
    print(e)
  }
)


#------------------------------------------------------------------#
#--- Running T learner (Complete case) - All pseudo outcome set ---#
#------------------------------------------------------------------#
tryCatch(
  {
    #--- Running model ---#
    T_CC <- T_learner(analysis = "Complete case",
                      data = GBSG2,
                      id = "ID2",
                      outcome = "event_3yr",
                      exposure = "trt",
                      outcome_observed_indicator = "r",
                      out_method = "Super learner",
                      out_covariates = out_cov_list,
                      out_SL_lib = out_lib,
                      newdata = GBSG2)
    
    
    model_info_list <- append(model_info_list,list(T_CC_pse2 = T_CC$CATE_est))
    
    #--- Collecting CATE estimate ---#
    GBSG2 <- cbind(GBSG2,T_CC_pse2_CATE_est=as.vector(T_CC$CATE_est))
    names(GBSG2)[length(GBSG2)] <- "T_CC_pse2_CATE_est"
    
  },
  #if an error occurs, tell me the error
  error=function(e) {
    message("An error occured when fitting the T-learner")
    print(e)
  }
)



#------------------------------------------------------------------#
#--- Running T learner (SL imputation) - All pseudo outcome set ---#
#------------------------------------------------------------------#
tryCatch(
  {
    #--- Running model ---#
    T_SL <- T_learner(analysis = "SL imputation",
                      data = GBSG2,
                      id = "ID2",
                      outcome = "event_3yr",
                      exposure = "trt",
                      outcome_observed_indicator = "r",
                      out_method = "Super learner",
                      out_covariates = out_cov_list,
                      out_SL_lib = out_lib,
                      imp_covariates = imp_cov_list,
                      imp_SL_lib = imp_lib,
                      newdata = GBSG2)
    
    
    model_info_list <- append(model_info_list,list(T_SL_pse2 = T_SL$CATE_est))
    
    #--- Collecting CATE estimate ---#
    GBSG2 <- cbind(GBSG2,T_SL_pse2_CATE_est=as.vector(T_SL$CATE_est))
    names(GBSG2)[length(GBSG2)] <- "T_SL_pse2_CATE_est"
    
  },
  #if an error occurs, tell me the error
  error=function(e) {
    message("An error occured when fitting the T-learner SL")
    print(e)
  }
)

#---------------------------------------------------------------------------#
#--- Running IPTW-IPCW learner - 10 CV folds - Single pseudo outcome set ---#
#---------------------------------------------------------------------------#
tryCatch(
  {
    #--- Running model ---#
    IPTW_IPCW_10_pse1 <- IPTW_IPCW_learner(data = GBSG2,
                                           id = "ID2",
                                           outcome = "event_3yr",
                                           exposure = "trt",
                                           outcome_observed_indicator = "r",
                                           splits = 10,
                                           e_method = "Super learner",
                                           e_covariates = ps_cov_list,
                                           e_SL_lib = e_lib,
                                           g_method = "Super learner",
                                           g_covariates = imp_cov_list,
                                           g_SL_lib = imp_lib,
                                           pse_method = "Super learner",
                                           pse_covariates = pse1_cov_list,
                                           pse_SL_lib = pse_lib,
                                           newdata = GBSG2)
    
    model_info_list <- append(model_info_list,list(IPTW_IPCW_pse1 = IPTW_IPCW_10_pse1$CATE_est))
    
    #--- Collecting CATE estimate ---#
    GBSG2 <- cbind(GBSG2,IPTW_IPCW_pse1_CATE_est=as.vector(IPTW_IPCW_10_pse1$CATE_est))
    names(GBSG2)[length(GBSG2)] <- "IPTW_IPCW_pse1_CATE_est"
    
  },
  #if an error occurs, tell me the error
  error=function(e) {
    message("An error occured when fitting the IPTW_IPCW-learner - 10 CV folds - Single pseudo outcome set")
    print(e)
  }
)

#------------------------------------------------------------------------#
#--- Running IPTW-IPCW learner - 10 CV folds - All pseudo outcome set ---#
#------------------------------------------------------------------------#
tryCatch(
  {
    #--- Running model ---#
    IPTW_IPCW_10_pse2 <- IPTW_IPCW_learner(data = GBSG2,
                                           id = "ID2",
                                           outcome = "event_3yr",
                                           exposure = "trt",
                                           outcome_observed_indicator = "r",
                                           splits = 10,
                                           e_method = "Super learner",
                                           e_covariates = ps_cov_list,
                                           e_SL_lib = e_lib,
                                           g_method = "Super learner",
                                           g_covariates = imp_cov_list,
                                           g_SL_lib = imp_lib,
                                           pse_method = "Super learner",
                                           pse_covariates = pse2_cov_list,
                                           pse_SL_lib = pse_lib,
                                           newdata = GBSG2)
    
    model_info_list <- append(model_info_list,list(IPTW_IPCW_pse2 = IPTW_IPCW_10_pse2$CATE_est))
    
    #--- Collecting CATE estimate ---#
    GBSG2 <- cbind(GBSG2,IPTW_IPCW_pse2_CATE_est=as.vector(IPTW_IPCW_10_pse2$CATE_est))
    names(GBSG2)[length(GBSG2)] <- "IPTW_IPCW_pse2_CATE_est"
    
  },
  #if an error occurs, tell me the error
  error=function(e) {
    message("An error occured when fitting the IPTW-IPCW - 10 CV folds - Single pseudo outcome set")
    print(e)
  }
)


end <- Sys.time()
end-start


##################


# Save output files
store_name <- paste("Data_example_output_SL",taskID,sep = "_")
print(store_name)

# Save output
save(model_info_list, file= paste0(store_name,".RData") )




