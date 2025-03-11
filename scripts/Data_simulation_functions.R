#######################################################################################
# Script: Function to simulate data for the extended DR-learner simulations where we
#         have a binary treatment setting and missingness. 
# Date: 02/02/23
# Author: Matt Pryce 
# Notes: Specify propensity score, unexposed outcome function, CATE, missingness function
#        and error. 
#######################################################################################

library(tidyverse)
library(stringr)
library(panelr)
library(lava)
library(MASS)
library(reshape2)
library(data.table)


#########################################################################
#--- Uni-variate outcome with missingness - Binary treatment setting ---#  
#########################################################################

#' @param n Number of observations to be simulated
#' @param x The number of covariates
#' @param x_form The distribution of the covariates; choose from either ("Unif","Binary")
#' @param e The function which generates the propensity score. Must take arguments from X1 - Xk, where k is number of covariates
#' @param Y_0 The function which generates the baseline outcomes for the covariate set w
#' @param Tau The function which generates the CATE for the covariate set w
#' @param G The function which generates the probability of not having a missing outcome. 
#' @param error The natural variation introduced in the unexposed outcome function. 
#' @param outcome_form Whether the outcome is binary or continuous 
# 
#' @return A dataset containing n observations 


data_sim_uni_miss <- function(n = 1000,
                              x = 10,
                              x_form = c("Unif","Norm"),
                              e,
                              Y_0,
                              Tau,
                              G,
                              error,
                              outcome_form = c("Binary","Continuous")
){
  ID <- rep(1:n)
  
  #Generate baseline covariates
  if (x_form == "Unif"){
    if (x>1){
      X <- data.frame(matrix(runif(n*x,-1,1), n, x))
      X <- data.frame(cbind(ID,X))
    }
    if (x==1){
      X1 <- runif(n,-1,1)
      X <- data.frame(cbind(ID,X1))
    }
  }
  if (x_form == "Norm"){
    tot_num_covs <- x
    Sigma <- diag(x = 1, nrow = tot_num_covs, ncol = tot_num_covs)
    offdiag(Sigma, type = 0) <- 0
    X <- data.frame(mvrnorm(n = n, mu = rep(x = 0, times = tot_num_covs), Sigma,empirical = T))
    X <- data.frame(cbind(ID,X))
  }
  

  #Generate the propensity score
  prop_score <- e(X)

  #Generate the treatment
  A <- rbinom(n, size=1, prob= prop_score)
  X <- cbind(X,A)

  #Generate Y_0 & Y_1 probabilities
  Y.0_prob_true <- Y_0(X)

  Y.1_prob_true <- Y.0_prob_true + Tau(X)
  
  
  #Calculating true CATE for test data
  CATE_True <- Y.1_prob_true - Y.0_prob_true
  

  #Generating Y.0 probability with natural variation
  if (error == "Normal"){
    epsilon <- rnorm(length(Y.0_prob_true),mean=0,sd=0.25)
  }
  Y.0_prob <- Y.0_prob_true + epsilon
  Y.1_prob <- Y.0_prob + Tau(X)

  if (outcome_form == "Binary"){
    #Generating the outcome for each individual if treated/untreated
    Y.0 <-  rbinom(n, size=1,prob = Y.0_prob)

    Y.1 <-  rbinom(n, size=1,prob = Y.1_prob)
  }
  else {
    Y.0 <-  Y.0_prob

    Y.1 <-  Y.1_prob
  }

  #Generating the observed outcome
  Y_no_miss <- (Y.1*A) + (Y.0*(1-A))

  #Generating the probability of not having a missing outcome
  G_prob <- G(X)

  #Identifying who has a missing outcome
  G_obs <- rbinom(n, size=1, prob= G_prob)
  Y <- Y_no_miss
  for (i in 1:n){
    if (G_obs[i] == 0){
      Y[i] <- NA
    }
  }
  C <- 1-G_obs

  # #Collating dataset
  sim_data <- data.frame(X,prop_score,Y.0_prob_true,Y.1_prob_true,CATE_True,epsilon,Y.0,Y.1,Y_no_miss,G_prob,G_obs,Y,C)
  return(sim_data)
}

#####################################################################

# e_func <- ps_funcs[[ps_list[5]]] 
# 
# #Selecting censoring function
# cen_func <- cen_funcs[[cen_list[5]]]
# 
# #Selecting baseline outcome function
# Y0_func <- Y0_funcs[[Y0_list[5]]]
# 
# #Selecting CATE function
# tau_func <- tau_funcs[[tau_list[5]]]
# 
# n=400
# 
# covs <- 6
# 
# sim_data <- data_sim_uni_miss(n = 2*n, x = covs, x_form = "Unif", 
#                               e = e_func, 
#                               Y_0 = Y0_func,   
#                               Tau = tau_func, 
#                               G=cen_func, 
#                               error = "Normal",
#                               outcome_form = "Countinuous")


############################

