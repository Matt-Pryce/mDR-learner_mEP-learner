# mDR-learner/mEP-learner
Implements the missing outcome extensions of the DR-learner and EP-learner. 

mDR-learner/mEP-learner
  - Nuisance parameters can be estimated inside function by specifying model
    parameters or they can be estimated outside the function and input. 
  - Crossfitting can be implemented using 10 CV validation (or not used)
  - If half sample bootstrap CIs are to be estimated, the pseudo-outcome regression
    must be run using method="Random forest", rf_CI must be set to TRUE and num_boot must be specifed 
    

DR-learner/EP-learner
  - Allows for missing outcomes to be imputed (using Parametric/RF/SL model)
  - Alternatively, can run using only complete cases or available cases 
  - If want to fit outcome imputation outside function then run Complete case 
    and specify the outcome to be the variable with missing values imputed. 
  - Imputation will not run if nuisance estimates are input and not estimated,
    instead if wish to use imputation in this setting, treat the analysis as a 
    complete case analysis 
  - Half sample bootstrap CIs can be estimated for these learners using the same guidance as above. 
    

mDR-learner longitudinal extension
  - Code for this extension follows the same notation as for the mDR-learner
  - This can be found in script/mDRL_learner.

T-learner
  - Same notes as DR-learner/EP-learner
  - If nuisance functions input, T-learner can only make predictions of input data 



**Supplementary materials**

Simulation study
  - A script which can be run on a HPC is provided for DGP 1 (sample size 400)
  - An example batchscript along with the parameter scripts and test datasets are also provided

GBSG2 trial analysis
  - A script which can be run on a HPC is provided 
  - 10 iterations were run to obtain median CATE estimates
