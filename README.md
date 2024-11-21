# mDR-learner
Implements the missing outcome extensions of the DR-learner and EP-learner. 

mDR-learner 
  - Nuisance parameters can be estimated inside function by specifying model
    parameters or they can be estimated outside the function and input. 
  - Crossfitting can be implemented using 10 CV validation (or not used)

DR-learner 
  - Allows for missing outcomes to be imputed (using Parametric/RF/SL model)
  - Alternatively, can run using only complete cases or available cases 
  - If want to fit outcome imputation outside function then run Complete case 
    and specify outcome as the outcome which has missing values imputed 
  - Won't run imputation if nuisance estimates are input and not estimated,
    instead treats it as a complete case analysis
    
T-learner
  - Same notes as DR-learner
  - If nuisance functions input, T-learner can only make predictions of input data 
  

""Supplementary materials""

Simulation study
  - A script which can be run on a HPC is provided for DGP 1 (sample size 400)
  - An example batchscript along with the parameter scripts and test datasets are also provided

ACTG175 trial analysis
  - A script which can be run on a HPC is provided 
  - 50 iterations were run to obtain median CATE estimates