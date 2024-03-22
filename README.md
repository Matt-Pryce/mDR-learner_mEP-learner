# mDR-learner
Implements the missing outcome extension of the DR-learner. 

Checking 


mDR-learner 
  - Nuisance parameters can be estimated inside function by specifying model
    parameters or they can be estimated outside the function and input. 
  - Crossfitting is implemented using 10 CV validation, 4 independent splits or
    no crossfitting 
  - If nuisance parameters input, do not use splits = 4. 

DR-learner 
  - Allows for missing outcomes to be imputed (using Parametric/RF/SL model)
  - Alternatively, can run using only complete cases or an available case 
    approach, using all available data for each model.
  - If want to fit outcome imputation outside function then run Complete case 
    and specify outcome as imputed outcome 
  - Won't run imputation if nuisance estimates are input and not estimated,
    instead treats it as a complete case analysis
    
T-learner
  - Same notes as DR-learner
  - If nuisance functions input, T-learner can only make predictions of input data 