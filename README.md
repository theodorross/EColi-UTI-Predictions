# EColi-UTI-Predictions
Using AST data from UTI causing E. coli isolates to predict subpopulation membership.

The code in this repository trains a Random Forest classifier and an XGBoost classifier to predict whether or not a bacterial isolate belongs to ST131 clade C.  This prediction is made using disk diffusion measurements from antibiotic susceptibility testing for ciprofloxacin, ceftazidim, and gentamicin.

## Reproducing
The shell script `create-env.sh` 
