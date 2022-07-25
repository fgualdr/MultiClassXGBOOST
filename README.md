# MultiClassXGBOOST with SHAPLEY Feature Importance

## run_xgboost.py:

Code to run XGBoost classifier on multiuple targets using same set of features.
This code has been used to model Kinase Inhibitor (KI) action using Multiple features (i.e. chromatin modifications, TFs chip-seq binding etc..).
The code has been designed to predict simultaneousle (multiclass type) up, down, and no changes induced by the KI treatment. 
The code will only work with 3 categories labelled "-1"=down effects "0"= no changes "1"= up effects.
Given the degree of unbalancing of the dataset custome weight has been computed giving overall higher importance to either up or down effects as opposed to the non changing (which compose the vast majority of the observations). Then within the those that chang, either up or dow regulatory effects, a higher weigth was attributed to the class with most effects.

### The custome weight will be therefore computed as follow:

n_sample = targetki.shape[0]\
n_0 = targetki[targetki == 0].shape[0] # number of no effects\
n_d = targetki[targetki == -1].shape[0] # number of down effects\
n_u = targetki[targetki == 1].shape[0] # number of up effects\

w_0 = n_sample/(2*n_0) # weight no-effects vs effects\
w_eff = n_sample/(2*(n_sample-n_0))  # weight effects vs no-effects\

w_u = (n_u/(n_d+n_u))*w_eff # weight for up-reg effects\
w_d = (n_d/(n_d+n_u))*w_eff # weight for down-reg effects\

### individual scalings were then re-scaled to range between 0 and 1 and the total sum will euql 1.
w_0c = w_0/(w_0+w_u+w_d)\
w_uc = w_u/(w_0+w_u+w_d)\
w_dc = w_d/(w_0+w_u+w_d)\
custom_weight = {0:w_dc, 1:w_0c,2:w_uc}\

In this way we will assemble a model in fabvour of observable effects giving higer importance to the main effects observed.
When the frequency of up and down regulatory effects is equal the weight is also equally distributed.


### The code imports:

1. matrix of featuares with indexes and header "features". This is supposed to be numeric matrix.
2. a matrix of target variables one per columns with same encodings specifying the categories "target".
3. the name of the target to run the XGBoost modelling "targetki"
4. Categories within the target matrix "targetcat" in the form of i.e. [-1 0 1]
5. path to the folder where to save the results

The code performs Classification via XGBoost performing parameter tuning via GridSearchCV for 4800 candidates, and 3x3 RepeatedStratifiedKFold Cross Validation. Therefore fitting 9 folds for each of 4800 candidates, totalling 43200 fits.
The code exploit 12CPUs.


To generate a python environment to execute the code run:
conda create -n env scikit-learn numpy pandas matplotlib xgboost skater shap seaborn cycler dill scipy patsy

code can be run as:

python test.py \
--targetki ki1 \
--features /path/to/features_matrix/ \
--target /path/to/target_matrix/ \
--savefolder /path/to/save_folder/ \
--targetcat -1 0 1

## run_shapley.py:
