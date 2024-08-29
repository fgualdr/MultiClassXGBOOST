# MultiClassXGBOOST with SHAPLEY Feature Importance

## run_xgboost.py:
A Singularity environment for reproducing the analysis can be downloaded from: docker://fgualdr/xgboostml.

This code implements an XGBoost classifier to model multiple targets using the same set of features. It has been used to predict the action of Kinase Inhibitors (KIs) using multiple features such as chromatin modifications, TF ChIP-seq binding, etc. The code is designed for multiclass classification to simultaneously predict upregulation, downregulation, or no changes induced by KI treatment. The model works with three categories labeled as "-1" (down effects), "0" (no changes), and "1" (up effects). Due to the dataset's imbalanced nature, custom weights have been computed to give higher importance to up or down effects compared to non-changing effects, which constitute the majority of observations. Among the changing effects (up or down), higher weight is assigned to the class with the most effects.

## The custom weight is computed as follows:
n_sample = targetki.shape[0]
n_0 = targetki[targetki == 0].shape[0]  # number of no effects
n_d = targetki[targetki == -1].shape[0]  # number of down effects
n_u = targetki[targetki == 1].shape[0]  # number of up effects

w_0 = n_sample / (2 * n_0)  # weight no-effects vs effects
w_eff = n_sample / (2 * (n_sample - n_0))  # weight effects vs no-effects
w_u = (n_u / (n_d + n_u)) * w_eff  # weight for up-reg effects
w_d = (n_d / (n_d + n_u)) * w_eff  # weight for down-reg effects

## Individual scalings are then re-scaled to range between 0 and 1, and the total sum equals 1:
w_0c = w_0 / (w_0 + w_u + w_d)
w_uc = w_u / (w_0 + w_u + w_d)
w_dc = w_d / (w_0 + w_u + w_d)
custom_weight = {0: w_0c, 1: w_uc, 2: w_dc}

This approach favors observable effects by giving higher importance to the main effects observed. If the frequencies of up and down regulatory effects are equal, the weights are equally distributed.

## The code requires the following inputs:

A matrix of features with indexes and headers named "features" (numeric matrix).
A matrix of target variables, one per column, with the same encoding specifying the categories ("target").
The name of the target to be used for XGBoost modeling ("targetki").
Categories within the target matrix ("targetcat") in the form of [-1, 0, 1].
The path to the folder where results will be saved.
The code performs classification using XGBoost, with parameter tuning via GridSearchCV for 4,800 candidates and 3x3 RepeatedStratifiedKFold cross-validation. This process involves fitting 9 folds for each of the 4,800 candidates, totaling 43,200 fits. The code utilizes 12 CPUs.

## To generate a Python environment to execute the code, run:
conda create -n env scikit-learn numpy pandas matplotlib xgboost skater shap seaborn cycler dill scipy patsy

## The code can then be executed as:
python test.py --targetki ki1 --features /path/to/features_matrix/ --target /path/to/target_matrix/ --savefolder /path/to/save_folder/ --targetcat -1 0 1

# Shapley Values for Feature Importance:
The code proceeds to compute Shapley values using a recently published Python implementation. For details, see the following resource: SHAP documentation.

For each output class, the code computes the Shapley values per observation and the Shapley feature interaction.
