# Copyright (c) 20017-2023 Francesco Gualdrini <francesco.gualdrini@gmail.com>

import sys
import os
from sklearn import preprocessing
from joblib import Memory
import joblib
from shutil import rmtree

import pickle
import json
import shutil
import numpy as np
import pandas as pd
import pylab 
import seaborn as sns
sns.set(style="ticks", color_codes=True, font_scale=1.5)
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import ListedColormap
from matplotlib import cm
from patsy import dmatrices
#%matplotlib inline
import mpl_toolkits
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import skew, norm, probplot, boxcox, f_oneway
from scipy import interp
from sklearn.base import BaseEstimator, TransformerMixin, clone, ClassifierMixin
from sklearn import metrics, tree, datasets
from sklearn.preprocessing import LabelEncoder, label_binarize, StandardScaler, PolynomialFeatures, MinMaxScaler, PowerTransformer, MinMaxScaler, minmax_scale, MaxAbsScaler, RobustScaler, Normalizer, QuantileTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold, cross_val_predict, train_test_split, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix, plot_confusion_matrix, f1_score, coverage_error, fbeta_score, average_precision_score
from skater.core.local_interpretation.lime.lime_tabular import LimeTabularExplainer
from skater.core.explanations import Interpretation
from skater.model import InMemoryModel
from sklearn.ensemble import VotingClassifier
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

# For reproducibility purpuse:
np.random.seed(31415)

# Parsing arguments:
import argparse

parser = argparse.ArgumentParser(description='Code to run XGBoost for multiple targets startin from same set of Features')
parser.add_argument('--targetki', action="store",help='name of target variable (column) in target matrix see --target');
parser.add_argument('--features', action="store",help='full path to matrix of featues with row anmes as in --target and column names');
parser.add_argument('--target', action="store",help='full path to matrix of target variables with categories encoded in --targetcat, row anmes must be within --features');
parser.add_argument('--savefolder', action="store",help='full path to folder to save results');
parser.add_argument('--targetcat', nargs='+',action="store",help='categories within --target to be supplied with space i.e. [-1 0 1]')

args = parser.parse_args();
print("targetki = %s" % args.targetki);
print("features = %s" % args.features);
print("target = %s" % args.target);
print("savefolder = %s" % args.savefolder);
print("targetcat = %s" % args.targetcat);

targetcat=args.targetcat
targetki=args.targetki

# Load the Features variable
XFeatures = pd.read_csv(args.features, sep='\t',index_col=0)
del XFeatures["Promoter"]
XFeatures = XFeatures.sort_index() 
print(XFeatures.shape)
# Check for finite numbers:
is_NaN = XFeatures.isnull()
row_has_NaN = is_NaN.any(axis=1)
rows_with_NaN = XFeatures[row_has_NaN]
np.any(np.isnan(XFeatures))
np.all(np.isfinite(XFeatures))

# Load target variables
YTargets = pd.read_csv(args.target, sep='\t',index_col=0)
YTargets = YTargets[YTargets.index.isin(XFeatures.index)]
YTargets = YTargets.sort_index() 
print(YTargets.shape)

# Fix naming -------------
cols = XFeatures.columns.str.replace(' ', '_')
cols = cols.str.replace('.', '_')
cols = cols.str.replace('(', '_')
cols = cols.str.replace(')', '_')
cols = cols.str.replace(',', '_')
cols = cols.str.replace(':', '_')
cols = cols.str.replace('-', '_')

X = XFeatures
X,y=X,YTargets

# save results
save_path=args.savefolder
subFolder = "XGBOOST_models_by_targets"
save = os.path.join(save_path,subFolder)

plt.style.use('default')
plt.rc('legend',fontsize=4)
plt.rc('axes', labelsize=4)

def get_results(model, name, data, true_labels, save_name, target_names = ["-1", "0", "1"], results=None, reasume=False):
    if hasattr(model, 'layers'):
        param = wtp_dnn_model.history.params
        best = np.mean(wtp_dnn_model.history.history['val_acc'])
        predicted_labels = model.predict_classes(data) 
        im_model = InMemoryModel(model.predict, examples=data, target_names=target_names)
    else:
        param = model.named_steps['gs'].best_params_
        best = model.named_steps['gs'].best_score_
        predicted_labels = model.predict(data).ravel()
        if hasattr(model, 'predict_proba'):
            im_model = InMemoryModel(model.predict_proba, examples=data, target_names=target_names)
        elif hasattr(clf, 'decision_function'):
            im_model = InMemoryModel(model.decision_function, examples=data, target_names=target_names)
    # arrange in table
    results_dict = {}
    results_dict['Mean Best Accuracy'] = str(best)
    results_dict['Best Parameters'] = str(param)
    results_df = pd.DataFrame([[str(best),dict(param)]],columns=['Mean Best Accuracy','Best Parameters'])
    results_df.to_csv(save_name + 'Best_Accuracy_Paramt.csv',sep ='\t')        
    y_pred = model.predict(data).ravel()
    display_model_performance_metrics(true_labels, predicted_labels = predicted_labels, target_names = target_names, save_name=save_name)
    if len(target_names)==2:
        ras = roc_auc_score(y_true=true_labels, y_score=y_pred)
    else:
        roc_auc_multiclass, ras = roc_auc_score_multiclass(y_true=true_labels, y_score=y_pred, target_names=target_names)
        results_df = pd.DataFrame(list(roc_auc_multiclass.items()))
        results_df.to_csv(save_name + 'roc_auc_multiclass.csv',sep ='\t') 
    prob, score_roc, roc_auc = plot_model_roc_curve(model, data, true_labels, save_name, label_encoder=None, class_names=target_names)
    # by class feature importance
    # classes
    classes = model.named_steps['gs'].classes_
    stt=[]
    for ss in model.steps:
        stt.append(ss[0])
    res = True in (ele == "pca" for ele in stt)
    if res==True:
        n_pcs= model.named_steps['pca'].components_.shape[0]
        most_important = [np.abs(model.named_steps['pca'].components_[i]).argmax() for i in range(n_pcs)]
        initial_feature_names = model.named_steps['get_cols'].feature_names
        most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]
        dic = {'PC{}'.format(i+1): most_important_names[i] for i in range(n_pcs)}
        df = pd.DataFrame(dic.items())
        feature_names = df[0]
    else:
        feature_names = np.array(model.named_steps['get_cols'].feature_names)
    ##
    r1 = pd.DataFrame([(prob, best, np.round(accuracy_score(true_labels, predicted_labels), 4), 
                         ras, roc_auc)], index = [name],
                         columns = ['Prob', 'CV Accuracy', 'Accuracy', 'ROC AUC Score', 'ROC Area'])
    if reasume:
        results = r1
    elif (name in results.index):        
        results.loc[[name], :] = r1
    else: 
        results = results.append(r1)
    return results

def roc_auc_score_multiclass(y_true, y_score, target_names, average = "macro"):
  #creating a set of all the unique classes using the actual class list
  unique_class = set(y_true)
  roc_auc_dict = {}
  mean_roc_auc = 0
  for per_class in unique_class:
    #creating a list of all the classes except the current class 
    other_class = [x for x in unique_class if x != per_class]
    #marking the current class as 1 and all other classes as 0
    new_y_true = [0 if x in other_class else 1 for x in y_true]
    new_y_score = [0 if x in other_class else 1 for x in y_score]
    num_new_y_true = sum(new_y_true)
    #using the sklearn metrics method to calculate the roc_auc_score
    roc_auc = roc_auc_score(new_y_true, new_y_score, average = average)
    roc_auc_dict[target_names[per_class]] = np.round(roc_auc, 4)
    mean_roc_auc += num_new_y_true * np.round(roc_auc, 4)
  mean_roc_auc = mean_roc_auc/len(y_true)  
  return roc_auc_dict, mean_roc_auc

def get_metrics(true_labels, predicted_labels):
    # arrange in table
    results_dict = {}
    results_dict['Accuracy'] = metrics.accuracy_score(true_labels, predicted_labels)
    results_dict['Precision'] = metrics.precision_score(true_labels, predicted_labels, average='weighted')
    results_dict['Recall'] = metrics.recall_score(true_labels, predicted_labels, average='weighted')
    results_dict['F1 Score'] = metrics.f1_score(true_labels, predicted_labels, average='weighted')
    results_df = pd.DataFrame(list(results_dict.items()))     
    return(results_df)
                        
def train_predict_model(classifier,  train_features, train_labels,  test_features, test_labels):
    # build model    
    classifier.fit(train_features, train_labels)
    # predict using model
    predictions = classifier.predict(test_features) 
    return predictions    

def display_confusion_matrix(true_labels, predicted_labels, target_names):
    total_classes = len(target_names)
    level_labels = [total_classes*[0], list(range(total_classes))]
    cm = metrics.confusion_matrix(y_true=true_labels, y_pred=predicted_labels)
    cm_frame = pd.DataFrame(data=cm)
    cm_frame.columns = pd.MultiIndex.from_tuples(zip(['Predicted:','Predicted:','Predicted:'], target_names))
    cm_frame.index = pd.MultiIndex.from_tuples(zip(['Actual:','Actual:','Actual:'], target_names))
    return(cm_frame) 
    
def display_classification_report(true_labels, predicted_labels, target_names):
    report = metrics.classification_report(y_true=true_labels, y_pred=predicted_labels, target_names=list(map(str,target_names)), output_dict=True)
    report = pd.DataFrame(report).transpose()
    return(report)
    
def display_model_performance_metrics(true_labels, predicted_labels, save_name, target_names):
    model_performance = get_metrics(true_labels=true_labels, predicted_labels=predicted_labels)
    model_performance.to_csv(save_name + 'performance.csv',sep ='\t') 
    classification_report = display_classification_report(true_labels=true_labels, predicted_labels=predicted_labels, target_names=target_names)
    classification_report.to_csv(save_name + 'classification_report.csv',sep ='\t') 
    confusion_matrix = display_confusion_matrix(true_labels=true_labels, predicted_labels=predicted_labels, target_names=target_names)
    confusion_matrix.to_csv(save_name + 'confusion_matrix.csv',sep ='\t') 

def plot_model_roc_curve(clf, features, true_labels, save_name, label_encoder=None, class_names=None):
    ## Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()   
    if hasattr(clf, 'classes_'):
        class_labels = clf.classes_
    elif label_encoder:
        class_labels = label_encoder.classes_
    elif class_names:
        class_labels = class_names
    else:
        raise ValueError('Unable to derive prediction classes, please specify class_names!')
    n_classes = len(class_labels)
    if n_classes == 2:
        if hasattr(clf, 'predict_proba'):
            prb = clf.predict_proba(features)
            if prb.shape[1] > 1:
                y_score = prb[:, prb.shape[1]-1] 
            else:
                y_score = clf.predict(features).ravel()
            prob = True
        elif hasattr(clf, 'decision_function'):
            y_score = clf.decision_function(features)
            prob = False
        else:
            raise AttributeError("Estimator doesn't have a probability or confidence scoring system!")
        fpr, tpr, _ = roc_curve(true_labels, y_score)      
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label='ROC curve (area = {0:3.2%})'.format(roc_auc), linewidth=2.5)
    elif n_classes > 2:
        if  hasattr(clf, 'clfs_'):
            y_labels = label_binarize(true_labels, classes=list(range(len(class_labels))))
        else:
            y_labels = label_binarize(true_labels, classes=class_labels)
        if hasattr(clf, 'predict_proba'):
            y_score = clf.predict_proba(features)
            prob = True
        elif hasattr(clf, 'decision_function'):
            y_score = clf.decision_function(features)
            prob = False
        else:
            raise AttributeError("Estimator doesn't have a probability or confidence scoring system!")
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_labels[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        ## Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_labels.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        ## Compute macro-average ROC curve and ROC area
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        # Finally average it and compute AUC
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        results_df = pd.DataFrame(roc_auc.items(),columns=['curve', 'ROC'])
        results_df.to_csv(save_name + 'ROC_by_class.csv',sep ='\t')
        ## Plot ROC curves
        plt.figure(figsize=(6, 4))
        plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:2.2%})'
                       ''.format(roc_auc["micro"]), linewidth=3)
        plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve (area = {0:2.2%})'
                       ''.format(roc_auc["macro"]), linewidth=3)
        for i, label in enumerate(class_names):
            plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:2.2%})'
                                           ''.format(label, roc_auc[i]), linewidth=2, linestyle=':')
        roc_auc = roc_auc["macro"]   
    else:
        raise ValueError('Number of classes should be atleast 2 or more')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(save_name +'ROC.pdf',format='pdf')
    precision = dict()
    recall = dict()
    average_precision = dict()
    if n_classes > 2: 
        plt.figure(figsize=(6, 4))
        for i, label in enumerate(class_names):
            precision[i], recall[i], _ = precision_recall_curve(y_labels[:, i], y_score[:, i])
            average_precision[i] = average_precision_score(y_labels[:, i], y_score[:, i])
            plt.plot(recall[i], precision[i], label='Precision recall {0} (area = {1:2.2%})'
                                           ''.format(label, average_precision[i]), linewidth=2, linestyle=':')  
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([-0.01, 1.0])
        plt.ylim([0.0, 1.01])
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.title('precision vs. recall curve')
        plt.legend(loc="lower right")
        plt.savefig(save_name +'Precision_Recall.pdf',format='pdf')
    average_precision_df = pd.DataFrame(average_precision.items(),columns=['curve', 'average_precision'])
    average_precision_df.to_csv(save_name + 'average_precision_by_class.csv',sep ='\t')
    return prob, y_score, roc_auc

def myplot(score,coeff,y,labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex,ys * scaley, c = y,s=0.1)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names
    def fit(self, X, y = None):
        return self
    def transform(self, X, y = None):
        return X[self.feature_names]

scorers = { 'f1': make_scorer(f1_score , average='macro'),
           'f1w': make_scorer(f1_score , average='weighted'),
           'balancedAUC': make_scorer(balanced_accuracy_score , adjusted=True),
            'roc_auc_ovr_scorer_weighted': make_scorer(roc_auc_score, needs_proba=True ,multi_class='ovr',average = 'weighted'),
            'roc_auc_ovr_scorer_macro': make_scorer(roc_auc_score, needs_proba=True ,multi_class='ovr',average = 'macro')
            }

cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=101)

# Print out name
print(targetki)
# Make the output folder
inhibit_folder="Inhibitor_" + targetki 
save = os.path.join(save_path,subFolder,inhibit_folder)
if not os.path.exists(save):
    os.makedirs(save)
    
le = preprocessing.LabelEncoder()
le.fit(targetcat)
y_sel = le.transform(y[targetki])
X_train, X_test, y_train, y_test = train_test_split(X, y_sel, test_size=0.3, random_state=101,stratify=y_sel)

# We deal with multiclass classification:
# stratify by a mixed y:
# compute the custome weight:
n_sample = y_train.shape[0]
n_0 = y_train[y_train == 1].shape[0]
n_d = y_train[y_train == 0].shape[0]
n_u = y_train[y_train == 2].shape[0]
w_0 = n_sample/(2*n_0)
w_eff = n_sample/(2*(n_sample-n_0))
w_u = (n_u/(n_d+n_u))*w_eff
w_d = (n_d/(n_d+n_u))*w_eff
w_0c = w_0/(w_0+w_u+w_d)
w_uc = w_u/(w_0+w_u+w_d)
w_dc = w_d/(w_0+w_u+w_d)
custom_weight = {0:w_dc, 1:w_0c,2:w_uc}

sample_weights = compute_sample_weight(
    class_weight=custom_weight,
    y=y_train#provide your own target name
)

clf = xgb.XGBClassifier(base_score=0.5, colsample_bylevel=1, n_jobs=4,eval_metric='mlogloss',booster = 'gbtree', objective='multi:softprob',
                                            use_label_encoder=False,colsample_bytree=1, gamma=0.0001, max_delta_step=0, random_state=101, 
                                           subsample=1,verbosity = 1)

# a list of dictionaries to specify the parameters that we'd want to tune
learning_rate = [ 0.1, 0.2, 0.3 ,0.4] #4
n_est = [60, 120,240,480] #4
max_depth = [3,6,12] #3
subsample = [0.8] #2
colsample_bytree = [ 0.8] #2
gamma= [0,0.5,1.5,2] #4
reg_alpha=[0,0.1,0.2,0.4,0.8] #5
reg_lambda=[0.1,0.5,1,1.5,2] #5

param_grid = [{ 
        'gamma': gamma
        ,'learning_rate': learning_rate
        ,'n_estimators': n_est
        ,'subsample':subsample
        ,'max_depth': max_depth
        ,'colsample_bytree': colsample_bytree
        ,'reg_alpha':reg_alpha
        ,'reg_lambda':reg_lambda 
}]

######################################################################################################
# MODEL ING
get_cols=FeatureSelector(feature_names=XFeatures.columns.tolist())
fit_params ={'gs__sample_weight':sample_weights} 
gs = GridSearchCV(estimator=clf, param_grid=param_grid, scoring=scorers, refit='balancedAUC', cv=cv, verbose=2, n_jobs=3)
XGBOOST_model = Pipeline([('get_cols',get_cols),('gs', gs)])

XGBOOST_model.fit(X_train,y_train,**fit_params)

save_name=save+'/XGBOOST_model'
results = get_results(XGBOOST_model, 'XGBOOST_model', X_test, y_test, target_names= XGBOOST_model.classes_ , save_name=save_name, reasume=True)
# Save the pipeline and the gs
filename = save + '/XGBOOST_model.sav'
with open(filename, 'wb') as file:
    joblib.dump(XGBOOST_model, file)
results.to_csv(save + '/Final_results.txt',sep ='\t',header=True, index=True)
