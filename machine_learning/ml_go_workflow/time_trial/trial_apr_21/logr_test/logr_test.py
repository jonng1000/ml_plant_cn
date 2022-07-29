# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 14:08:17 2020

@author: weixiong001

Tests logistric regression/linear svm with pipeline and other
objects to perform the entire machine learning workflow
without data leakage. 
Several things use n_jobs parameter, main one is random search

Customised for go terms. Modified from mm_go_v2_logr_test.py in
~/machine_learning/older_ml_runs/workflow_v3/time_trial/logr_test$

"""
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score 
from datetime import datetime
import pickle

# Input variables/contants
KFOLD = 2  # number of k-folds
ITER = 1  # iterations for random search
FT_FILE = 'feature_type.txt'
ML_FILE = sys.argv[1]
# Added this code, as this script is for go term prediction
ML_FILE = ML_FILE.replace('go_GO:', 'go_GO_')
class_label = sys.argv[1]
fi_file = class_label + '_fi.txt'
scores_file = class_label + '_scores.txt'

def get_time():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")


print('Script started:', get_time())
# Reads in dataset, columns only
infile = open(ML_FILE,'rb')
pickle_df = pickle.load(infile)
infile.close()
# This is changed due to above code to make it compatible for GO terms
data =  pickle_df
# Reads in the file with the types of features
ft_df = pd.read_csv(FT_FILE, sep='\t', index_col=0)

# Gets separate lists of continuous and categorical features
# Creates dict to for downcasting
cont_feat = ft_df.loc[ft_df['Feature type'] == 'continuous', :].index
cat_feat = ft_df.loc[ft_df['Feature type'] == 'categorical', :].index
all_cont_feat = [x for x in data.columns if (x.split('_')[0] + '_') in cont_feat]
all_cat_feat = [x  for x in data.columns if (x.split('_')[0] + '_') in cat_feat]
dtype_dict = {}
for name in all_cont_feat:
    dtype_dict[name] = 'float32'
for name in all_cat_feat:
    dtype_dict[name] = 'int8'

# Not used here, as this script is for go term prediction
# Reads in whole dataset
#data = pd.read_csv(ML_FILE, dtype=dtype_dict, sep='\t', index_col=0)
# Added this code, as this script is for go term prediction
temp_cont = data[all_cont_feat].apply(pd.to_numeric, downcast='float')
temp_cat = data[all_cat_feat].apply(pd.to_numeric, downcast='integer')
data = pd.concat([temp_cont, temp_cat], axis=1)

# Remove class label from list of features, need to do this so that in the
# preprocessing stage, it will not try and find the class label column in
# the data
if class_label in set(all_cont_feat):
    all_cont_feat.remove(class_label)
elif class_label in set(all_cat_feat):
    all_cat_feat.remove(class_label)

# This code block executes ML workflow

# Pipeline object to fill missing NA values with median, and apply standard
# scaling, on continuous features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
# Pipeline object to fill missing NA values with 0, on categorical features
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=0))
])
# preprocessor preprocesses data according to Pipelines above
# Preprocessing takes about 1 min for full dataset
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, all_cont_feat),
        ('cat', categorical_transformer, all_cat_feat)
    ],
    n_jobs=-1)

# Models
logr = LogisticRegression(solver='saga', max_iter=5000,
                          random_state=42, class_weight='balanced',
                          n_jobs=50)
lsv = LinearSVC(random_state=42, class_weight='balanced')

# Models' hyperparameters
logr_grid = {
    'logr__C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
}

lsv_grid = {
    'lsv__C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
}

# Container to hold model output
logr_output = {'fi': [], 'scores': []}
lsv_output = {'fi': [], 'scores': []}
# Container to hold all info for conversion to df for saving to .csv  
# Comment below out as its just a placeholder, to remind me of what models
# to test to see when algo converges
'''
models = [(lsv_grid, lsv, lsv_output, 'lsv'),
          (logr_grid, logr, logr_output, 'logr'),
'''
models = [(logr_grid, logr, logr_output, 'logr')]

model_scores = []
fi_values = []
for model in models:
    print(model[3], 'started:', get_time())
    # Pipleline to perform full workflow from preprocessing to model fitting
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          (model[3], model[1])
    ])

    # Create X and y data set
    X = data.drop(columns=class_label)
    y = data.loc[:, class_label]

    # K_fold objects
    outer_kf = StratifiedKFold(n_splits=KFOLD, shuffle=True, random_state=42)
    inner_kf = StratifiedKFold(n_splits=KFOLD, shuffle=True, random_state=42)

    # Looping through the outer loop, feeding each training set into RSCV
    # as the inner loop
    for train_index,test_index in outer_kf.split(X, y):
        # Train sets and random search object
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]
        X_test = X.iloc[test_index]
        y_test = y.iloc[test_index]
        hp_sr = RandomizedSearchCV(estimator=clf, param_distributions=model[0],
                                   n_iter=ITER, scoring='f1', cv=inner_kf,
                                   n_jobs=100)

        # RSCV is looping through the training data to find the best parameters
        # This is the inner loop
        hp_sr.fit(X_train, y_train)
        # The best hyper parameters from RSCV is now being tested on the
        # unseen outer
        # loop test data
        y_pred = hp_sr.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        f1 = f1_score(y_test, y_pred)
        pre = precision_score(y_test, y_pred)
        re = recall_score(y_test, y_pred) 

        # Appending score
        run = pd.Series([model[3], tn, fp, fn, tp, f1, pre, re],
                        index=['model_name', 'tn', 'fp', 'fn', 'tp', 'f1',
                               'precision', 'recall'])
        model_scores.append(run)
        # Appending feature importance
        if model[3] in ['ada', 'brf', 'rf']:
            fi = hp_sr.best_estimator_.named_steps[model[3]].feature_importances_
        elif model[3] in ['logr', 'lsv']:
            fi = hp_sr.best_estimator_.named_steps[model[3]].coef_
            fi = fi[0]
        fi_sort = pd.Series(fi, index=X.columns).sort_values(ascending=False)
        fi_sort.name = model[3]
        fi_values.append(fi_sort)

# Assembling results and saving it to file
df_scores = pd.concat(model_scores, axis=1).T
df_scores.index.name = 'id'
df_fi = pd.concat(fi_values, axis=1)
df_fi.index.name = 'features'
df_scores.to_csv(scores_file, sep='\t')
df_fi.to_csv(fi_file, sep='\t')
print('Script end:', get_time())
