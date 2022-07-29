# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 14:08:17 2020

@author: weixiong001

Runs multiple models with pipeline and other objects to train the 
machine learning models without data leakage. 
Several things use n_jobs parameter

Customised for go terms. HP test for all models
Modified from tt_multi_models.py in
~/machine_learning/ml_go_workflow//time_repeat
"""
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score
from datetime import datetime
import pickle

# Input variables/contants
FT_FILE = 'feature_type.txt'
ML_FILE = sys.argv[1]
# Added this code, as this script is for go term prediction
ML_FILE = ML_FILE.replace('go_GO:', 'go_GO_')
class_label = sys.argv[1]
temp_class_label = class_label.replace('go_GO:', 'go_GO_')
scores_file = temp_class_label + '_scores.txt'

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
ada = AdaBoostClassifier(random_state=42)
rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=120)
logr = LogisticRegression(solver='saga', n_jobs=120,
                          random_state=42, class_weight='balanced')
lsv = LinearSVC(random_state=42, class_weight='balanced')
gbc = GradientBoostingClassifier(random_state=42)
xgc = xgb.XGBClassifier(random_state=42, n_jobs=120)

# Models' hyperparameters
ada_grid = {
    'ada__n_estimators': [100, 120, 130, 150, 200],
    'ada__learning_rate': [0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75, 0.775, 0.8]
}

logr_grid = {
    'logr__C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
}

lsv_grid = {
    'lsv__C':  [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000],
}

rf_grid = {
    'rf__ccp_alpha': [0, 0.1, 0.001, 0.0001],
    'rf__max_features': ['sqrt', 0.1, 0.2, 0.3, 0.4, 0.5, 0.75],
    'rf__n_estimators': [50, 100, 200, 500],
    'rf__max_depth': [20, 50, 100, 200, None],
}

gbc_grid = {
    'gbc__learning_rate': [0.04, 0.06, 0.08, 0.1, 0.2],
    'gbc__subsample': [0.3, 0.5, 0.7, 1],
    'gbc__ccp_alpha': [0, 0.1, 0.001, 0.0001],
    'gbc__max_features': [None, 0.2, 0.3, 0.5, 0.7],
    'gbc__n_estimators': [50, 100, 200, 500],
    'gbc__max_depth': [1, 2, 3, 5, 7],
}

xgc_grid = {
    'xgc__learning_rate': [0.1, 0.2, None, 0.4, 0.5],
    'xgc__subsample': [0.3, 0.5, 0.7, None],
    'xgc__max_features': [None, 0.2, 0.3, 0.5, 0.7],
    'xgc__n_estimators': [50, 100, 200, 500],
    'xgc__max_depth': [1, 3, None, 9, 12],
}


# Container to hold all info for conversion to df for saving to .csv  
models = [(ada, 'ada', ada_grid),
          (rf, 'rf', rf_grid),
          (logr, 'logr', logr_grid),
          (lsv, 'lsv', lsv_grid),
          (gbc, 'gbc', gbc_grid),
          (xgc, 'xgc', xgc_grid)]
# This stores the start and end times of the model
# Named model_scores as its based on an earlier script
model_scores = []

for model in models:
    print(model[1], 'started:', get_time())
    time_start = get_time()
    # Pipleline to perform full workflow from preprocessing to model fitting
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          (model[1], model[0])
    ])

    # Create X and y data set
    X = data.drop(columns=class_label)
    y = data.loc[:, class_label]

    # K_fold objects
    outer_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Looping through the outer loop, feeding each training set into RSCV
    # as the inner loop
    for train_index,test_index in outer_kf.split(X, y):
        # Train sets and random search object
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]
        X_test = X.iloc[test_index]
        y_test = y.iloc[test_index]
        hp_sr = RandomizedSearchCV(estimator=clf, param_distributions=model[2],
                                   n_iter=10, scoring='f1', cv=inner_kf,
                                   n_jobs=120)

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
        time_end = get_time()
        run = pd.Series([model[1], tn, fp, fn, tp, f1, pre, re, time_start,
                         time_end],
                        index=['model_name', 'tn', 'fp', 'fn', 'tp', 'f1',
                               'precision', 'recall', 'time_start', 'time_end'])
        model_scores.append(run)

# Assembling results and saving it to file
# Named df_scores as its based on an earlier script
df_scores = pd.concat(model_scores, axis=0)
df_scores.index.name = 'id'
df_scores.to_csv(scores_file, sep='\t')
print('Script end:', get_time())
