# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 14:08:17 2020

@author: weixiong001

Run RF model with pipeline and other objects to perform the entire machine learning
workflow without data leakage. This workflow is ran once just to test, and prints
out the classification report. Only ColumnTransformer and RandomForestClassifier
have the n_jobs parameter.

Wide range of hyperparameters, balanced class weights
RF uses 1 process, grid search uses 60 processes

This script took >1wk to run, so temrinated it
"""
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score 
from datetime import datetime

# Input variable
data_file = sys.argv[1]  # ml data file
FT_FILE = 'feature_type.txt'
scores_file = data_file.split('.')[0] + '_scores.txt'
RUNS = 1

def get_time():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")


print('Script started:', get_time())
# Reads in dataset, takes ~1 min
data = pd.read_csv(data_file, sep='\t', index_col=0)
# Reads in the file with the types of features
ft_df = pd.read_csv(FT_FILE, sep='\t', index_col=0)

# Gets separate lists of continuous and categorical features
cont_feat = ft_df.loc[ft_df['Feature type'] == 'continuous', :].index
cat_feat = ft_df.loc[ft_df['Feature type'] == 'categorical', :].index
all_cont_feat = [x for x in data.columns if (x.split('_')[0] + '_') in cont_feat]
all_cat_feat = [x  for x in data.columns if (x.split('_')[0] + '_') in cat_feat]

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

# Models' hyperparameters
rf_grid = {
    'classifier__n_estimators': [50, 100, 300, 500, 800, 1100, 1500],
    'classifier__max_features': [50, 100, 200, 500, 1000, 1500, 2000, 3000],
    'classifier__max_depth': [10, 20, 25, 40, 80, 160, None],
    'classifier__ccp_alpha': [0, 0.0001, 0.0002, 0.0004, 0.0007, 0.001, 0.005,
                              0.01, 0.05]
}

# Pipleline to perform full workflow from preprocessing to model fitting
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier',
                       RandomForestClassifier(random_state=42,
                                              class_weight='balanced'))
                     ])

X = data.drop(columns=['class_label'])
y = data.loc[:, 'class_label']

model_scores = []
for one_run in range(RUNS):
    print('run', one_run+1, 'started:', get_time())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                        random_state=42,
                                                        stratify=y)
    gd_sr = GridSearchCV(estimator=clf, param_grid=rf_grid,
                         scoring='f1', cv=10, n_jobs=60)
    gd_sr.fit(X_train, y_train)

    # Getting all output
    best_parameters = pd.Series(gd_sr.best_params_)
    y_pred = gd_sr.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    f1 = f1_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    re = recall_score(y_test, y_pred) 
    run = pd.Series([tn, fp, fn, tp, f1, pre, re],
                    index=['tn', 'fp', 'fn', 'tp', 'f1', 'precision',
                           'recall'], name='run')
    model_scores.append(run)

df_scores = pd.concat(model_scores, axis=1).T
df_scores.reset_index(drop=True, inplace=True)
df_scores.index.name = 'runs'
df_scores.to_csv(scores_file, sep='\t')
print('Script end:', get_time())
