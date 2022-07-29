# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 14:08:17 2020

@author: weixiong001

Run a  model with pipeline and other objects to perform the entire machine 
learning workflow without data leakage. 
Column transformer, random forest and grid search have the n_jobs parameter.

Grid search uses multiple processes, 5-fold cv
"""
import sys
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score 
from datetime import datetime

# Input variables/contants
FT_FILE = '~/machine_learning/feature_type.txt'
DATA_FOLDER = './data_16'
SCORES_FILE = 'knn_cell16_scores.txt'

def get_time():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")


print('Script started:', get_time())

model_scores = []
for a_file in os.listdir(DATA_FOLDER):
    # This code block involves iterating and reading in files
    file_path = DATA_FOLDER + '/' + a_file
    # cell location name
    cell_loc = a_file.split('_', 1)[1].split('.')[0]
    # Reads in dataset, takes ~1 min
    data = pd.read_csv(file_path, sep='\t', index_col=0)
    # Reads in the file with the types of features
    ft_df = pd.read_csv(FT_FILE, sep='\t', index_col=0)

    # This code block executes ML workflow
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
    grid = {
        'knn__n_neighbors': [3, 5, 10, 20, 50],
        'knn__weights': ['uniform', 'distance'],
        'knn__p': [1, 2, 3]
    }
    
    # Pipleline to perform full workflow from preprocessing to model fitting
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('knn',
                           KNeighborsClassifier())
    ])

    X = data.drop(columns=['class_label'])
    y = data.loc[:, 'class_label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                        random_state=42,
                                                        stratify=y)
    gd_sr = GridSearchCV(estimator=clf, param_grid=grid,
                         verbose=2, scoring='f1', cv=5, n_jobs=50)
    gd_sr.fit(X_train, y_train)

    # Getting all output
    best_parameters = pd.Series(gd_sr.best_params_)
    y_pred = gd_sr.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    f1 = f1_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    re = recall_score(y_test, y_pred) 
    run = pd.Series([cell_loc, tn, fp, fn, tp, f1, pre, re],
                    index=['cell_loc', 'tn', 'fp', 'fn', 'tp', 'f1', 'precision',
                           'recall'], name='run')
    model_scores.append(run)
    print(a_file, 'finished',  get_time())

df_scores = pd.concat(model_scores, axis=1).T
df_scores.reset_index(drop=True, inplace=True)
df_scores.index.name = 'id'
df_scores.to_csv(SCORES_FILE, sep='\t')
print('Script end:', get_time())
