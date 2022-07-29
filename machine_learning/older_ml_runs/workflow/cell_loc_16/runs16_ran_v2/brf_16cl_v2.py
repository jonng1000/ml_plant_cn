# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 14:08:17 2020

@author: weixiong001

Run balanced RF model with pipeline and other objects to perform the entire 
machine learning workflow without data leakage. 
Column transformer, model and grid search have the n_jobs parameter.

Grid search uses multiple processes, 5-fold cv
Improved version of brf_16cl.py
"""
import sys
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score 
from datetime import datetime

# Input variables/contants
FT_FILE = '~/machine_learning/feature_type.txt'
DATA_FOLDER = './data_16'
MODEL_NAME = 'brf'
# Seed number for test train split, and to separate files
# from different repeats
NUM = int(sys.argv[1])

def get_time():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")


print('Script started:', get_time())

model_scores = []
for a_file in os.listdir(DATA_FOLDER):
    # This code block involves iterating and reading in files
    file_path = DATA_FOLDER + '/' + a_file
    # cell location name
    cell_loc = a_file.split('_', 1)[1].split('.')[0]
    # Reads in dataset, columns only
    data =  pd.read_csv(file_path, sep='\t', index_col=0, nrows=0)
    # Reads in the file with the types of features
    ft_df = pd.read_csv(FT_FILE, sep='\t', index_col=0)
    
    # Gets separate lists of continuous and categorical features
    cont_feat = ft_df.loc[ft_df['Feature type'] == 'continuous', :].index
    cat_feat = ft_df.loc[ft_df['Feature type'] == 'categorical', :].index
    all_cont_feat = [x for x in data.columns if (x.split('_')[0] + '_') in cont_feat]
    all_cat_feat = [x  for x in data.columns if (x.split('_')[0] + '_') in cat_feat]
    class_labels = [x for x in data.columns if x.startswith('class_')]
    uint8_mapping = all_cat_feat + class_labels
    dtype_dict = {name:'uint8' for name in uint8_mapping}

    # Reads in whole dataset
    data = pd.read_csv(file_path, dtype=dtype_dict, sep='\t', index_col=0)

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

    # Models' hyperparameters
    grid = {
        MODEL_NAME + '__max_features': [40, 80, 200],
        MODEL_NAME + '__n_estimators': [100, 200, 500],
        MODEL_NAME + '__max_depth': [20, 50, None],
    }

    # Pipleline to perform full workflow from preprocessing to model fitting
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          (MODEL_NAME,
                           BalancedRandomForestClassifier(random_state=42))
    ])

    X = data.drop(columns=['class_label'])
    y = data.loc[:, 'class_label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                        random_state=NUM,
                                                        stratify=y)
    gd_sr = GridSearchCV(estimator=clf, param_grid=grid,
                         verbose=1, scoring='f1', cv=5, n_jobs=120)
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

    # Create grid search results file
    gd_sr_results = pd.DataFrame.from_dict(gd_sr.cv_results_)
    gd_sr_results.index.name = 'id'
    gsr_name = MODEL_NAME + '_' + cell_loc + str(NUM) + '_gsr.txt'
    gd_sr_results.to_csv(gsr_name, sep='\t')

    # Create feature importance file
    fi = gd_sr.best_estimator_.named_steps[MODEL_NAME].feature_importances_
    fi_sort = pd.DataFrame(fi, index=X_train.columns,
                           columns=[cell_loc + '_importance']).\
                           sort_values(cell_loc + '_importance', ascending=False)
    fi_sort.index.name = 'feature'

    fi_sort_name = MODEL_NAME + '_' + cell_loc + str(NUM) + '_fi.txt'
    fi_sort.to_csv(fi_sort_name, sep='\t')

    print(gd_sr.best_params_)
    print(a_file, 'finished',  get_time())
    
df_scores = pd.concat(model_scores, axis=1).T
df_scores.reset_index(drop=True, inplace=True)
df_scores.index.name = 'id'
scores_file = MODEL_NAME + str(NUM) + '_c16_scores.txt'
df_scores.to_csv(scores_file, sep='\t')

print('Script end:', get_time())
