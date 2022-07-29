# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 14:08:17 2020

@author: weixiong001

Run random forest with pipeline and other objects to perform the entire machine 
learning workflow without data leakage. 
Several things use n_jobs parameter, main one is random search

Improved version of ada_16cl_v2.py
"""
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score 
from datetime import datetime

# Input variables/contants
FT_FILE = 'feature_type.txt'
ML_FILE = 'ml_dataset_dc.txt'
MODEL_NAME = 'rf'
class_label = sys.argv[1]
fi_file = class_label + '_fi.txt'
scores_file = class_label + '_scores.txt'

def get_time():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")


print('Script started:', get_time())
# Reads in dataset, columns only
data =  pd.read_csv(ML_FILE, sep='\t', index_col=0, nrows=0)
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

# Reads in whole dataset
data = pd.read_csv(ML_FILE, dtype=dtype_dict, sep='\t', index_col=0)

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

# Models' hyperparameters
grid = {
    MODEL_NAME + '__ccp_alpha': uniform(0, 0.001),
    MODEL_NAME + '__max_features': randint(50, 1200),
    MODEL_NAME + '__n_estimators': randint(50, 451),
    MODEL_NAME + '__max_depth': [10, 20, 50, 70, 100, 125, 150, 200, 500, None],
}

# Pipleline to perform full workflow from preprocessing to model fitting
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      (MODEL_NAME,
                       RandomForestClassifier(random_state=42))
])

# Create X and y data set
X = data.drop(columns=class_label)
y = data.loc[:, class_label]

# K_fold and containers for scores and feature importance
outer_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model_scores = []
fi_values = []

# Looping through the outer loop, feeding each training set into RSCV
# as the inner loop
for train_index,test_index in outer_kf.split(X, y):
    # Train sets and random search object
    X_train = X.iloc[train_index]
    y_train = y.iloc[train_index]
    X_test = X.iloc[test_index]
    y_test = y.iloc[test_index]
    hp_sr = RandomizedSearchCV(estimator=clf, param_distributions=grid,
                           n_iter=30, scoring='f1', cv=inner_kf, n_jobs=100)

    # RSCV is looping through the training data to find the best parameters
    # This is the inner loop
    hp_sr.fit(X_train, y_train)
    # The best hyper parameters from RSCV is now being tested on the unseen outer
    # loop test data
    y_pred = hp_sr.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    f1 = f1_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred)
    re = recall_score(y_test, y_pred) 

    # Appending score
    run = pd.Series([MODEL_NAME, tn, fp, fn, tp, f1, pre, re],
                index=['model_name', 'tn', 'fp', 'fn', 'tp', 'f1', 'precision',
                       'recall'])
    model_scores.append(run)
    # Appending feature importance
    fi = hp_sr.best_estimator_.named_steps[MODEL_NAME].feature_importances_
    fi_sort = pd.Series(fi, index=X.columns).sort_values(ascending=False)
    fi_values.append(fi_sort)

# Assembling results and saving it to file
df_scores = pd.concat(model_scores, axis=1).T
df_scores.index.name = 'id'
df_fi = pd.concat(fi_values, axis=1)
df_fi.index.name = 'feautres'
df_fi.insert(0, 'model_name', MODEL_NAME)

df_scores.to_csv(scores_file, sep='\t')
df_fi.to_csv(fi_file, sep='\t')
print('Script end:', get_time())

