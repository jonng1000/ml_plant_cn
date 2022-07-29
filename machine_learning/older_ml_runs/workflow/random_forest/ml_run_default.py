# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 14:08:17 2020

@author: weixiong001

Run RF model with pipeline and other objects to perform the entire machine learning
workflow without data leakage. This workflow is ran once just to test, and prints
out the classification report. Only ColumnTransformer and RandomForestClassifier
have the n_jobs parameter.

Uses default hyperparameters
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

# Pipleline to perform full workflow from preprocessing to model fitting
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier',
                       RandomForestClassifier(random_state=42))
                     ])

X = data.drop(columns=['class_label'])
y = data.loc[:, 'class_label']

model_scores = []
for one_run in range(RUNS):
    print('run', one_run+1, 'started:', get_time())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                        random_state=42,
                                                        stratify=y)
    df_train = pd.concat([X_train, y_train], axis=1)
    df_0 = df_train[df_train['class_label'] == 0]
    df_1 = df_train[df_train['class_label'] == 1]

    # Calculates sample size of minority class, for downsampling of
    # majority class
    if len(df_1) < len(df_0):
        sample_num = len(df_1)
        sample_0 = df_0.sample(n=sample_num, random_state=42)
        df_train_balanced = pd.concat([sample_0, df_1], axis=0)
    elif len(df_1) > len(df_0):
        sample_num = len(df_0)
        sample_1 = df_1.sample(n=sample_num, random_state=42)
        df_train_balanced = pd.concat([sample_1, df_0], axis=0)
    else:
        print('Classes are equal!')
    
    df_train_balanced = shuffle(df_train_balanced, random_state=42)
    X_train = df_train_balanced.drop(columns=['class_label'])
    y_train = df_train_balanced.loc[:, 'class_label']

    clf.fit(X_train, y_train)

    # Getting all output
    y_pred = clf.predict(X_test)
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