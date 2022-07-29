# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 14:08:17 2020

@author: weixiong001

Run RF model with imblearn pipeline and other objects to perform the entire machine learning
workflow without data leakage. SMOTE used. SMOTE, ColumnTransformer and 
RandomForestClassifier have the n_jobs parameter.

Modified from ml_bcw_shp.py
"""
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score 
from sklearn.model_selection import learning_curve
from sklearn.metrics import plot_roc_curve
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline
from datetime import datetime
from matplotlib import pyplot as plt

# Input variable
data_file = sys.argv[1]  # ml data file
FT_FILE = 'feature_type.txt'
scores_file = data_file.split('.')[0] + '_scores.txt'

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

bool_array = data.columns.isin(all_cat_feat)
sm = SMOTENC(categorical_features=bool_array, random_state=42, n_jobs=20)

# Pipleline to perform full workflow from preprocessing to model fitting
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('smote', sm),
                      ('classifier',
                       RandomForestClassifier(ccp_alpha=0.0002,
                                              max_features=100,
                                              n_estimators=200,
                                              random_state=42,
                                              n_jobs=20))
                     ])

X = data.drop(columns=['class_label'])
y = data.loc[:, 'class_label']

model_scores = []

print('run', 'started:', get_time())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,
                                                    random_state=42,
                                                    stratify=y)
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

# Plotting learning curve
# Takes 50 min with 10 training subsets
train_size = np.linspace(.1, 1, 10)
sample_sizes, train_score, valid_score = learning_curve(clf, 
               X_train, y_train, train_sizes=train_size,
               verbose=1, cv=5, scoring='f1', n_jobs=-1)

train_mean = np.mean(train_score, axis=1)
train_std = np.std(train_score, axis=1)
valid_mean = np.mean(valid_score, axis=1)
valid_std = np.std(valid_score, axis=1)

plt.fill_between(sample_sizes, train_mean - train_std, 
                 train_mean + train_std, color='b', alpha=.1)
plt.plot(sample_sizes, train_mean, 'bo-', label='Training')
plt.fill_between(sample_sizes, valid_mean - valid_std, 
                 valid_mean + valid_std, color='r', alpha=.1)
plt.plot(sample_sizes, valid_mean, 'ro-', label='Validation')
plt.legend()
plt.savefig('lc_membrane_ohp.png')
plt.figure()

# Exploring models
fig, ax = plt.subplots()
curve = plot_roc_curve(clf, X_test, y_test, ax=ax)
auc_score = curve.roc_auc
ax.plot([0.0, 1.0], [0.0, 1.0], 'r-')
ax.set_xlabel('1-specificity/FPR')
ax.set_ylabel('sensitivity/TPR')
plt.grid()
plt.title('AUC: %.2f' %auc_score)
fig.savefig('auc_membrane_ohp.png')
plt.figure()

print('Script end:', get_time())
