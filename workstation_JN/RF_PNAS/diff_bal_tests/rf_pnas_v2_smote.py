# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 14:08:17 2020

@author: weixiong001

Runs RF model with default hyperparameters.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTENC
from datetime import datetime

file = './fully_processed.txt'
runs = 100
class_labels = 'AraCyc annotation'
scores_file = 'rf_scores_smote.txt'
test_fraction = 0.1

def get_time():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

print("Script started:", get_time())
print()

df = pd.read_csv(file, sep='\t', index_col=0)

# df.dtypes.astype(str).value_counts()
# Out[3]: 
# int64      9478
# float64      60
# dtype: int64

# df.isna().sum().sum()
# Out[4]: 0

# Based on LabelEncoder from fully_proc.py
df_targets = df[class_labels]
SM_data = df[df[class_labels] == 1]
GM_data = df[df[class_labels] == 0]
test_size = int(len(SM_data) * test_fraction)

model_scores = []
for i in range(runs):
    SM_test = SM_data.sample(test_size)
    GM_test = GM_data.sample(test_size)
    train_df = df.drop(SM_test.index)
    train_df = train_df.drop(GM_test.index)
    
    y_train = train_df[class_labels]
    X_train = train_df.drop([class_labels], axis=1)
    test_concat = pd.concat([SM_test, GM_test])
    y_test = test_concat[class_labels]
    X_test = test_concat.drop([class_labels], axis=1)
    
    print('Started iteration', i+1, get_time())

    cat = X_train.select_dtypes(include='int64').columns
    mask = X_train.columns.isin(cat)
    sm = SMOTENC(categorical_features=mask, n_jobs=12)
    
    # Balancing
    X_bal, y_bal = sm.fit_resample(X_train, y_train)
    df_train_balanced = pd.concat([X_bal, y_bal], axis=1)
    df_train_balanced = shuffle(df_train_balanced)

    # X_train and y_train variables from train_test_split are now
    # reassigned to this
    X_train = df_train_balanced.drop([class_labels], axis=1)
    y_train = df_train_balanced[class_labels]

    sc = StandardScaler()
    X_train_con = X_train.select_dtypes(include='float64')
    X_train_int = X_train.select_dtypes(include='int64')
    sc.fit(X_train_con)
    X_train_scaled = sc.transform(X_train_con)
    X_tcs_df = pd.DataFrame(data=X_train_scaled, index=X_train_con.index,
                            columns=X_train_con.columns)
    X_train_scaled = pd.concat([X_tcs_df, X_train_int], axis=1, sort=False)
    
    X_test_con = X_test.select_dtypes(include='float64')
    X_test_int = X_test.select_dtypes(include='int64')
    X_test_scaled = sc.transform(X_test_con)
    X_testcs_df = pd.DataFrame(data=X_test_scaled, index=X_test_con.index,
                               columns=X_test_con.columns)
    X_test_scaled = pd.concat([X_testcs_df, X_test_int], axis=1, sort=False)

    print('Finished scaling')
    print('Starting RF model', get_time())
    
    rf = RandomForestClassifier()
    rf.fit(X_train_scaled, y_train)
    y_hat = rf.predict(X_test_scaled)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_hat).ravel()
    f1 = f1_score(y_test, y_hat)
    pre = precision_score(y_test, y_hat)
    re = recall_score(y_test, y_hat) 
    one_run = pd.Series([tn, fp, fn, tp, f1, pre, re],
                        index=['tn', 'fp', 'fn', 'tp', 'f1', 'precision',
                               'recall'],
                        name='run')
    model_scores.append(one_run)
    
df_scores = pd.concat(model_scores, axis=1).T
df_scores.reset_index(drop=True, inplace=True)
df_scores.index.name = 'runs'
df_scores.to_csv(scores_file, sep='\t')
print('Script finished', get_time())
