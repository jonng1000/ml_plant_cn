# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 14:08:17 2020

@author: weixiong001

Run RF model many times and extract the most important features.
No random features.
Edited from rf_feature_extract_v3.py
"""
import sys
import pandas as pd
from sklearn.inspection import permutation_importance
# Can comment out these two lines if running on workstation
# Needed to ensure below module is imported correctly
# path is different for linux
PATH = '/mnt/d/GoogleDrive/machine_learning/RF_PNAS/feature_extract'
sys.path.insert(1, PATH + '/build_top_features')
import rfe_module_v2 as rfm

# For testing
# file = 'membrane_GO.txt'
# Input variables
file = sys.argv[1]  # ml data file
scores_file = file.split('_GO')[0] + '_scores.txt'
feat_impt_file = file.split('_GO')[0] + '_feat.txt'
perm_file = file.split('_GO')[0] + '_perm.txt'
#print(sys.argv)
#print(file, scores_file, feat_impt_file)
runs = 10
CLASS_LABELS = 'AraCyc annotation'

# Reading in data and dividing into classes
print("Script started:", rfm.get_time())
print()
df = rfm.read_df(file)
pos = df[CLASS_LABELS].value_counts().idxmin()
neg = df[CLASS_LABELS].value_counts().idxmax()
print(df[CLASS_LABELS].value_counts())
print('positive class:', pos)
print('negative class:', neg)
df.loc[:, 'AraCyc annotation'].replace([pos, neg], [1, 0], inplace=True)

SM_data, GM_data = rfm.sep_df(df, CLASS_LABELS)
test_size = int(len(SM_data)/10)
print()

model_scores = []
feat_impt = []

for i in range(runs):
    print('Started iteration', i+1, rfm.get_time())
    SM_test, GM_test, train_df = rfm.split_test_train(SM_data, GM_data, test_size, df)
    y_train, X_train, y_test, X_test = rfm.sep_feat_labels(train_df,
                                                       CLASS_LABELS,
                                                       SM_test, GM_test)
    # Balance via undersampling majority class
    balance_train = rfm.balancing(X_train, y_train, CLASS_LABELS)
    # X_train and y_train variables from train_test_split are now
    # reassigned to this
    X_train = balance_train.drop([CLASS_LABELS], axis=1)
    y_train = balance_train[CLASS_LABELS]
    
    scaling_obj, X_train_scaled = rfm.scales_continous(X_train)
    X_test_scaled = rfm.scales_test(scaling_obj, X_test)
    rf_model, y_hat = rfm.random_forest(X_train_scaled, y_train, X_test_scaled)
    one_run = rfm.scores(y_test, y_hat)
    model_scores.append(one_run)
    
    fi_sort = pd.DataFrame(rf_model.feature_importances_, 
                           index=X_train.columns,
                           columns=['importance']).sort_values('importance',
                                                               ascending=False)
    feat_impt.append(fi_sort)
    
    result = permutation_importance(rf_model, X_test_scaled, y_test, n_jobs=-1)
    sorted_idx = result.importances_mean.argsort()
    array_data = result.importances[sorted_idx].T
    labels = X_test.columns[sorted_idx]
    perm_df = pd.DataFrame(array_data, columns=labels)
    perm_df.index.name = 'runs'
    perm_df.to_csv(perm_file, sep='\t')
    
df_scores = pd.concat(model_scores, axis=1).T
df_scores.reset_index(drop=True, inplace=True)
df_scores.index.name = 'runs'
df_scores.to_csv(scores_file, sep='\t')

df_feat_i = pd.concat(feat_impt, axis=1)
df_feat_i.columns = ['impt' + str(i) for i in range(runs)]
df_feat_i.index.name = 'features'
df_feat_i.to_csv(feat_impt_file, sep='\t')
print('Script finished', rfm.get_time())

