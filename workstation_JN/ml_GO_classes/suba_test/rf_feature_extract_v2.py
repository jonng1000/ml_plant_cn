# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 14:08:17 2020

@author: weixiong001

Run RF model many times and extract the most important features.
No random features.
Edited from rf_feature_extract.py
"""
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score
from datetime import datetime

# For testing
# file = 'membrane_GO.txt'
# Input variables
file = sys.argv[1]  # ml data file
scores_file = file.split('_GO')[0] + '_scores.txt'
feat_impt_file = file.split('_GO')[0] + '_feat.txt'
#print(sys.argv)
#print(file, scores_file, feat_impt_file)
runs = 100
class_labels = 'AraCyc annotation'

def get_time():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

def read_df(data_file):
    '''
    Reads in dataframe from a tab-delimited .txt file
    '''
    data = pd.read_csv(data_file, sep='\t', index_col=0)
    return data

def sep_df(data_frame):
    '''
    Separates dataframe into positive class dataframe, and negative class
    dataframe
    '''
    # Based on LabelEncoder from fully_proc.py
    targets = data_frame[class_labels]
    data1 = data_frame[targets == 1]
    data0 = data_frame[targets == 0]
    return data1, data0

def split_test_train(pos_class, neg_class, sample_size, data_frame):
    '''
    Splits dataframe into positive and negative class for test set
    respectively, and remove them from the train set
    '''
    pos_test = pos_class.sample(sample_size)
    neg_test = neg_class.sample(sample_size)
    train = data_frame.drop(pos_test.index)
    train = train.drop(neg_test.index)
    return pos_test, neg_test, train

def sep_feat_labels(train, class_labels, pos_test, neg_test):
    '''
    Takes in training dataframe, class labels, positive and negative test
    sets, creates training class labels, training features, and test set.
    '''
    train_labels = train[class_labels]
    train_features = train.drop([class_labels], axis=1)
    test_concat = pd.concat([pos_test, neg_test])
    test_labels = test_concat[class_labels]
    test_features = test_concat.drop([class_labels], axis=1)
    return train_labels, train_features, test_labels, test_features

def balancing(train_features, train_labels):
    '''
    Combines training features and labels, and undersamples majority class to
    get balanced datasets
    '''
    df_train_combined = pd.concat([train_features, train_labels], axis=1)
    df_0 = df_train_combined[df_train_combined[class_labels] == 0]
    df_1 = df_train_combined[df_train_combined[class_labels] == 1]
    # Assumes minority class is the desired class, and is the second item
    # in the array
    sample_0 = df_0.sample(n=df_train_combined[class_labels].value_counts().loc[1])
    df_train_balanced = pd.concat([sample_0, df_1], axis=0)
    df_train_balanced = shuffle(df_train_balanced)
    return df_train_balanced

def scales_continous(train_features):
    '''
    Takes in training features, and returns a standardscaling object, and
    scaled training features
    '''
    sc = StandardScaler()
    float_features = train_features.select_dtypes(include='float64')
    int_features = train_features.select_dtypes(include='int64')
    sc.fit(float_features)
    train_scaled = sc.transform(float_features)
    ts_df = pd.DataFrame(data=train_scaled, index=float_features.index,
                         columns=float_features.columns)
    train_scaled = pd.concat([ts_df, int_features], axis=1, sort=False)
    return sc, train_scaled

def scales_test(sc_obj, test_features):
    '''
    Takes in standardscaling object, and test features, and scales them.
    Returns scaled test features
    '''
    float_features = test_features.select_dtypes(include='float64')
    int_features = test_features.select_dtypes(include='int64')
    test_scaled = sc_obj.transform(float_features)
    ts_df = pd.DataFrame(data=test_scaled, index=float_features.index,
                           columns=float_features.columns)
    test_scaled = pd.concat([ts_df, int_features], axis=1, sort=False)
    return test_scaled

def random_forest(train_scaled, labels, test_scaled):
    '''
    Fits RF model to scaled train set, and tests it against scaled test set.
    Returns RF model and predictions. Modified from original function in 
    rf_feature_extract_beta.py
    '''
    rf = RandomForestClassifier(n_jobs=-1)
    rf.fit(train_scaled, labels)
    predictions = rf.predict(test_scaled)
    return rf, predictions

def scores(true_labels, predictions):
    '''
    Takes in class labels from the test set, and scores model predictions.
    Return a row of tn, fp, fn, tp, f1, precision and recall scores
    '''
    tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
    f1 = f1_score(true_labels, predictions)
    pre = precision_score(true_labels, predictions)
    re = recall_score(true_labels, predictions) 
    run = pd.Series([tn, fp, fn, tp, f1, pre, re],
                    index=['tn', 'fp', 'fn', 'tp', 'f1', 'precision',
                           'recall'], name='run')
    return run

# Reading in data and dividing into classes
print("Script started:", get_time())
print()
df = read_df(file)
pos = df[class_labels].value_counts().idxmin()
neg = df[class_labels].value_counts().idxmax()
print(df[class_labels].value_counts())
print('positive class:', pos)
print('negative class:', neg)
df.loc[:, 'AraCyc annotation'].replace([pos, neg], [1, 0], inplace=True)

SM_data, GM_data = sep_df(df)
test_size = int(len(SM_data)/10)
print()

model_scores = []
feat_impt = []

for i in range(runs):
    print('Started iteration', i+1, get_time())
    SM_test, GM_test, train_df = split_test_train(SM_data, GM_data, test_size, df)
    y_train, X_train, y_test, X_test = sep_feat_labels(train_df,
                                                       class_labels,
                                                       SM_test, GM_test)
    # Balance via undersampling majority class
    balance_train = balancing(X_train, y_train)
    # X_train and y_train variables from train_test_split are now
    # reassigned to this
    X_train = balance_train.drop([class_labels], axis=1)
    y_train = balance_train[class_labels]
    
    scaling_obj, X_train_scaled = scales_continous(X_train)
    X_test_scaled = scales_test(scaling_obj, X_test)
    rf_model, y_hat = random_forest(X_train_scaled, y_train, X_test_scaled)
    one_run = scores(y_test, y_hat)
    model_scores.append(one_run)
    
    fi_sort = pd.DataFrame(rf_model.feature_importances_, 
                           index=X_train.columns,
                           columns=['importance']).sort_values('importance',
                                                               ascending=False)
    feat_impt.append(fi_sort)
    
df_scores = pd.concat(model_scores, axis=1).T
df_scores.reset_index(drop=True, inplace=True)
df_scores.index.name = 'runs'
df_scores.to_csv(scores_file, sep='\t')

df_feat_i = pd.concat(feat_impt, axis=1)
df_feat_i.columns = ['impt' + str(i) for i in range(runs)]
df_feat_i.index.name = 'features'
df_feat_i.to_csv(feat_impt_file, sep='\t')
print('Script finished', get_time())

