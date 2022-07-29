# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 11:27:54 2019

@author: weixiong001
"""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import learning_curve
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from datetime import datetime

now = datetime.now()
current_time = now.strftime("%d/%m/%Y %H:%M:%S")
print("Script started:", current_time)
print() 
plt.ioff() # Turn off inline plotting

script_name = 'svm_loops'

df = pd.read_csv("proc_PNAS_data_ML.csv", sep="\t", index_col=0)
df_targets = df['Category']
df_features = df.drop(['Category'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(df_features, df_targets,
                                                    test_size=.1)
lst_params = []
lst_results = []
lst_scores = []
for i in range(100):
    print('Started iteration', i+1, current_time)   
    # Balancing
    df_train_combined = pd.concat([X_train, y_train], axis=1)
    
    df_0 = df_train_combined[df_train_combined['Category'] == 0]
    df_1 = df_train_combined[df_train_combined['Category'] == 1]
    
    #undersample
    sample_0 = df_0.sample(n=df_targets.value_counts().loc[1])
    df_train_balanced = pd.concat([sample_0,df_1], axis=0)
    df_train_balanced = shuffle(df_train_balanced)
    
    # X_train and y_train variables from train_test_split are now
    # reassigned to this
    X_train = df_train_balanced.drop(['Category'], axis=1)
    y_train = df_train_balanced['Category']
    
    # Scaling my features
    # Need to scale X_test as well, in the ML workshop, I did it later
    X_train_cont = X_train.loc[:, 'mean_exp':'OG_size']
    X_train_cat = X_train.loc[:, 'single_copy':'viridiplantae']
    sc = StandardScaler()
    sc.fit(X_train_cont)
    X_train_cont_scaled = sc.transform(X_train_cont)
    X_tcs_df = pd.DataFrame(data=X_train_cont_scaled, index=X_train_cont.index,
                            columns=X_train_cont.columns)
    X_train_scaled = pd.concat([X_tcs_df, X_train_cat], axis=1, sort=False)
    
    X_test_cont = X_test.loc[:, 'mean_exp':'OG_size']
    X_test_cat = X_test.loc[:, 'single_copy':'viridiplantae']
    X_test_cont_scaled = sc.transform(X_test_cont)
    X_testcs_df = pd.DataFrame(data=X_test_cont_scaled, index=X_test_cont.index,
                            columns=X_test_cont.columns)
    X_test_scaled = pd.concat([X_testcs_df, X_test_cat], axis=1, sort=False)
    print('Finished oversampling and scaling')
    print('Starting grid search', current_time)
    svm = SVC()    
    grid_param = {
            'kernel': ['linear', 'rbf'],
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'gamma': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
    }
    gd_sr = GridSearchCV(estimator=svm, param_grid=grid_param,
                         scoring='f1', cv=10, n_jobs=-1)
    gd_sr.fit(X_train_scaled, y_train)
    best_parameters = pd.Series(gd_sr.best_params_)
    # Appending gridsearch results to save it
    lst_params.append(best_parameters)
    # Creating model
    svm = SVC(C=best_parameters['C'], gamma=best_parameters['gamma'],
              kernel=best_parameters['kernel'])    
    print('Fitting model')
    svm.fit(X_train_scaled, y_train)
    y_hat = svm.predict(X_test_scaled)
    # Generating series to save model predictions
    iter_name = i+1
    ytest_series = pd.Series(y_test, name='y_test_' + str(iter_name))
    lst_results.append(ytest_series.reset_index(drop=True))
    yhat_series = pd.Series(y_hat, name='y_hat_' + str(iter_name))
    lst_results.append(yhat_series)
    prob_sgd = svm.decision_function(X_test_scaled)
    prob_series = pd.Series(prob_sgd, name='prob_' + str(iter_name))
    lst_results.append(prob_series)
    tn, fp, fn, tp = confusion_matrix(y_test, y_hat).ravel()
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    one_run = pd.Series([tn, fp, fn, tp, tpr, tnr],
                        index=['tn', 'fp', 'fn', 'tp', 'tpr', 'tnr'],
                        name='run_' + str(iter_name))
    lst_scores.append(one_run)
    print()

print('Printing dataframes', current_time)
df_params = pd.DataFrame(lst_params)
df_results = pd.DataFrame(lst_results)
df_scores = pd.concat(lst_scores, axis=1).T
df_params.to_csv('params_matrix.csv', sep='\t')
df_results.to_csv('params_results.csv', sep='\t')
df_scores.to_csv('all_scores.csv', sep='\t')
print('Script finished', current_time)
