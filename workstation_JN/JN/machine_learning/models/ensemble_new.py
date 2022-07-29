# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 11:27:54 2019

@author: weixiong001
"""

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from datetime import datetime
import warnings

# Turn off warnings
warnings.filterwarnings('ignore')


def get_time():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")


print("Script started:", get_time())
print()
plt.ioff()  # Turn off inline plotting

svm = SVC(probability=True)
rf = RandomForestClassifier()
dtc = DecisionTreeClassifier()
knn = KNeighborsClassifier()
mlp = MLPClassifier()

runs = 100

df = pd.read_csv("proc_PNAS_data_ML.csv", sep="\t", index_col=0)
# Need this for undersampling code
df_targets = df['Category']
SM_data = df[df['Category'] == 1]
GM_data = df[df['Category'] == 0]

# At the end, used to make ytest_df for saving to .csv
list_tests = []
# Models' hyperparameters
svm_grid = {
    'kernel': ['linear', 'rbf'],
    'C': [0.001, 0.01, 0.1, 1, 10],
    'gamma': [0.001, 0.01, 0.1, 1]
}
rf_grid = {
    'n_estimators': [50, 100, 300, 500, 800],
    'criterion': ['gini', 'entropy'],
    'max_features': [2, 3, 4]
}
dtc_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None],
        'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
        'max_features': [1, 2, 3, 4, 5, 6, 7]
}
knn_grid = {
        'n_neighbors': [1, 3, 5, 7, 9, 11],
        'p': [1, 2, 3, 4, 5],
        'weights': ['uniform', 'distance'],
        'leaf_size' : [3,10,30,60]
}
mlp_grid = {
        'hidden_layer_sizes': [(50,), (100,), (150,)],
        'solver': ['lbfgs', 'sgd', 'adam'],
        'activation' : ['identity', 'logistic', 'tanh', 'relu'],
        'alpha': [1e-5,1e-4,1e-3,1e-2,1e-1],
}

svm_output = {'params': [], 'results': [], 'scores': []}
rf_output = {'params': [], 'results': [], 'scores': []}
dtc_output = {'params': [], 'results': [], 'scores': []}
knn_output = {'params': [], 'results': [], 'scores': []}
mlp_output = {'params': [], 'results': [], 'scores': []}
# Container to hold stuff, for conversion to df for saving to .csv  
models = [(svm_grid, svm, svm_output, 'svm'), (rf_grid, rf, rf_output, 'rf'),
          (dtc_grid, dtc, dtc_output, 'dtc'), (knn_grid, knn, knn_output, 'knn'),
          (mlp_grid, mlp, mlp_output, 'mlp')]


for i in range(runs):
    SM_test = SM_data.sample(40)
    GM_test = GM_data.sample(40)
    train_df = df.drop(SM_test.index)
    train_df = train_df.drop(GM_test.index)
    
    y_train = train_df['Category']
    X_train = train_df.drop(['Category'], axis=1)
    
    test_concat = pd.concat([SM_test, GM_test])
    y_test = test_concat['Category']
    X_test = test_concat.drop(['Category'], axis=1)
    # Needed at the end
    ytest_series = pd.Series(y_test, name='y_test')
    list_tests.append(ytest_series)

    print('Started iteration', i+1, get_time())
    # Balancing
    df_train_combined = pd.concat([X_train, y_train], axis=1)

    df_0 = df_train_combined[df_train_combined['Category'] == 0]
    df_1 = df_train_combined[df_train_combined['Category'] == 1]

    # undersample
    sample_0 = df_0.sample(n=df_targets.value_counts().loc[1])
    df_train_balanced = pd.concat([sample_0, df_1], axis=0)
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
    X_testcs_df = pd.DataFrame(data=X_test_cont_scaled,
                               index=X_test_cont.index,
                               columns=X_test_cont.columns)
    X_test_scaled = pd.concat([X_testcs_df, X_test_cat], axis=1, sort=False)

    print('Finished oversampling and scaling')
    print('Starting grid search', get_time())
    for model in models:
        gd_sr = GridSearchCV(estimator=model[1], param_grid=model[0],
                             scoring='f1', cv=10, n_jobs=-1)
        gd_sr.fit(X_train_scaled, y_train)
        # Getting all output
        best_parameters = pd.Series(gd_sr.best_params_)
        y_hat = gd_sr.predict(X_test_scaled)
        y_hat_series = pd.Series(y_hat, name='y_hat')
        prob_sgd = gd_sr.predict_proba(X_test_scaled)[:, 1]
        prob_series = pd.Series(prob_sgd, name='prob')
        tn, fp, fn, tp = confusion_matrix(y_test, y_hat).ravel()
        tpr = tp / (tp + fn)
        tnr = tn / (tn + fp)
        one_run = pd.Series([tn, fp, fn, tp, tpr, tnr],
                            index=['tn', 'fp', 'fn', 'tp', 'tpr', 'tnr'],
                            name='run')
        # Appending output to save it
        model[2]['params'].append(best_parameters)

        model[2]['results'].append(y_hat_series)
        model[2]['results'].append(prob_series)
        model[2]['scores'].append(one_run)
    print()

print('Printing dataframes', get_time())
for model in models:
    df_params = pd.DataFrame(model[2]['params'])
    df_results = pd.DataFrame(model[2]['results'])
    df_scores = pd.concat(model[2]['scores'], axis=1).T
    df_params.to_csv(model[3] + '_params_matrix.csv', sep='\t')
    df_results.to_csv(model[3] + '_params_results.csv', sep='\t')
    df_scores.to_csv(model[3] + '_all_scores.csv', sep='\t')
ytests_df = pd.concat(list_tests, axis=1)
ytests_df.to_csv('y_test.csv', sep='\t')
print('Script finished', get_time())
