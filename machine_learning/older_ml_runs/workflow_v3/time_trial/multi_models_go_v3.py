# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 14:08:17 2020

@author: weixiong001

Runs multiple models with pipeline and other objects to perform the 
entire machine learning workflow without data leakage. 
Several things use n_jobs parameter, main one is random search

Customised for go terms. Modified from multi_models_go_v2.py in
~/machine_learning/workflow_v3
"""
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from scipy.stats import loguniform
from scipy.stats import randint
from sklearn.ensemble import AdaBoostClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score 
from datetime import datetime
import pickle

# Input variables/contants
FT_FILE = 'feature_type.txt'
ML_FILE = sys.argv[1]
# Added this code, as this script is for go term prediction
ML_FILE = ML_FILE.replace('go_GO:', 'go_GO_')
class_label = sys.argv[1]
fi_file = class_label + '_fi.txt'
scores_file = class_label + '_scores.txt'

def get_time():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")


print('Script started:', get_time())
# Reads in dataset, columns only
infile = open(ML_FILE,'rb')
pickle_df = pickle.load(infile)
infile.close()
# This is changed due to above code to make it compatible for GO terms
data =  pickle_df
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

# Not used here, as this script is for go term prediction
# Reads in whole dataset
#data = pd.read_csv(ML_FILE, dtype=dtype_dict, sep='\t', index_col=0)
# Added this code, as this script is for go term prediction
temp_cont = data[all_cont_feat].apply(pd.to_numeric, downcast='float')
temp_cat = data[all_cat_feat].apply(pd.to_numeric, downcast='integer')
data = pd.concat([temp_cont, temp_cat], axis=1)

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

# Models
ada = AdaBoostClassifier(random_state=42)
brf = BalancedRandomForestClassifier(random_state=42)
rf = RandomForestClassifier(random_state=42, class_weight='balanced')

# Models' hyperparameters
ada_grid = {
    'ada__n_estimators': randint(100, 151),
    'ada__learning_rate': uniform(0.6, 0.8),
}

brf_grid = {
    'brf__max_features': randint(50, 1200),
    'brf__n_estimators': randint(50, 451),
    'brf__max_depth': [10, 20, 50, 70, 100, 125, 150, 200, 500, None],
}

rf_grid = {
    'rf__ccp_alpha': uniform(0, 0.001),
    'rf__max_features': randint(50, 1200),
    'rf__n_estimators': randint(50, 451),
    'rf__max_depth': [10, 20, 50, 70, 100, 125, 150, 200, 500, None],
}

# Container to hold model output
ada_output = {'fi': [], 'scores': []}
brf_output = {'fi': [], 'scores': []}
rf_output = {'fi': [], 'scores': []}
# Container to hold all info for conversion to df for saving to .csv  
'''
models = [(rf_grid, rf, rf_output, 'rf'),
          (ada_grid, ada, ada_output, 'ada'),
          (brf_grid, brf, brf_output, 'brf')]
'''
models = [(brf_grid, brf, brf_output, 'brf'),
          (ada_grid, ada, ada_output, 'ada'),
          (rf_grid, rf, rf_output, 'rf')]

model_scores = []
fi_values = []
print(class_label, 'tested')
for model in models:
    print(model[3], 'started:', get_time())
    # Pipleline to perform full workflow from preprocessing to model fitting
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          (model[3], model[1])
    ])

    # Create X and y data set
    X = data.drop(columns=class_label)
    y = data.loc[:, class_label]

    # K_fold objects
    outer_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Looping through the outer loop, feeding each training set into RSCV
    # as the inner loop
    for train_index,test_index in outer_kf.split(X, y):
        # Train sets and random search object
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]
        X_test = X.iloc[test_index]
        y_test = y.iloc[test_index]
        hp_sr = RandomizedSearchCV(estimator=clf, param_distributions=model[0],
                                   n_iter=30, scoring='f1', cv=inner_kf,
                                   n_jobs=100)

        # RSCV is looping through the training data to find the best parameters
        # This is the inner loop
        hp_sr.fit(X_train, y_train)
        # The best hyper parameters from RSCV is now being tested on the
        # unseen outer
        # loop test data
        y_pred = hp_sr.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        f1 = f1_score(y_test, y_pred)
        pre = precision_score(y_test, y_pred)
        re = recall_score(y_test, y_pred) 

        # Appending score
        run = pd.Series([model[3], tn, fp, fn, tp, f1, pre, re],
                        index=['model_name', 'tn', 'fp', 'fn', 'tp', 'f1',
                               'precision', 'recall'])
        model_scores.append(run)
        # Appending feature importance
        if model[3] in ['ada', 'brf', 'rf']:
            fi = hp_sr.best_estimator_.named_steps[model[3]].feature_importances_
        # In v3 of script, dont need this ellif statement, but just left it in
        elif model[3] in ['logr', 'lsv']:
            fi = hp_sr.best_estimator_.named_steps[model[3]].coef_
            fi = fi[0]
        fi_sort = pd.Series(fi, index=X.columns).sort_values(ascending=False)
        fi_sort.name = model[3]
        fi_values.append(fi_sort)

# Assembling results and saving it to file
df_scores = pd.concat(model_scores, axis=1).T
df_scores.index.name = 'id'
df_fi = pd.concat(fi_values, axis=1)
df_fi.index.name = 'features'
df_scores.to_csv(scores_file, sep='\t')
df_fi.to_csv(fi_file, sep='\t')
print('Script end:', get_time())
