# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 14:08:17 2020

@author: weixiong001

Runs random forest with pipeline and other objects to perform the 
entire machine learning workflow without data leakage. 
Several things use n_jobs parameter

Customised for go terms, use selected hps and a specific number of
cores
Pickles saved model to a file

Modified from rf_go.py in
~/machine_learning/ml_go_workflow/go_runs
"""
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score 
from datetime import datetime
import pickle
from joblib import dump

# Input variables/contants
FT_FILE = 'feature_type.txt'
ML_FILE = sys.argv[1]
OUTPUT_FOLDER = 'output/'
CORES = 4  # Num of cores for rf
# Added this code, as this script is for go term prediction
ML_FILE = ML_FILE.replace('go_GO:', 'go_GO_')
class_label = sys.argv[1]
temp_class_label = class_label.replace('go_GO:', 'go_GO_')
fi_file = OUTPUT_FOLDER + temp_class_label + '_fi.txt'
scores_file = OUTPUT_FOLDER + temp_class_label + '_scores.txt'
model_file = OUTPUT_FOLDER + temp_class_label + '_model'

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

# Create model
rf = RandomForestClassifier(ccp_alpha=0.001, max_features=0.2,
                            n_estimators=50, max_depth=200,
                            random_state=42, oob_score=True,
                            n_jobs=CORES, class_weight='balanced')
model = ['rf', rf]

# Container to hold scores and feature importance
model_scores = []
fi_values = []

print(class_label, 'tested', get_time())
time_start = get_time()
# Pipleline to perform full workflow from preprocessing to model fitting
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      (model[0], model[1])
])

# Create X and y data set
X = data.drop(columns=class_label)
y = data.loc[:, class_label]

# With random shuffling of features
for i in range(1, 101):
    X = X.apply(lambda x: x.sample(frac=1).values)

    # Training
    clf.fit(X, y)

    # Appending score
    y_pred = np.argmax(clf[model[0]].oob_decision_function_, axis=1)
    f1 = f1_score(y, y_pred)
    pre = precision_score(y, y_pred)
    re = recall_score(y, y_pred)
    # Put this here as its needed directly below
    time_end = get_time()
    # Appending scores and end times
    run = pd.Series([model[0], i, clf[model[0]].oob_score_,
                     f1, pre, re, time_start, time_end],
                    index=['model_name', 'rs iteration', 'oob_accuracy',
                           'oob_f1', 'oob_precision', 'oob_recall',
                           'time_start', 'time_end'])
    model_scores.append(run)
    # Appending feature importance
    fi = clf[model[0]].feature_importances_
    fi_sort = pd.Series(fi, index=X.columns).sort_values(ascending=False)
    fi_sort.name = model[0]
    fi_values.append(fi_sort)
    
# Assembling results and saving it to file
df_scores = pd.concat(model_scores, axis=1).T
df_scores.index.name = 'id'
df_fi = pd.concat(fi_values, axis=1)
df_fi.index.name = 'features'
df_scores.to_csv(scores_file, sep='\t')
df_fi.to_csv(fi_file, sep='\t')
dump(clf, model_file) 
print('Script end:', get_time())
