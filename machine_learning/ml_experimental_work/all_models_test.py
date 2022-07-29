# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 14:08:17 2020

@author: weixiong001

Runs multiple models with pipeline and other objects to train the 
machine learning models without data leakage. 
Several things use n_jobs parameter

Customised for go terms.
Modified from tt_multi_models.py in
~/machine_learning/ml_go_workflow//time_repeat
"""
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
import pickle

# Input variables/contants
FT_FILE = 'feature_type.txt'
ML_FILE = sys.argv[1]
# Added this code, as this script is for go term prediction
ML_FILE = ML_FILE.replace('go_GO:', 'go_GO_')
class_label = sys.argv[1]
temp_class_label = class_label.replace('go_GO:', 'go_GO_')
scores_file = temp_class_label + '_scores.txt'

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
rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=120)
logr = LogisticRegression(solver='saga', n_jobs=120,
                          random_state=42, class_weight='balanced')
lsv = LinearSVC(random_state=42, class_weight='balanced')
gbc = GradientBoostingClassifier(random_state=42)
xgc = xgb.XGBClassifier(random_state=42, n_jobs=120)
import xgboost as xgb


# Container to hold all info for conversion to df for saving to .csv  
models = [(ada, 'ada'),
          (rf, 'rf'),
          (logr, 'logr'),
          (lsv, 'lsv'),
          (gbc, 'gbc'),
          (xgc, 'xgc')]
# This stores the start and end times of the model
# Named model_scores as its based on an earlier script
model_scores = []

for model in models:
    print(model[1], 'started:', get_time())
    time_start = get_time()
    # Pipleline to perform full workflow from preprocessing to model fitting
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          (model[1], model[0])
    ])

    # Create X and y data set
    X = data.drop(columns=class_label)
    y = data.loc[:, class_label]

    # K_fold objects
    inner_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Assign X and y to train sets, so that I can modify less things
    # further down the script
    X_train = X
    y_train = y
    
    scores = cross_val_score(clf, X, y, cv=inner_kf, scoring='f1')
    time_end = get_time()
    # Appending start and end times
    temp_df = pd.DataFrame({'run':range(1,6),'score':scores,
                            'model_name':model[1], 'time_start':time_start,
                            'time_end':time_end})
    model_scores.append(temp_df)

# Assembling results and saving it to file
# Named df_scores as its based on an earlier script
df_scores = pd.concat(model_scores, axis=0)
df_scores.index.name = 'id'
df_scores.to_csv(scores_file, sep='\t')
print('Script end:', get_time())
