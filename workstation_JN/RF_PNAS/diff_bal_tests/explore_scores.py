# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 15:54:09 2019

@author: weixiong001

Plots scores
"""

import pandas as pd
import math
import seaborn as sns
from matplotlib import pyplot as plt

df = pd.read_csv("rf_scores_smote.txt", sep="\t", index_col=0)
# Original param is this, but removed it as its
# too big figsize=(20, 15)
fig1, ax1 = plt.subplots()
ax1.boxplot([df['f1'], df['precision'], df['recall']])
ax1.set_xticklabels(['f1', 'precision', 'recall'])
plt.savefig('boxplot_rf_smote.png')
