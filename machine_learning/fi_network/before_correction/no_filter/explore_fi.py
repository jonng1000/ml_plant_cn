# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 16:28:29 2021

@author: weixiong001

Takes about a few minutues
Plots histogram of feature importance
"""

from datetime import datetime
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

FILE = 'big_fi.txt'
FIG = 'big_fi.png'

def get_time():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

df = pd.read_csv(FILE, sep='\t', index_col=0)
stacked = df.stack()
all_values = stacked.reset_index()
all_values.rename(columns={0: 'fi'}, inplace=True)

g = sns.histplot(data=all_values, x='fi', bins=16, log_scale=(False, True))
g.figure.savefig(FIG)
plt.close()

'''
>>> all_values
                features                      level_1        fi
0          go_GO:0022414                go_GO:0000003  0.039806
1          go_GO:0022414                go_GO:0000030  0.000000
2          go_GO:0022414                go_GO:0000038  0.000000
3          go_GO:0022414                go_GO:0000041  0.000000
4          go_GO:0022414                go_GO:0000096  0.000000
...                  ...                          ...       ...
112497856  go_GO:0000003                  pfa_PF01963  0.000000
112497857  go_GO:0000003                  pfa_PF01965  0.000000
112497858  go_GO:0000003                  pfa_PF01966  0.000000
112497859  go_GO:0000003  cin_DRE-like promoter motif  0.000000
112497860  go_GO:0000003                       pep_mw  0.000124

[112497861 rows x 3 columns]
>>> all_values['fi'].value_counts()
0.000000    111324571
0.020000          468
0.020408          392
0.020833          320
0.021277          280
              ...
0.001667            1
0.000061            1
0.000149            1
0.038780            1
0.000730            1
Name: fi, Length: 1165793, dtype: int64

# Majority of my feature importance valus are 0
>>> 111324571 / 112497861 * 100
98.95705572570841
'''
