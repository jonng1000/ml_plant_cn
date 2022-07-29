# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 16:28:29 2021

@author: weixiong001

Need to run on CN as file takes up too much RAM
Boxplot of mutual rank distribution
"""

import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd

FILE = 'mutual_ranks.txt'
FIG = 'fi_mr_box.png'

# Cannot do the below due to some weird warnig
# FutureWarning: elementwise comparison failed; returning scalar instead,
# but in the future will perform
# elementwise comparison
# Something to do with a numpy and pandas clash, use the below workaround
#df = pd.read_csv(FILE, sep='\t', index_col=0)
df = pd.read_csv(FILE, sep='\t')
df.set_index(['id'], inplace=True)

g = sns.boxplot(data=df, x='MR', orient='v')
g.figure.savefig(FIG)
plt.close()

#sorted_df = df.sort_values(by=['MR'])
