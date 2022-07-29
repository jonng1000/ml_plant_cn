# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 16:28:29 2021

@author: weixiong001
Exploring drawing line plot for all mutual ranks
"""

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd

FILE = 'mutual_ranks.txt'
FIG = 'fi_mr_line.png'

# Cannot do the below due to some weird warnig
# FutureWarning: elementwise comparison failed; returning scalar instead,
# but in the future will perform
# elementwise comparison
# Something to do with a numpy and pandas clash, use the below workaround
#df = pd.read_csv(FILE, sep='\t', index_col=0)
df = pd.read_csv(FILE, sep='\t')
df.set_index(['id'], inplace=True)
df.sort_values(by=['MR'], inplace=True)
df['placeholder'] = np.arange(len(df))
g = sns.lineplot(data=df, x='placeholder', y='MR', sort=False,
                 ci=None, marker='o', linestyle='')
g.set_xticklabels([])
g.xaxis.set_tick_params(bottom=False)
g.figure.savefig(FIG)
plt.close()
