# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 16:28:29 2021

@author: weixiong001

Takes about a few minutues
"""

from datetime import datetime
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

FILE = 'nonzero_mr.txt'
FIG = 'nonzero_mr.png'

def get_time():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")

df = pd.read_csv(FILE, sep='\t', index_col=0)

g = sns.histplot(data=df, x='MR', bins=16, log_scale=(False, True))
g.figure.savefig(FIG)
plt.close()
