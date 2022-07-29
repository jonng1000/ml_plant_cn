# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 16:28:29 2021

@author: weixiong001

Need to run on CN if input file takes up too much RAM
Explores mutual rank (MR) distributiom for different feature categories
"""

import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from datetime import datetime

FILE = '../mutual_ranks.txt'

def get_time():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")


print('Script started:', get_time())
# Cannot do the below due to some weird warnig
# FutureWarning: elementwise comparison failed; returning scalar instead,
# but in the future will perform
# elementwise comparison
# Something to do with a numpy and pandas clash, use the below workaround
#df = pd.read_csv(FILE, sep='\t', index_col=0)
df = pd.read_csv(FILE, sep='\t')
df.set_index(['id'], inplace=True)

go = ('go_')
expr = ('spm_', 'tpm_', 'dge_', 'dia_', 'dit_')
prot_dom = ('mob_', 'pfa_', 'tmh_', 'num_')
networks = ('ppi_', 'pid_', 'coe_', 'cid_', 'ttr_', 'tti_', 'ttf_', 'agn_', 'agi_')
others = ('ort_', 'phy_', 'sin_', 'tan_', 'pep_', 'cin_', 'cif_', 'gwa_', 
          'twa_', 'hom_', 'ntd_', 'gbm_', 'con_', 'ptm_')

df.loc[df['f1'].str.startswith(go), 'f1'] = 'GO'
df.loc[df['f1'].str.startswith(expr), 'f1'] = 'Expression'
df.loc[df['f1'].str.startswith(prot_dom), 'f1'] = 'Protein Domain'
df.loc[df['f1'].str.startswith(networks), 'f1'] = 'Networks'
df.loc[df['f1'].str.startswith(others), 'f1'] = 'Others'

df.loc[df['f2'].str.startswith(go), 'f2'] = 'GO'
df.loc[df['f2'].str.startswith(expr), 'f2'] = 'Expression'
df.loc[df['f2'].str.startswith(prot_dom), 'f2'] = 'Protein Domain'
df.loc[df['f2'].str.startswith(networks), 'f2'] = 'Networks'
df.loc[df['f2'].str.startswith(others), 'f2'] = 'Others'

def create_bool(bool1, bool2, dataf):
    if bool1 == bool2:
        bool_sel = (dataf['f1'] == bool1) & (dataf['f2'] == bool2)
    else:
        temp1 = (dataf['f1'] == bool1) & (dataf['f2'] == bool2)
        temp2 = (dataf['f1'] == bool2) & (dataf['f2'] == bool1)
        bool_sel = temp1 | temp2
    return bool_sel

def plot_hist(bool_sel, name, dataf):
    g = sns.histplot(data=dataf.loc[bool_sel, :], x='MR', bins=16, log_scale=(False, True))
    g.set_title(name)
    figure_name = name + '_mr.png'
    g.figure.savefig(figure_name)
    plt.close()
    return None


# GO pairs
GO_GO = create_bool('GO', 'GO', df)
GO_Expr = create_bool('GO', 'Expression', df)
GO_PD = create_bool('GO', 'Protein Domain', df)
GO_Net = create_bool('GO', 'Networks', df)
GO_O = create_bool('GO', 'Others', df)

plot_hist(GO_GO, 'GO_GO', df)
plot_hist(GO_Expr, 'GO_Expr', df)
plot_hist(GO_PD, 'GO_PD', df)
plot_hist(GO_Net, 'GO_Net', df)
plot_hist(GO_O, 'GO_O', df)

# Expression pairs
Expr_Expr = create_bool('Expression', 'Expression', df)
Expr_PD = create_bool('Expression', 'Protein Domain', df)
Expr_Net = create_bool('Expression', 'Networks', df)
Expr_O = create_bool('Expression', 'Others', df)

plot_hist(Expr_Expr, 'Expr_Expr', df)
plot_hist(Expr_PD, 'Expr_PD', df)
plot_hist(Expr_Net, 'Expr_Net', df)
plot_hist(Expr_O, 'Expr_O', df)

# Protein domain pairs
PD_PD = create_bool('Protein Domain', 'Protein Domain', df)
PD_Net = create_bool('Protein Domain', 'Networks', df)
PD_O = create_bool('Protein Domain', 'Others', df)

plot_hist(PD_PD, 'PD_PD', df)
plot_hist(PD_Net, 'PD_Net', df)
plot_hist(PD_O, 'PD_O', df)

# Network pairs
Net_Net = create_bool('Networks', 'Networks', df)
Net_O = create_bool('Networks', 'Others', df)

plot_hist(Net_Net, 'Net_Net', df)
plot_hist(Net_O, 'Net_O', df)

# Others pairs
O_O = create_bool('Others', 'Others', df)

plot_hist(O_O, 'O_O', df)

print('Script end:', get_time())
