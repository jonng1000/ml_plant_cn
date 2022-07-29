# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 16:31:44 2020

@author: weixiong001

Takes all GO terms, and uses them as features, does 1HE
All labels will have values of either 0 or 1, depending if the gene
has that label or not. Needs CN as uses too much RAM for desktop

Takes ~1 min to run

Remove features if only 1 gene has it
"""
import pandas as pd

GO_TERMS = 'GO_counts.txt'
GO_FEAT = 'go_features_dataset.txt'

go_terms_df = pd.read_csv(GO_TERMS, sep='\t', index_col=0)

# Get all go terms and convert them into binary values, to indicate
# if the gene has that label or not
go_terms_df.loc[:, ['Genes']] = go_terms_df['Genes'].str.split(' ')
one_gene_cp = go_terms_df.explode('Genes').loc[:, ['Genes']]
one_gene_cp = one_gene_cp.reset_index()
one_gene_cp = one_gene_cp.set_index('Genes')
binary_labels = pd.get_dummies(one_gene_cp, prefix='go')
'''
# Just to ensure that there's only 0 and 1, which is what I see
import numpy as np
np.unique(binary_labels.values)
Out[7]: array([0, 1], dtype=uint8)
'''
# These combine duplicated genes into the same row, as the same
# gene occuplies multiple rows if it has >1 go term
# After combining duplicates, each row has one gene, with all its
# labels
# This step uses about 33 GB RAM, need to use CN
combined_rows = binary_labels.groupby(binary_labels.index).sum()
# Drop columns if only 1 gene has that feature
threshold = len(combined_rows) - 1
# About 1.6k GO terms have only 1 gene
to_drop = combined_rows.columns[(combined_rows == 0).sum() == threshold]
removed = combined_rows.drop(columns=to_drop)

'''
# Shows number of genes and features
removed.shape
Out[65]: (31302, 5709)
# Checks that all genes have at least 1 GO term associated with
# it, check passes
>>> (removed.sum(axis=1) >= 1).all()
True
'''
removed.to_csv(GO_FEAT, sep='\t')
