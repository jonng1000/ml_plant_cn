# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 13:04:00 2019

@author: weixiong001

Creates a new Orthogroups file which corrects the typo in the gene IDs 
(because I didn't rename my cds files before running OF previously). 

IMPT: Run using workstation as its very slow on desktop. Desktop takes >1h but
workstation takes a few seconds.
"""

import pandas as pd
import numpy as np
import csv
import math

orthogrps_df = pd.read_csv('./Orthogroups.tsv',
                           sep='\t', index_col=0)
orthogrps_df_copy = orthogrps_df.copy()

# Each orthogroup is an index, and the value in each othogroup is one string, 
# containing all the genes from the same species in the group, 
# hence further downstream processing is needed
for index, row in orthogrps_df.iterrows():
    orthogroup = orthogrps_df.loc[index]
    # Splits long string into a list of substring, where each substring is 
    # a gene with its long ID
    ug_split = orthogroup.str.split(', ')
    for index2, value2 in ug_split.items():
        if type(value2) == list:
            # Splits each long gene ID and selects the ID which is a whole number,
            # as the gene's ID
            ug_split_split = [gene.split('|')[1] for gene in value2]
            # Makes a list of genes with the correct ID
            processed_genes = ' '.join(ug_split_split)
            # Replace original values with correct values (genes with correct 
            # ID)
            orthogrps_df_copy.loc[index, index2] = processed_genes
        #  To deal with nan
        elif math.isnan(value2):
            continue
        else:
            print('Error, unexpected type!')
            break
    print(index, 'done')
    
orthogrps_df_copy.to_csv('Orthogroups_edited161019.tsv', sep='\t')