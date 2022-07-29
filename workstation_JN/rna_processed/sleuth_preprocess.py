# -*- coding: utf-8 -*-
"""
Created on 310820

@author: weixiong
Preprocessing flies for sleuth. Splits all experiments into their specific
sub experiments with 1 test and 1 control set of samples
"""

import numpy as np
import pandas as pd
import os

FILE = 'dge_test_control_labels.tsv'
KAL_FOLDER = './kal_output'
SLEUTH_META = './sleuth_metadata'

df = pd.read_csv(FILE, sep='\t', index_col=0)
df.drop(columns=['Annotation1', 'Annotation2'], inplace=True)
df.drop_duplicates(inplace=True)
df.reset_index(inplace=True)
path_lst = []
for a_run in df['Run']:
    path = './kal_output' + '/' +  a_run + '_kal_out'
    path_lst.append(path)
df['path'] = path_lst

grouped = df.groupby('Experiment')
for name, group in grouped:
    # Takes care of the situation where there's only one type of test and control
    # samples
    if set(group['Control_test_label']) == {'control_1', 'test_1'}:
        partial = group.drop(columns=['Experiment'])
        partial.index = np.arange(1, len(partial) + 1)
        complete = partial.rename(columns={'Run': 'sample',
                                           'Control_test_label': 'condition'})
        metadata_file = SLEUTH_META + '/' + name + '_sm.txt'
        complete.to_csv(metadata_file, sep='\t')
        continue
    # Takes care of the situation where there's more than one type of test
    # and control
    # samples, but each control is only paired with one type of test sample     
    temp_set = {cond_type[-1] for cond_type in \
                group['Control_test_label'].unique()}
    if 'a' not in temp_set:
        for cond_type in group['Control_test_label'].unique():
            if cond_type.startswith('control'):
                test_name = 'test_' + cond_type.split('_')[1]
                temp_control = group.loc[group['Control_test_label'] == \
                                         cond_type]
                temp_test = group.loc[group['Control_test_label'] == test_name]
                partial = pd.concat([temp_control, temp_test])
                partial = partial.drop(columns=['Experiment'])
                partial.index = np.arange(1, len(partial) + 1)
                complete = partial.rename(columns=\
                                          {'Run': 'sample',
                                           'Control_test_label': 'condition'})
                metadata_file = SLEUTH_META + '/' + name + '_' + \
                                cond_type.split('_')[1]  + '_sm.txt'
                complete.to_csv(metadata_file, sep='\t')
        continue

    if 'a' in  temp_set:
        for cond_type in group['Control_test_label'].unique():
            if cond_type.startswith('control'):
                temp_control = group.loc[group['Control_test_label'] == \
                                         cond_type]
                test_num = 'test_' + cond_type.split('_')[1]
                for a_type in group['Control_test_label'].unique():
                    if test_num in a_type:
                        temp_test = group.loc[group['Control_test_label'] \
                                              == a_type]
                        partial = pd.concat([temp_control, temp_test])
                        partial = partial.drop(columns=['Experiment'])
                        partial.index = np.arange(1, len(partial) + 1)
                        complete = partial.rename(columns=\
                                          {'Run': 'sample',
                                           'Control_test_label': 'condition'})
                        metadata_file = SLEUTH_META + '/' + name + '_' + \
                                        a_type.split('_')[1]  + '_sm.txt'
                        complete.to_csv(metadata_file, sep='\t')
        continue

