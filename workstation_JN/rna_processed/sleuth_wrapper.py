# -*- coding: utf-8 -*-
"""
Created on 310820

@author: weixiong
Wrapper for running_sleuth_cli.R to run it on all sub experiments to get their output 
"""

from datetime import datetime
import pandas as pd
import os

INPUT_FOLDER = 'sleuth_metadata'
FILE = 'dge_test_control_labels.tsv'
KAL_FOLDER = './kal_output'

print('Script starts', datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
for a_file in os.listdir('./' + INPUT_FOLDER):
    print('Processing file', datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    file_path = './' + INPUT_FOLDER + '/' + a_file
    df = pd.read_csv(file_path, sep='\t', index_col=0)
    test_sample = [x for x in df['condition'].unique()
                   if x.startswith('test')][0]
    os.system('Rscript running_sleuth_cli.R ' + a_file + ' ' + test_sample)
    
print('Script ends', datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
