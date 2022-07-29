# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 10:46:11 2020

@author: weixiong001

Explores complete list of continuous features vs list of features successfully
completed by my ml workflow, to check to see if there's any errors and if all
features are successfully completed
"""

import csv

FILE = 'mod_class_labels_linux.txt'
FILE2 = 'output_names_cor.txt'

targets_set = set()
with open(FILE, newline='') as csvfile:
    file_reader = csv.reader(csvfile, delimiter='\t', )
    for row in file_reader:
        targets_set.add(row[1])

done_set = set()
with open(FILE2, newline='') as csvfile:
    file_reader = csv.reader(csvfile, delimiter='\t', )
    for row in file_reader:
        job_id = '_'.join(row[0].split('_')[:2])
        done_set.add(job_id)
        
'''
# Previous ml runs where workflow is buggy due to special characters in cont
feature names
FILE = 'mod_class_labels_linux.txt'
FILE2 = 'output_names.txt'
>>> len(targets_set) - len(done_set)
7
>>> len(targets_set)
3155
>>> len(done_set)
3148
'''
'''
# ML runs where workflow is corrected due to special characters in cont
feature names
FILE = 'mod_class_labels_linux.txt'
FILE2 = 'output_names_cor.txt'
>>> len(targets_set) - len(done_set)
0
>>> len(targets_set)
3155
>>> len(done_set)
3155
'''
