# -*- coding: utf-8 -*-
"""
Created on Tue May 4 14:37:57 2021

@author: weixiong001

Use my ml script with multiple models, on multiple targes
Customised for go terms. Modified from multi_go_wrapper.py
in ~/machine_learning/ml_go_workflow/cores_test
"""
import csv
import os
import sys

# Various targets are possible
# class_targets.txt, class_targets_dge.txt
TARGETS = sys.argv[1]
ML_FILE = 'ml_dataset_dc.txt'

with open(TARGETS, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
    for row in reader:
        # Renames the class so that it is compatible with
        # downstream scripts
        one_target = 'go_' + row[0]
        # Various scripts are possible
        os.system('python rem_specific_pcg.py ' + one_target
                  + ' ' + ML_FILE)
        os.system('python rf_go.py ' + one_target)
        file_remove = one_target.replace(':', '_')
        os.remove(file_remove)
        
