# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 21:34:46 2020

@author: weixiong001

Use my ml script with rf, on multiple targets for hp testing
Customised for go terms. Modified from multi_go_wrapper.py
in ~/machine_learning/workflow_v2
"""
import csv
import os

# Various targets are possible
# class_targets.txt, class_targets_dge.txt
TARGETS = 'ct_go_hp_v2.txt'

with open(TARGETS, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
    for row in reader:
        one_target = row[0]
        # Various scripts are possible
        # multi_models.py, rf_16cl_v3.py
        os.system('python rem_specific_pcg.py ' + one_target)
        os.system('python rf_m_classes_v2.py ' + one_target)
        file_remove = one_target.replace(':', '_')
        os.remove(file_remove)
