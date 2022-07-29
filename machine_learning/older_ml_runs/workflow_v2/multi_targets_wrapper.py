# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 21:34:46 2020

@author: weixiong001

Use my ml script with multiple models, on multiple targes
"""
import csv
import os

# Various targets are possible
# class_targets.txt, class_targets_dge.txt
TARGETS = 'class_targets_dge.txt'

with open(TARGETS, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
    for row in reader:
        one_target = row[0]
        # Various scripts are possible
        # multi_models.py, rf_16cl_v3.py
        os.system('python multi_models.py ' + one_target)
