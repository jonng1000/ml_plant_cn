# -*- coding: utf-8 -*-
"""
Spyder Editor

Wrapper for python script to download RNA-seq data from ENA/NCBI and process it
via kallisto
"""   
import os
import pandas as pd

FILE = 'rna_seq_dl.txt'

df = pd.read_csv(FILE, sep="\t", index_col=0)

for run_id in df['Run'].drop_duplicates():
    os.system('python rna_kal_cli.py ' + run_id)
