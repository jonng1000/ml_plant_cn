# -*- coding: utf-8 -*-
"""
Spyder Editor
Creates a list, sorted in ascending order, of the number of RNA reads
produced by kallistio, from the RNA-seq data.
Also checks thru urls for RNA-seq download, and sees if there are 3 files per
dataset. If there are, that means that there are 2 paired reads, and 1 unpaired
read. Note: This only needed to be done because my earlier version
of the RNA-seq download script, processes unpaired reads first, which is a bug
since a few paired read data has unpaired reads as well (but the paired read
is the one which should be downloaded and processed).
"""   
import os
import json
from datetime import datetime

PATH = 'kal_output/'

print('Script starts', datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
n_proc_reads = []
# Iterating through all kallisto output files
for i in os.listdir(PATH):
    # Need to exclude the outdated folder as its contents can be ignored
    if i == 'outdated_kal':
        continue
    with open(PATH + i + '/run_info.json') as json_file:
        # Gets number of processed reads into a list with its associated
        # run id
        data = json.load(json_file)
        run_id = i.split('_')[0]
        if data['n_processed'] == 0:
            print(run_d, 'has zero reads')
        n_proc_reads.append((run_id, data['n_processed']))

        os.system('python check_3rna_cli.py ' + run_id)
# Sorts list of processed reads in ascending order
n_proc_reads.sort(key=lambda id_reads: id_reads[1])
print('Script ends', datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
