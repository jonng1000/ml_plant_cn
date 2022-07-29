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
import urllib.request
from datetime import datetime

PATH = 'kal_output/'

print('Script starts', datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
n_proc_reads = []
# First part of ENA download url (ftp)
# ascp doesnt work with urllib.request
first_ena = 'ftp://ftp.sra.ebi.ac.uk/vol1/fastq/'

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
        database = run_id[:3]
        if data['n_processed'] == 0:
            print(run_d, 'has zero reads')
        n_proc_reads.append((run_id, data['n_processed']))

    # Assembling download url
    # Second part of download url
    second = run_id[:6]
    # Third part of download url
    third = '00' + run_id[-1]
    # Creates partial download url based on accession
    if database == 'ERR' or database == 'SRR':
        if len(run_id) == 9:
            dl_url = first_ena + second + '/' + run_id + '/'
        elif len(run_id) == 10:
            # Squeezed first_ena+second to keep code to 1 line
            dl_url = first_ena + second + '/' + third + '/' + run_id + '/'
        else:
            print('Accession length unrecognised')
    else:
        print('Database id unrecognised')

    # Checks to see if ftp link is valid
    # Assembles full download url here
    single_read = False
    paired_read1 = False
    paired_read2 = False
    pass_checks = False

    # Checks if single read fastq exists
    try:
        urllib.request.urlopen(dl_url + run_id + '.fastq.gz')
        single_read = True
    except urllib.error.URLError:
        pass
    # Checks if paired read 1 fastq exists
    try:
        urllib.request.urlopen(dl_url + run_id + '_1.fastq.gz')
        paired_read1 = True
    except urllib.error.URLError:
        pass
    # Checks if paired read 2 fastq exists
    try:
        urllib.request.urlopen(dl_url + run_id + '_2.fastq.gz')
        paired_read2 = True
    except urllib.error.URLError:
        pass
    # Checks that single read and not paired read fastq exists
    if single_read:
        if (not paired_read1) and (not paired_read2):
            pass_checks = True
    # Checks that single read and paired read fastq (3 files) exists
    if single_read and (paired_read1 and paired_read2):
        print(run_id, 'has 3 fastq files')
        pass_checks = True
    # Checks that not single read and paired read fastq  exists
    if not single_read:
        if paired_read1 and paired_read2:
            pass_checks = True
    # Checks that at least one of the above checks passes
    if not pass_checks:
        print(run_id, 'fails number of fastq files check')
        
 
# Sorts list of processed reads in ascending order
n_proc_reads.sort(key=lambda id_reads: id_reads[1])
print('Script ends', datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
