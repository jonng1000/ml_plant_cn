# -*- coding: utf-8 -*-
"""
Spyder Editor
Command line version of check_3rna_files.py with further modifications
Checks thru urls for RNA-seq download, and sees if there are 3 files per
dataset. If there are, that means that there are 2 paired reads, and 1 unpaired
read. Note: This only needed to be done because my earlier version
of the RNA-seq download script, processes unpaired reads first, which is a bug
since a few paired read data has unpaired reads as well (but the paired read
is the one which should be downloaded and processed).
"""
import sys
import json
import urllib.request

PATH = 'kal_output/'

# First part of ENA download url (ftp)
# ascp doesnt work with urllib.request
first_ena = 'ftp://ftp.sra.ebi.ac.uk/vol1/fastq/'

run_id = sys.argv[1]
database = run_id[:3]
kal_json_path = PATH  + run_id + '_kal_out' + '/run_info.json'
# Iterating through all kallisto output files    
with open(kal_json_path) as json_file:
    # Gets number of processed reads into a list with its associated
    # run id
    data = json.load(json_file)
    if data['n_processed'] == 0:
        print(run_d, 'has zero reads')

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
