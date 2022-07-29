# -*- coding: utf-8 -*-
"""
Spyder Editor

Command line version of rna_dl_kallisto.py, with modifications
Downloads RNA-seq data from ENA/NCBI. Handles 6 or 7 digit ERA/SRA run IDs
Modifications: Checks for paired reads first, before running kallisto.
Handles the problem of _1.fastq.gz, _2.fastq.gz and .fastq.gz being downloaded,
where .fastq.gz refers to unpaired reads. This is because under this situation,
the script will check for _1.fastq.gz first, and processes it if it is found.

Things to improve for next version of script:
Get script to check if download url .fastq.gz or _1.fastq.gz is valid, if it is
download that specific file only. This is useful for kallisto which doesnt need
_2.fastq.gz files for quantifying gene expression, hence for paired end reads,
can skip downloading _2.fastq.gz files. Could save 1 day of downloads.
"""   
import os, shutil
from datetime import datetime
import sys

# name of index file, produced by kallisto
INDEX = 'Athaliana.idx'

print('Script starts', datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
run_id = sys.argv[1]
database = run_id[:3]
accession = run_id
# First part of ENA download url
# Works for both ENA and SRR accessions
# String cannot follow conventional indentaion as it messes up the command
first_ena = 'ascp -QT -l 1000m -P33001 -i /home/workstation/.aspera/cli/etc/asperaweb_id_dsa.openssh era-fasp@fasp.sra.ebi.ac.uk:/vol1/fastq/'
# Second part of download url
second = accession[:6]
# Third part of download url
third = '00' + accession[-1]
# Creates download url based on accession
if database == 'ERR' or database == 'SRR':
    if len(accession) == 9:
        dl_url = first_ena + second + '/' + accession + ' .'
    elif len(accession) == 10:
        dl_url = first_ena + second + '/' + third + '/' + accession + ' .'
    else:
        print('Accession length unrecognised')
else:
    print('Database id unrecognised')
# Downloads RNA-seq data
os.system(dl_url)

# Name of kallisto quant output folder
out_file = accession + '_kal_out'
# Runs kallisto quant based on whether RNA seq data is single or paired read
# Checks if paired sequence data is downloaded first
if os.path.isfile('./' + accession + '/' + accession + '_1.fastq.gz'):
    print('Downloaded completed, paired read')
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    os.system('kallisto quant -i ' + INDEX + ' -b 100 -o ' + out_file +
              ' --single -l 200 -s 20 -t 8 ./' + accession + '/' +
              accession + '_1.fastq.gz')
elif os.path.isfile('./' + accession + '/' + accession + '.fastq.gz'):
    print('Downloaded completed, single read')
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    os.system('kallisto quant -i ' + INDEX + ' -b 100 -o ' + out_file +
              ' --single -l 200 -s 20 -t 8 ./' + accession + '/' +
              accession + '.fastq.gz')
else:
    print(accession, 'not downloaded correctly')
# Checks to ensure kallisto output file is created
if os.path.isfile('./' + out_file + '/run_info.json'):
    print('Kallisto quant completed')
    print(accession, 'completed')
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
else:
    print(accession, 'not processed correctly')

# Remove RNA-seq data
try:
    shutil.rmtree('./' + accession)
    print('File deleted')
except FileNotFoundError:
    print(accession, 'not found')

print('Script ends', datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
