# -*- coding: utf-8 -*-
"""
Created on 290520

@author: weixiong

This script takes in the inteproscan output file, selects the protein domain
information, and converts it into dummary variables (binary features).
Also counts the number of protein domains (including and
excluding repeated ones - 2 different features), disordered
regions and transmembrane helixes.

Total estimated time to run this is 10 min. Edited from analyse_interpro.py in
/mnt/d/GoogleDrive/machine_learning/my_features/data_preprocessing
Using Signature accession instead of description as ID, and represents missing
values as NA. This is different from orginal version of the script.
"""

#import pandas as pd
import modin.pandas as pd

# Comment out one of it depending on OS I am using
#PATH = 'D:/GoogleDrive/machine_learning/my_features/interpro_files/'
PATH = './'

FILE = PATH + 'ath_aa_processed.fa.tsv'
OUTPUT = 'protein_doms.txt'

lst_names = ['Protein_Accession', 'Sequence_MD5_digest', 'Sequence Length',
             'Analysis', 'Signature Accession', 'Signature Description',
             'Start location', 'Stop location', 'Score', 'Status', 'Date',
             'InterPro_annotations_accession',
             'InterPro_annotations_description', 'GO_annotations',
             'Pathways_annotations']
df = pd.read_csv(FILE, sep='\t',  names=lst_names)

lst_selection = ['Protein_Accession', 'Analysis', 'Signature Accession']
df_selection = df.loc[:, lst_selection]
pfam = df_selection.loc[df_selection['Analysis'] == 'Pfam',:]

# OHE protein domains
pfam_nodup = pfam.drop_duplicates()
grouped_pfam = pfam_nodup.groupby('Protein_Accession')
concat_pfam = grouped_pfam['Signature Accession'].apply(' '.join)
pfam_ohe = concat_pfam.str.get_dummies(sep=' ').add_prefix('pfa_')

# Counts number of domains, including repeated ones
pfam_counts = pfam.groupby('Protein_Accession').count().\
              loc[:, 'Signature Accession']
pfam_counts.name = 'num_counts'

# Counts number of domains, excluding repeated ones
pfam_u_counts = grouped_pfam.count().loc[:, 'Signature Accession']
pfam_u_counts.name = 'num_u_counts'

# Counts mobi database hits
mobi = df_selection.loc[df_selection['Analysis'] == 'MobiDBLite',:]
mobi_grp = mobi.groupby(['Protein_Accession']).count()
mobi_counts = mobi_grp.loc[:, 'Signature Accession']
mobi_counts.name = 'mob_counts'

# Counts tmhmm database hits
tmhmm = df_selection.loc[df_selection['Analysis'] == 'TMHMM',:]
tmhmm_grp = tmhmm.groupby(['Protein_Accession']).count()
tmhmm_counts = tmhmm_grp.loc[:, 'Signature Accession']
tmhmm_counts.name = 'tmh_counts'

pro_dom = pd.concat([pfam_ohe, pfam_counts, pfam_u_counts, mobi_counts,
                     tmhmm_counts], axis=1)
pro_dom.index.name = 'Gene'
pro_dom.to_csv(OUTPUT, sep='\t', na_rep='NA')

# Extra info just for checking
# Number of GB used in memory by dataframe
#>>> sum(pro_dom.memory_usage())/1000/1000/1000
#0.842723136
# About 248 M cells in dataframe
#>>> pro_dom.size
#105315021
# Dimensons of dataframe
#>>> pro_dom.shape
#(25371, 4151)
# Calculates the number of 0s and 1s in dataframe
# Not entirely accurate as some of the 1s are from non-domain features,
# but the general idea, that domains form a sparse matrix, still holds
#>>> pro_dom.isin([0]).sum().sum()
#90574611
#>>> pro_dom.isin([1]).sum().sum()
#37852


