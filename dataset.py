"""Datasets"""

# External imports
import pandas as pd
import numpy as np
import re
import os

# Internal imports
import option

# BRCA dataset
# columns: genes
# rows: patients
_brca_file = os.path.join(option.database, 'BRCA_U133A.csv')
if os.path.isfile(_brca_file):
    brca = pd.read_csv( _brca_file, index_col=0)
else:
    brca = None

# TCGA dataset
# columns: genes
# rows: patients
_tcga_file = os.path.join(option.database, 'BreastCancer/TCGA.tsv')
if os.path.isfile(_tcga_file):
    tcga = pd.read_csv( _tcga_file, sep='\t', index_col=0)
else:
    tcga = None

# TCGA dataset with tumor and normal expressions
# columns: patients with _tumor and _normal suffixes
# rows: genes
_tcga_nt_file = os.path.join(option.database, 'mRNA_normal_tumor.tsv')
if os.path.isfile(_tcga_nt_file):
    tcga_nt = pd.read_csv( _tcga_nt_file, sep="\t", index_col=0)
else:
    tcga_nt = None

def normal (df=tcga_nt):
    '''
    Extract patients with _normal suffix
    '''
    if df is None:
        return None

    return df.filter(regex=(".*_normal")).rename(columns=lambda x: re.sub('_normal', '', x))

def tumor (df=tcga_nt):
    '''
    Extract patients with _tumor suffix
    '''
    if df is None:
        return None

    return df.filter(regex=(".*_tumor")).rename(columns=lambda x: re.sub('_tumor', '', x))

# Gene effect dataset
# columns: genes
# rows: cell lines
_geffect_file = os.path.join(option.database, 'Achilles_gene_effect.csv')
if os.path.isfile(_tcga_nt_file):
    geffect = pd.read_csv( _geffect_file, index_col=0)
    geffect = geffect.rename(columns=lambda x: re.sub(r'(.*) \([0-9]*\)',r'\1',x))
else:
    geffect = None

# Breast cell lines list
_breast_cell_lines_file = os.path.join(option.database, 'Breast_cell_lines.txt')
if os.path.isfile(_breast_cell_lines_file):
    breast_cell_lines = np.fromfile( _breast_cell_lines_file, dtype=str, sep=',')
else:
    breast_cell_lines = None

# Gene effect dataset
# columns: numbers of references 
# rows: genes
_pubmed_file = os.path.join(option.database, 'genes_pubmed_refs_nums.tsv')
if os.path.isfile(_pubmed_file):
    pubmed = pd.read_csv( _pubmed_file, sep='\t', index_col=0, names=['gene', 'refs_num'])
else:
    pubmed = None