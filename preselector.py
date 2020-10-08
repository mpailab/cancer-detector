"""
Feature pre-selection functions

Every feature pre-selection function inputs expression dataframe and 
returns list of features:

def preselector(df, **kwargs):
    some code
"""

# Internal imports
import feature

def brca (df, **kwargs):
    '''
    Select features from BRCA dataset
    '''
    return [ g for g in df if g in feature.brca ]

def tcga (df, **kwargs):
    '''
    Select features from TCGA dataset
    '''
    return [ g for g in df if g in feature.tcga ]

def good_genes (df, **kwargs):
    '''
    Select good genes
    '''
    return [ g for g in df if g in feature.good_genes ]

def super_good_genes (df, **kwargs):
    '''
    Select super good genes
    '''
    return [ g for g in df if g in feature.super_good_genes ]

def oncotype_dx (df, **kwargs):
    '''
    Select oncotype_dx genes
    '''
    return [ g for g in df if g in feature.oncotype_dx ]

def mammaprint (df, **kwargs):
    '''
    Select mammaprint genes
    '''
    return [ g for g in df if g in feature.mammaprint ]