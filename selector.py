"""
Feature pre-selection functions

Every feature pre-selection function inputs expression dataframe and number n,
and returns list of features:

def selector(df, n, **kwargs):
    some code

TODO: for supervised feature selection one should also pass 
      subset of datasets for feature selection
"""

# External imports
import numpy as np
import scipy    

# Internal imports
import dataset

def nt_pvalue (df, n, **kwargs):
    '''
    Select n features with respect to normal-tumor pvalues
    '''
    df_normal = dataset.normal()
    df_tumor = dataset.tumor()
    if df_normal is None or df_tumor is None:
        return []

    genes = [ g for g in df if g in df_normal.index and 
                               any(df_normal[p][g] != df_tumor[p][g] for p in df_normal) ]
    df_normal = df_normal.filter(items=genes, axis=0)
    df_tumor = df_tumor.filter(items=genes, axis=0)

    t_test_results = scipy.stats.ttest_rel(df_normal, df_tumor, axis=1)
    ind = np.argsort(t_test_results.pvalue)

    return genes[ind[:n]]

def geffect (df, n, cell_lines=dataset.breast_cell_lines, **kwargs):
    '''
    Select n features with respect to gene effect
    '''
    if dataset.geffect is None:
        return []

    gene_effect = dataset.geffect.loc[cell_lines].mean(axis=0).sort_values()
    return gene_effect[:n].index.to_numpy()

def pubmed (df, n, **kwargs):
    '''
    Select n features with respect to the number of pubmed references
    '''
    if dataset.pubmed is None:
        return []

    genes = df.columns.to_numpy()
    pubmed_df = dataset.pubmed.filter(items=genes, axis=0).sort_values(by='refs_num', ascending=False)
    return pubmed_df[:n].index.to_numpy()