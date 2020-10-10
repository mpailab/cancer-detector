"""
Feature selection functions

Every feature selection function inputs expression dataframe and number n,
and returns list of features:

def selector(df, n, **kwargs):
    some code

TODO: for supervised feature selection one should also pass 
      subset of datasets for feature selection
"""

# External imports
import numpy as np
import scipy    
from scipy.stats import spearmanr

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

def top_from_file (df, n, **kwargs):
    '''
    Select n top features from a file
    Lines format in the file: <feature><sep><any info>
    By default <sep> is any whitespace
    '''
    filepath = kwargs["file"]
    sep = kwargs.get("sep", None)
    
    with open(filepath) as f:
        lines = f.readlines()
    return list(filter(lambda x: x in df.columns, map(lambda x: x.split(sep)[0], lines)))[:n]

def max_correlation(df, n, **kwargs):
    '''
    Input expression dataframe and number n, return list
    of n selected features
    Uses Spearman correlation to select the most important genes
    TODO: for supervised feature selection one should also pass
    subset of datasets for feature selection
    '''
    datasets = kwargs["datasets"]

    df_subset = df.loc[df["Dataset"].isin(datasets)]
    X = df_subset.drop(columns=["Class", "Dataset", "Dataset type"]).to_numpy()
    Y = df_subset["Class"].to_numpy()
    n_genes = X.shape[1]
    corr_coeff = np.zeros(n_genes)
    for i in range(n_genes):
        corr, pval =  spearmanr(X[:,i], Y)
        corr_coeff[i] = corr

    features = df_subset.drop(columns=["Class", "Dataset", "Dataset type"]).columns

    return [feature for feature, corr_coeff in sorted(zip(features, corr_coeff), key=lambda x: x[1], reverse=True)][0:n]


def min_p_value(df, n, **kwargs):
    '''
    Input expression dataframe and number n, return list
    of n selected features
    Uses Spearman p-value to select most important genes
    TODO: for supervised feature selection one should also pass
    subset of datasets for feature selection
    '''
    datasets = kwargs["datasets"]

    df_subset = df.loc[df["Dataset"].isin(datasets)]
    X = df_subset.drop(columns=["Class", "Dataset", "Dataset type"]).to_numpy()
    Y = df_subset["Class"].to_numpy()
    n_genes = X.shape[1]
    p_values = np.zeros(n_genes)
    for i in range(n_genes):
        corr, pval =  spearmanr(X[:,i], Y)
        p_values[i] = pval

    features = df_subset.drop(columns=["Class", "Dataset", "Dataset type"]).columns

    return [feature for feature, p_values in sorted(zip(features, p_values), key=lambda x: x[1], reverse=False)][0:n]

