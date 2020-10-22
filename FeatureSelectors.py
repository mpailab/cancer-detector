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
import sys

import pandas as pd
import numpy as np

from scipy.stats import spearmanr, ttest_ind, ttest_rel

import xgboost
import shap

from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

# Internal imports
import dataset


def t_test(df, n, **kwargs):
    '''
    Select n features with respect to p-values of the T-test for relapse
    and non-relapse samples.
    '''

    datasets = kwargs.get("datasets", None)
    if not datasets:
        # By default, use all datasets except validation one
        datasets = np.unique(df.loc[df["Dataset type"] != "Validation", "Dataset"])

    df_subset = df.loc[df["Dataset"].isin(datasets)]
    X = df_subset.drop(columns=["Class", "Dataset", "Dataset type"]).to_numpy()
    y = df_subset["Class"].to_numpy()

    t_statistics, pvalues = ttest_ind(X[y == 0], X[y == 1], axis=0)
    features = df_subset.drop(columns=["Class", "Dataset", "Dataset type"]).columns

    return [feature for feature, pvalue in sorted(zip(features, pvalues), key=lambda x: x[1])][0:n]


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

def geffect (df, n, **kwargs):
    '''
    Select n features with respect to gene effect

    Parameters:

    cell_lines : list or numpy array, default dataset.breast_cell_lines

        The cell lines to filter on gene effect table
    '''
    if dataset.geffect is None:
        return []

    cell_lines = kwargs.get("cell_lines", dataset.breast_cell_lines)
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

def nt_diff (df, n, **kwargs):
    '''
    Select n features with respect to difference between normal and tumor expressions
    diff: int or function (n, t -> float), where n, t are np.arrays of equal length
    '''
    diff = kwargs.get("diff", 0)

    g_mean = lambda ar: ar.prod() ** (1. / ar.size)
    a_mean = lambda ar: ar.sum() / ar.size
    default = {
        0: lambda n, t: g_mean(np.absolute(t - n) / n),
        1: lambda n, t: g_mean(np.absolute(t - n) / t),
        2: lambda n, t: abs(a_mean(t - n)),
        3: lambda n, t: a_mean(np.absolute(t - n)),
        4: lambda n, t: g_mean(t / n),
        5: lambda n, t: g_mean(n / t),
        6: lambda n, t: max(np.absolute(t - n))
        }
    if diff in default.keys():
        diff = default[diff]

    df_normal = dataset.normal()
    df_tumor = dataset.tumor()
    if df_normal is None or df_tumor is None:
        return []

    genes = [ g for g in df if g in df_normal.index and
                               any(df_normal[p][g] != df_tumor[p][g] for p in df_normal) ]
    df_normal = df_normal.filter(items=genes, axis=0)
    df_tumor = df_tumor.filter(items=genes, axis=0)

    dist = {g : diff(df_normal.loc[g], df_tumor.loc[g]) for g in genes}
    genes = list(filter(lambda g: str(dist[g]) not in ['nan', 'inf', '-inf'] , genes))
    genes.sort(key = lambda g: dist[g], reverse=True)

    #[print("{} : {}".format(g, dist[g])) for g in genes]
    return genes[:n]

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
        corr, _ =  spearmanr(X[:,i], Y)
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
    datasets = kwargs.get("datasets", None)
    if not datasets:
        # By default, use all datasets except validation one
        datasets = np.unique(df.loc[df["Dataset type"] != "Validation", "Dataset"])

    df_subset = df.loc[df["Dataset"].isin(datasets)]
    X = df_subset.drop(columns=["Class", "Dataset", "Dataset type"]).to_numpy()
    Y = df_subset["Class"].to_numpy()
    n_genes = X.shape[1]
    p_values = np.zeros(n_genes)
    for i in range(n_genes):
        _, pval =  spearmanr(X[:,i], Y)
        p_values[i] = pval

    features = df_subset.drop(columns=["Class", "Dataset", "Dataset type"]).columns

    return [feature for feature, p_values in sorted(zip(features, p_values), key=lambda x: x[1], reverse=False)][0:n]

def boosting_shapley(df, n, **kwargs):
    '''
    Input expression dataframe and number n, return list
    of n selected features
    TODO: for supervised feature selection one should also pass
    subset of datasets for feature selection
    '''
    datasets = kwargs.get("datasets", None)
    if not datasets:
        # By default, use all datasets except validation one
        datasets = np.unique(df.loc[df["Dataset type"] != "Validation", "Dataset"])

    eta = kwargs.get("eta", 0.001)
    num_rounds = kwargs.get("num_rounds", 3000)
    early_stopping_rounds = kwargs.get("early_stopping_rounds", 40)
    subsample = kwargs.get("subsample", 0.8)

    df_subset = df.loc[df["Dataset"].isin(datasets)]
    X = df_subset.drop(columns=["Class", "Dataset", "Dataset type"]).to_numpy()
    y = df_subset["Class"].to_numpy()
    xgboost_input = xgboost.DMatrix(X, label=y)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": eta,
        "subsample": subsample,
        "base_score": np.mean(y)
    }
    model = xgboost.train(
        params,
        xgboost_input,
        num_rounds,
        evals = [(xgboost_input, "test")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False
    )

    shap_values = shap.TreeExplainer(model).shap_values(X)
    feature_importances = np.mean(np.abs(shap_values), axis=0)
    features = df_subset.drop(columns=["Class", "Dataset", "Dataset type"]).columns

    return [feature for feature, importance in sorted(zip(features, feature_importances), key=lambda x: x[1], reverse=True)][0:n]

def linearSVC(df, n, **kwargs):
    '''
    Select n features with respect to coefficients of LinearSVC.
    '''

    datasets = kwargs.get("datasets", None)
    _df = df if datasets is None else df.loc[df["Dataset"].isin(datasets)]
    genes = _df.drop(columns=["Class", "Dataset", "Dataset type"]).columns.to_numpy()

    X = _df.drop(columns=["Class", "Dataset", "Dataset type"]).to_numpy()
    y = _df["Class"].to_numpy()

    c_best = 1.0
    delta = float("inf")
    for c in np.logspace(-4, 4, 9):
        clf = make_pipeline( StandardScaler(),
                             LinearSVC( penalty='l1', C=c, dual=False))
        clf.fit(X, y)
        coef = clf.named_steps['linearsvc'].coef_
        coef = np.resize(coef,(coef.shape[1],))
        m = len(np.nonzero(coef)[0])
        if n <= m and delta > m - n:
            delta = m - n
            c_best = c

    clf = make_pipeline( StandardScaler(),
                         LinearSVC( penalty='l1', C=c_best, dual=False))
    clf.fit(X, y)
    coef = clf.named_steps['linearsvc'].coef_
    coef = np.resize(coef,(coef.shape[1],))
    ind  = np.nonzero(coef)[0]
    np.random.shuffle(ind)

    return genes[ind[:n]]

##########################################################################################

HASH = {
    'nt_pvalue'         : nt_pvalue,
    'geffect'           : geffect,
    'pubmed'            : pubmed,
    'top_from_file'     : top_from_file,
    'nt_diff'           : nt_diff,
    'max_correlation'   : max_correlation,
    'min_p_value'       : min_p_value
}

def get (name):
    '''
    Get selector by name

    Returns:
        a selector function if name is its name;
        None, otherwise.
    '''
    return HASH[name] if name in HASH else None

def funcs ():
    '''
    Get all avaliable selectors

    Returns:
        list of functions
    '''
    return list(HASH.values())

def names ():
    '''
    Get names of all avaliable selectors

    Returns:
        list of strings
    '''
    return list(HASH.keys())


if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1], sep="\t", index_col=0)
    print(df)

    print(t_test(df, 20))
