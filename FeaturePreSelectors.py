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
import pandas as pd
import numpy as np


def from_file(df, **kwargs):
    '''
    Pre-select features from a file
    Lines format in the file: <feature><sep><any info>
    By default <sep> is any whitespace
    '''
    path_to_file = kwargs["path_to_file"]
    sep = kwargs.get("sep", None)

    with open(path_to_file) as f:
        lines = f.readlines()

    return list(filter(lambda x: x in df.columns, map(lambda x: x.split(sep)[0], lines)))
