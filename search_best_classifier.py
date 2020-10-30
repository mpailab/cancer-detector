import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from ExhaustivePipeline import ExhaustivePipeline


if __name__ == "__main__":
    import sys
    df = pd.read_csv(sys.argv[1], sep="\t", index_col=0)
    #n_k = pd.read_csv(sys.argv[2], sep="\t")
    n_k = pd.DataFrame({"n": [10, 10], "k": [8, 9]})  # Several seconds test
    n_processes = int(sys.argv[3])

    to_search = [
        {
            "classifier": SVC,
            "classifier_kwargs": {"kernel": "linear", "class_weight": "balanced"},
            "classifier_CV_ranges": {"C": np.logspace(-4, 4, 9, base=4)}
        },
        {
            "classifier": LogisticRegression,
            "classifier_kwargs": {"class_weight": "balanced"},
            "classifier_CV_ranges": {"C": np.logspace(-4, 4, 9, base=4)}
        },
        {
            "classifier": RandomForestClassifier,
            "classifier_kwargs": {"class_weight": "balanced"},
            "classifier_CV_ranges": {"n_estimators": [10, 50, 100, 500]}
        },
        {
            "classifier": KNeighborsClassifier,
            "classifier_kwargs": {},
            "classifier_CV_ranges": {"n_neighbors": np.arange(1, 10, 2), "weights": ["uniform", "distance"]}
        }
    ]

    for classifier_params in to_search:
        pipeline = ExhaustivePipeline(df, n_k, n_processes=n_processes, verbose=True, **classifier_params)
        results = pipeline.run()
        print(results)  # TODO: save this dataframe to file
