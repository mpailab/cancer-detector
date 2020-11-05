import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from ExhaustivePipeline import ExhaustivePipeline
from FeaturePreSelectors import from_file


if __name__ == "__main__":
    import sys
    df = pd.read_csv(sys.argv[1], sep="\t", index_col=0)
    #n_k = pd.read_csv(sys.argv[2], sep="\t")
    n_k = pd.DataFrame({"n": [23], "k": [8]})  # Several seconds test
    n_processes = int(sys.argv[3])

    to_search = [
        {
            "classifier": SVC,
            "classifier_kwargs": {"kernel": "linear", "class_weight": "balanced"},
            "classifier_CV_ranges": {"C": np.logspace(-4, 4, 9, base=4)}
        },
    ]

    for classifier_params in to_search:
        pipeline = ExhaustivePipeline(
            df, n_k, n_processes=n_processes, verbose=True,
            feature_pre_selector=from_file, feature_pre_selector_kwargs={"path_to_file": "network.txt"},
            main_scoring_threshold=0.6,
            **classifier_params
        )
        results = pipeline.run()
        results.to_csv("network_linear_SVM.tsv", sep="\t")
        print(results)  # TODO: save this dataframe to file
