import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from ExhaustivePipeline import ExhaustivePipeline
from FeaturePreSelectors import from_file


if __name__ == "__main__":
    import sys, datetime
    df = pd.read_csv(sys.argv[1], sep="\t", index_col=0)
    n_k = pd.read_csv(sys.argv[2], sep="\t")
    n_processes = int(sys.argv[3])
    
    #n_k = pd.DataFrame({"n": [23, 20], "k": [8, 9]})

    to_search = [
        {
            # Linear SVM
            "classifier": SVC,
            "classifier_kwargs": {"kernel": "linear", "class_weight": "balanced"},
            "classifier_CV_ranges": {"C": np.logspace(-4, 4, 9, base=4)}
        },
        {
            # Non-linear SVM. Available options for kernel: rbf, poly and sigmoid
            "classifier": SVC,
            "classifier_kwargs": {"kernel": "rbf", "gamma": "scale", "class_weight": "balanced"},
            "classifier_CV_ranges": {"C": np.logspace(-4, 4, 9, base=4)}
        },
        {
            # Logistic regression
            "classifier": LogisticRegression,
            "classifier_kwargs": {"class_weight": "balanced"},
            "classifier_CV_ranges": {"C": np.logspace(-4, 4, 9, base=4)}
        },
        {
            # kNN
            "classifier": KNeighborsClassifier,
            "classifier_kwargs": {
                "weights": "uniform",  # TODO: "uniform" or "distance"?
                "metric": "minkowski"  # TODO: custom function with weighting proportionally to classes
            },
            "classifier_CV_ranges": {
                "n_neighbors": [1, 3, 5, 7, 9]  # TODO: which range to use there?
            }
        },
        {
            # Random forest
            "classifier": RandomForestClassifier,
            "classifier_kwargs": {
                "n_estimators": 100,  # TODO: Number of trees. Should we cross-validate this? Which range to use?
                "max_depth": None,  # TODO: Maximum depth of a tree. Should we use this parameter? Should we cross-validate this? Which range to use?
                "max_features": "sqrt",  # Size of feature subset for each tree. Sqrt, log or a fraction. Should we cross-validate this? Which range to use?
                "max_samples": None,  # Size of a training set subset for each tree. Should we cross-validate this? Which range to use?
                "class_weight": "balanced"
            },
            "classifier_CV_ranges": {}
        }
    ]

    for classifier_params in to_search:
        pipeline = ExhaustivePipeline(
            df, n_k, n_processes=n_processes, verbose=True,
            feature_pre_selector=from_file, feature_pre_selector_kwargs={"path_to_file": "network.txt"},
            main_scoring_threshold=0.6,
            **classifier_params
        )
        results = pipeline.run()
        results.to_csv("output_{}_{}.tsv".format(
            classifier_params["classifier"].__name__,
            datetime.datetime.now().strftime("%d.%m.%Y_%H-%M")
        ), sep="\t")
