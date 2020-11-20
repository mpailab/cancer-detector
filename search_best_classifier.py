import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

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
                "weights": "distance"
            },

            "classifier_CV_ranges": {
                "p": [1, 2],  # L1 and L2 metric
                "n_neighbors": list(range(1, int(sqrt(154)) + 1))  # 154 is the size of the traning set
            }
        },
        {
            # Random forest
            "classifier": RandomForestClassifier,
            "classifier_kwargs": {
                "n_estimators": 100,  # TODO: some papers say that 100 will be sufficient, some say 500. Check 500
                "class_weight": "balanced",
            },
            "classifier_CV_ranges": {
                "max_features": ["log2", "sqrt", 1/3],
                "max_samples": [0.2, 0.55, 0.9]
            }
        },
        {
            # Gradient boosting
            "classifier": XGBClassifier,
            "classifier_kwargs": {
                "objective": "binary:logistic",
                "eval_metric": "logloss",  # TODO: maybe min(TPR, TNR)? Or this is important?
                "verbosity": 0,
            },
            "classifier_CV_ranges": {
                "subsample": [0.6, 0.7, 0.8],
                "n_estimators": [500, 1000, 1500],  # Number of boosting rounds
                "max_depth": [6, 8],  # Maximum tree depth
                "learning_rate": [0.001, 0.005, 0.01],  # Learning rate (eta)
                "early_stopping_rounds"= [20, 50]
            }
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
