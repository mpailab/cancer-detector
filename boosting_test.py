import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

from ExhaustivePipeline import ExhaustivePipeline
from FeaturePreSelectors import from_file


if __name__ == "__main__":
    import sys, datetime
    df = pd.read_csv(sys.argv[1], sep="\t", index_col=0)
    n_processes = int(sys.argv[2])
    #n_k = pd.read_csv(sys.argv[2], sep="\t")
    
    n_k = pd.DataFrame({"n": [47], "k": [4]})

    to_search = [
        {
            # Gradient boosting
            "classifier": XGBClassifier,
            "classifier_kwargs": {
                "objective": "binary:logistic",
                "eval_metric": "logloss",  # TODO: maybe min(TPR, TNR)? Or this is important?
                "verbosity": 0,
                "max_depth": 6,
                "subsample": 0.8,
                "n_estimators": 5000,
                "early_stopping_rounds": 40,
                "learning_rate": 0.001,
                "n_jobs": 1,
                "nthread": 1
            },
            "classifier_CV_ranges": {
                #"subsample": [0.6, 0.7, 0.8],
                #"n_estimators": [500, 1000, 5000],  # Number of boosting rounds
                #"max_depth": [6, 8],  # Maximum tree depth
                #"learning_rate": [0.001, 0.005, 0.01],  # Learning rate (eta)
                #"early_stopping_rounds": [20, 50]
            }
        }
    ]

    for classifier_params in to_search:
        pipeline = ExhaustivePipeline(
            df, n_k, n_processes=n_processes, verbose=True,
            #feature_pre_selector=from_file, feature_pre_selector_kwargs={"path_to_file": "network.txt"},
            main_scoring_threshold=0.6,
            **classifier_params
        )
        results = pipeline.run()
        results.to_csv("output_{}_{}.tsv".format(
            classifier_params["classifier"].__name__,
            datetime.datetime.now().strftime("%d.%m.%Y_%H-%M")
        ), sep="\t")
