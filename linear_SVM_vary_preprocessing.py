import pandas as pd
import numpy as np

from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler, KBinsDiscretizer

from ExhaustivePipeline import ExhaustivePipeline


if __name__ == "__main__":
    import sys, datetime
    df = pd.read_csv(sys.argv[1], sep="\t", index_col=0)
    n_k = pd.read_csv(sys.argv[2], sep="\t")
    n_processes = int(sys.argv[3])
    preprocessing = sys.argv[4]

    if preprocessing == "StandardScaler":
        preprocessor = StandardScaler
        preprocessor_kwargs = {}
    elif preprocessing == "Normalizer":
        preprocessor = Normalizer
        preprocessor_kwargs = {}
    elif preprocessing == "MinMaxScaler":
        preprocessor = MinMaxScaler
        preprocessor_kwargs = {}
    elif preprocessing == "KBinsDiscretizer_2":
        preprocessor = KBinsDiscretizer
        preprocessor_kwargs = {"n_bins": 2}
    elif preprocessing == "KBinsDiscretizer_10":
        preprocessor = KBinsDiscretizer
        preprocessor_kwargs = {"n_bins": 10}
    elif preprocessing == "KBinsDiscretizer_20":
        preprocessor = KBinsDiscretizer
        preprocessor_kwargs = {"n_bins": 20}
    else:
        print("Please provide correct preprocessing string")
        quit()
    
    classifier_params = {
        "classifier": SVC,
        "classifier_kwargs": {"kernel": "linear", "class_weight": "balanced"},
        "classifier_CV_ranges": {"C": np.logspace(-4, 4, 9, base=4)}
    }

    pipeline = ExhaustivePipeline(
        df, n_k, n_processes=n_processes, random_state=17, verbose=True,
        main_scoring_threshold=0.6,
        preprocessor=preprocessor, preprocessor_kwargs=preprocessor_kwargs,
        **classifier_params
    )
    results = pipeline.run()
    results.to_csv("output_linear_SVM_{}_{}.tsv".format(
        preprocessing,
        datetime.datetime.now().strftime("%d.%m.%Y_%H-%M")
    ), sep="\t")
