import pandas as pd
import numpy as np

from multiprocessing import Pool
import time
import math
import itertools
from natsort import natsorted

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from FeatureSelectors import t_test
from AccuracyMetrics import TPR, TNR, min_TPR_TNR


class ExhaustivePipeline:
    def __init__(
        self, df, n_k, n_processes=1, verbose=True,
        feature_pre_selector=None, feature_pre_selector_kwargs={},
        feature_selector=t_test, feature_selector_kwargs={},
        preprocessor=StandardScaler, preprocessor_kwargs={},
        classifier=SVC, classifier_kwargs={"kernel": "linear", "class_weight": "balanced"},
        classifier_CV_ranges={"C": np.logspace(-4, 4, 9, base=4)}, classifier_CV_folds=5,
        scoring_functions={"TPR": TPR, "TNR": TNR, "min_TPR_TNR": min_TPR_TNR},
        main_scoring_function="min_TPR_TNR", main_scoring_threshold=0.65
    ):
        '''
        df: pandas dataframe. Rows represent samples, columns represent features (e.g. genes).
        df should also contain three columns:
            -  "Class": binary values associated with target variable;
            -  "Dataset": id of independent dataset;
            -  "Dataset type": "Training", "Filtration" or "Validation".

        n_k: pandas dataframe. Two columns must be specified:
            -  "n": number of features for feature selection
            -  "k": tuple size for exhaustive search
        '''

        self.df = df
        self.n_k = n_k
        self.n_processes = n_processes
        self.verbose = verbose

        self.feature_pre_selector = feature_pre_selector
        self.feature_pre_selector_kwargs = feature_pre_selector_kwargs

        self.feature_selector = feature_selector
        self.feature_selector_kwargs = feature_selector_kwargs

        self.preprocessor = preprocessor
        self.preprocessor_kwargs = preprocessor_kwargs

        self.classifier = classifier
        self.classifier_kwargs = classifier_kwargs  # Fixed parameters
        self.classifier_CV_ranges = classifier_CV_ranges  # e.g. {"C": np.logspace(-4, 4, 9, base=4)}
        self.classifier_CV_folds = classifier_CV_folds

        self.scoring_functions = scoring_functions  # e.g. {"min_TPR_TNR": fun(y_true, y_pred), "TPR": ..., ...}
        self.main_scoring_function = main_scoring_function  # e.g. "min_TPR_TNR"
        self.main_scoring_threshold = main_scoring_threshold

    def run(self):
        # First, pre-select features
        if self.feature_pre_selector:
            features = self.feature_pre_selector(self.df, **self.feature_pre_selector_kwargs)
        else:
            features = self.df.columns.drop(["Class", "Dataset", "Dataset type"]).to_list()

        df_pre_selected = self.df[features + ["Class", "Dataset", "Dataset type"]].copy()

        # Start iterating over n, k pairs
        all_result_dfs = []
        for n, k in zip(self.n_k["n"], self.n_k["k"]):
            features = self.feature_selector(df_pre_selected, n, **self.feature_selector_kwargs)
            df_selected = df_pre_selected[features + ["Class", "Dataset", "Dataset type"]].copy()

            feature_subsets = list(itertools.combinations(features, k))
            chunk_size = math.ceil(len(feature_subsets) / self.n_processes)
            process_args = []
            for i in range(self.n_processes):
                start = chunk_size * i
                end = chunk_size * (i + 1) if i < self.n_processes - 1 else len(feature_subsets)
                process_args.append((df_selected, feature_subsets[start:end]))

            start_time = time.time()

            with Pool(self.n_processes) as p:
                process_results = p.map(self.exhaustive_search_over_chunk, process_args, chunksize=1)

            df_n_k_results = pd.concat(process_results, axis=0)
            df_n_k_results["n"] = n
            df_n_k_results["k"] = k
            all_result_dfs.append(df_n_k_results)

            end_time = time.time()
            if self.verbose:
                print("Pipeline iteration finished in {} seconds for n={}, k={} (n_processes = {})".format(end_time - start_time, n, k, self.n_processes))

        return pd.concat(all_result_dfs, axis=0)

    def exhaustive_search_over_chunk(self, args):
        # args is a tuple, this is how multiprocessing works
        df_selected, feature_subsets = args

        results = []
        for features_subset in feature_subsets:
            features_subset = list(features_subset)  # Convert list to tuple for convinience

            # Extract training set
            df_train = df_selected.loc[df_selected["Dataset type"] == "Training", features_subset + ["Class"]]
            X_train = df_train.drop(columns=["Class"]).to_numpy()
            y_train = df_train["Class"].to_numpy()

            # Fit preprocessor and transform training set
            preprocessor = self.preprocessor(**self.preprocessor_kwargs)
            preprocessor.fit(X_train)
            X_train = preprocessor.transform(X_train)

            # Fit classifier with CV search of unknown parameters
            classifier = self.classifier(**self.classifier_kwargs)

            # TODO: seed as pipeline parameter
            splitter = StratifiedKFold(n_splits=self.classifier_CV_folds, shuffle=True, random_state=17)
            searcher = GridSearchCV(
                classifier,
                self.classifier_CV_ranges,
                scoring={s: make_scorer(self.scoring_functions[s]) for s in self.scoring_functions},
                cv=splitter,
                refit=False
            )
            searcher.fit(X_train, y_train)

            all_params = searcher.cv_results_["params"]
            mean_test_scorings = {s: searcher.cv_results_["mean_test_" + s] for s in self.scoring_functions}
            best_ind = np.argmax(mean_test_scorings[self.main_scoring_function])
            best_params = {param: all_params[best_ind][param] for param in all_params[best_ind]}

            # Refit classifier with best parameters
            classifier = self.classifier(**self.classifier_kwargs, **best_params)
            classifier.fit(X_train, y_train)

            item = {"Features subset": features_subset, "Scores": {}}
            filtration_passed = True
            for dataset, dataset_type in df_selected[["Dataset", "Dataset type"]].drop_duplicates().to_numpy():
                df_test = df_selected.loc[df_selected["Dataset"] == dataset, features_subset + ["Class"]]
                X_test = df_test.drop(columns=["Class"]).to_numpy()
                y_test = df_test["Class"].to_numpy()

                # Normalize dataset using preprocessor fit on training set
                X_test = preprocessor.transform(X_test)

                y_pred = classifier.predict(X_test)
                item["Scores"][dataset] = {}
                for s in self.scoring_functions:
                    item["Scores"][dataset][s] = self.scoring_functions[s](y_test, y_pred)

                if (
                    dataset_type in ["Training", "Filtration"] and
                    item["Scores"][dataset][self.main_scoring_function] < self.main_scoring_threshold
                ):
                    filtration_passed = False

            if filtration_passed:
                results.append(item)

        df_results = pd.DataFrame(
            columns=["{};{}".format(dataset, s) for dataset in natsorted(np.unique(df_selected["Dataset"])) for s in self.scoring_functions]
        )
        for item in results:
            index = ";".join(item["Features subset"])
            for dataset in item["Scores"]:
                for s in item["Scores"][dataset]:
                    df_results.loc[index, "{};{}".format(dataset, s)] = item["Scores"][dataset][s]

        return df_results


def feature_pre_selector_template(df, **kwargs):
    '''
    Input expression dataframe, return list of features
    TODO: special function which load genes from specified file
    '''
    pass


def feature_selector_template(df, n, **kwargs):
    '''
    Input expression dataframe and number n, return list
    of n selected features
    TODO: for supervised feature selection one should also pass
    subset of datasets for feature selection
    '''
    pass


class Preprocessor_template:
    '''
    This class should have three methods:
        -  __init__
        -  fit
        -  transform
    Any sklearn classifier preprocessor be suitable
    '''
    def __init__(self, **kwargs):
        pass

    def fit(self, X):
        pass

    def transform(self, X):
        pass


class Classifier_template:
    '''
    This class should have three methods:
        -  __init__
        -  fit
        -  predict
    Any sklearn classifier will be suitable
    '''
    def __init__(self, **kwargs):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass


if __name__ == "__main__":
    import sys
    df = pd.read_csv(sys.argv[1], sep="\t", index_col=0)
    #n_k = pd.DataFrame({"n": [10, 10], "k": [8, 9]})  # Several seconds test
    n_k = pd.read_csv("n_k.tsv", sep="\t")
    n_processes = 4

    import time
    for _, row in n_k.iterrows():
        n_k_subdf = pd.DataFrame({"n": [row["n"]], "k": [row["k"]]})

        start_time = time.time()
        pipeline = ExhaustivePipeline(df, n_k_subdf, n_processes=n_processes)
        results = pipeline.run()
        end_time = time.time()
        print("Pipeline finished in {} seconds for n={}, k={} (n_processes = {})".format(end_time - start_time, row["n"], row["k"], n_processes))
