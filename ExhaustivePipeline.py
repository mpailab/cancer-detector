import itertools


class ExhaustivePipeline:
    def __init__(
        self, df, n_k, n_threads,
        feature_pre_selector=None, feature_pre_selector_kwargs={},
        feature_selector, feature_selector_kwargs,
        preprocessor, preprocessor_kwargs,
        classifier, classifier_kwargs, classifier_CV_ranges, classifier_CV_folds,
        scoring_functions, main_scoring_function, main_scoring_threshold
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

        # TODO: add default values for some arguments

        self.df = df
        self.n_k = n_k
        self.n_threads = n_threads

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
            df_pre_selected = self.df[features + ["Class", "Dataset", "Dataset type"]].copy()
        else:
            df_pre_selected = self.df.copy()


        # Start iterating over n, k pairs
        for n, k in zip(self.n_k["n"], self.n_k["k"]):
            features = self.feature_selector(df_pre_selected, n, **self.feature_selector_kwargs)
            df_selected = df_pre_selected[features + ["Class", "Dataset", "Dataset type"]]

            # TODO: this loop should be run in multiple processes
            results = []
            for features_subset in itertools.combinations(features, k):
                # Extract training set
                df_train = df_selected.loc[df_selected["Dataset type"] == "Training", features_subset + ["Class"]]
                X_train = df_train.drop(columns=["Class"]).to_numpy()
                y_train = df_train["Class"].to_numpy()

                # Fit preprocessor and transform training set
                preprocessor = self.preprocessor(**self.preprocessor_kwargs)
                preprocessor.fit(X_train)
                X_train = preprocessor.transform(X_train)

                # Fit classifier with CV search of unknown parameters
                classifier = self.classifier(**classifier_kwargs)

                # TODO: seed as pipeline parameter
                splitter = StratifiedKFold(n_splits=self.classifier_CV_folds, shuffle=True, random_state=17)
                searcher = GridSearchCV(
                    classifier,
                    self.classifier_CV_ranges,
                    scoring=self.scoring_functions,
                    cv=splitter,
                    refit=False,
                    iid=False
                )
                searcher.fit(X_train, y_train)

                all_params = searcher.cv_results_["params"]
                mean_test_scorings = {s: searcher.cv_results_["mean_test_" + s] for s in self.scoring_functions}
                best_ind = np.argmax(mean_test_scorings[self.main_scoring_function])
                best_params = {param: all_params[max_ind][param] for param in all_params[max_ind]}

                # Refit classifier with best parameters
                classifier = self.classifier(**classifier_kwargs, **best_params)
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
                        dataset_type.isin(["Training", "Filtration"]) and
                        item["Scores"][dataset][self.main_scoring_function] < self.main_scoring_threshold
                    ):
                        filtration_passed = False

                if filtration_passed:
                    results.append(item)


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
