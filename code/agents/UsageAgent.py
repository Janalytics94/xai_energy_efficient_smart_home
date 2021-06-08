#! /usr/bin/env python3

# Usage Agent
# ===============================================================================================
class Usage_Agent:
    import pandas as pd

    def __init__(self, input_df, device):
        self.input = input_df
        self.device = device

    # train test split
    # -------------------------------------------------------------------------------------------
    def train_test_split(self, df, date, train_start="2013-11-01"):
        select_vars = [
            self.device + "_usage",
            self.device + "_usage_lag_1",
            self.device + "_usage_lag_2",
            "active_last_2_days",
        ]
        df = df[select_vars]
        X_train = df.loc[train_start:date, df.columns != self.device + "_usage"]
        y_train = df.loc[train_start:date, df.columns == self.device + "_usage"]
        X_test = df.loc[date, df.columns != self.device + "_usage"]
        y_test = df.loc[date, df.columns == self.device + "_usage"]
        return X_train, y_train, X_test, y_test

    ################Mine##############################
    def skModels(self,model, X, y):
        # models we want to try
        names = ["knn", "linear svm", 
        "rbv svm", "gaussian process","descision tree", "random forest", 
        "nn", "ada boost","nb", "qda", 'logit']
        classifiers = [ KNeighborsClassifier(3),
                        SVC(kernel="linear", C=0.025),
                        SVC(gamma=2, C=1),
                        GaussianProcessClassifier(1.0 * RBF(1.0)),
                        DecisionTreeClassifier(max_depth=5),
                        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
                        MLPClassifier(alpha=1, max_iter=1000),
                        AdaBoostClassifier(),
                        GaussianNB(),
                        QuadraticDiscriminantAnalysis()]
        model_types = [type(classifier) for classifier in classifiers]
        if model in names:
            dict_of_classifiers = dict(zip(names, classifiers))
            return  dict_of_classifiers, model_types   
        else:
            raise InputError('Unknown model type.')

    def fit_skModels(self, model_type, X,y):
        if model_type in dict_of_classifers:
            fitted_model = dict_of_classifiers[model_type].fit(X,y)
            return fitted_model
        else:
            raise InputError('Unknown model type')
    
    def skModels_predict(self, model, X):
        import sklearn
        types = [sklearn.neighbors._classification.KNeighborsClassifier,
                sklearn.svm._classes.SVC,
                sklearn.svm._classes.SVC,
                sklearn.gaussian_process._gpc.GaussianProcessClassifier,
                sklearn.tree._classes.DecisionTreeClassifier,
                sklearn.ensemble._forest.RandomForestClassifier,
                sklearn.neural_network._multilayer_perceptron.MLPClassifier,
                sklearn.ensemble._weight_boosting.AdaBoostClassifier,
                sklearn.naive_bayes.GaussianNB,
                sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis]
        if type(model) in types:
            y_hat = model.predict(X)
        else:
            raise InputError('Unknown model type.')
        return y_hat
#############################################################################################################################
    
    
    # model training and evaluation
    # -------------------------------------------------------------------------------------------
    def fit_smLogit(self, X, y):
        import statsmodels.api as sm

        return sm.Logit(y, X).fit(disp=False)

    def fit(self, X, y, model_type):
        if model_type == "logit":
            model = self.fit_smLogit(X, y)
        else:
            raise InputError("Unknown model type.")
        return model

    def predict(self, model, X):
        import statsmodels
        import numpy as np

        X = np.array(X)

        if type(model) == statsmodels.discrete.discrete_model.BinaryResultsWrapper:
            y_hat = model.predict(X)
        else:
            raise InputError("Unknown model type.")
        return y_hat

    
    def auc(self, y_true, y_hat):
        import sklearn.metrics
        return sklearn.metrics.roc_auc_score(y_true, y_hat)
    
    def evaluate(
        self, df, model_type, train_start, predict_start="2014-01-01", predict_end=-1, return_errors=False
    ):
        import pandas as pd
        import numpy as np
        from tqdm import tqdm

        dates = pd.DataFrame(df.index)
        dates = dates.set_index(df.index)["Time"]
        predict_start = pd.to_datetime(predict_start)
        predict_end = (
            pd.to_datetime(dates.iloc[predict_end])
            if type(predict_end) == int
            else pd.to_datetime(predict_end)
        )
        dates = dates.loc[predict_start:predict_end]
        y_true = []
        y_hat_train = {}
        y_hat_test = []
        auc_train_dict = {}
        auc_test = []

        for date in tqdm(dates.index):
            errors = {}
            try:
                X_train, y_train, X_test, y_test = self.train_test_split(
                    df, date, train_start
                )
                # fit model
                model = self.fit_skModels(X_train, y_train, model_type)
                # predict
                y_hat_train.update({date: self.skModels_predict(model, X_train)})
                y_hat_test += list(self.skModels_predict(model, X_test))
                # evaluate train data
                auc_train_dict.update(
                    {date: self.auc(y_train, list(y_hat_train.values())[-1])}
                )
                y_true += list(y_test)
            except Exception as e:
                errors[date] = e

        auc_test = self.auc(y_true, y_hat_test)
        auc_train = np.mean(list(auc_train_dict.values()))

        if return_errors:
            return auc_train, auc_test, auc_train_dict, errors
        else:
            return auc_train, auc_test, auc_train_dict
        
    # pipeline function: predicting device usage
    # -------------------------------------------------------------------------------------------        
    def pipeline(self, df, date, model_type, train_start):
        X_train, y_train, X_test, y_test = self.train_test_split(df, date, train_start)
        model = self.fit_skModels(X_train, y_train, model_type)
        return self.skModels_predict(model, X_test)    

    
