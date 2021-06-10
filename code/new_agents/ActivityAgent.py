#! /usr/bin/env python3
import pandas as pd
from helper_functions import Helper


# Activity Agent
# ===============================================================================================
class Activity_Agent:
    def __init__(self, activity_input_df):
        self.input = activity_input_df

    # train test split
    # -------------------------------------------------------------------------------------------
    def get_Xtest(self, df, date, time_delta="all", target="activity"):
        import pandas as pd
        from helper_functions import Helper

        helper = Helper()

        if time_delta == "all":
            output = df.loc[pd.to_datetime(date) :, df.columns != target]
        else:
            df = helper.get_timespan(df, date, time_delta)
            output = df.loc[:, df.columns != target]
        return output

    def get_ytest(self, df, date, time_delta="all", target="activity"):
        import pandas as pd
        from helper_functions import Helper

        helper = Helper()

        if time_delta == "all":
            output = df.loc[pd.to_datetime(date) :, target]
        else:
            output = helper.get_timespan(df, date, time_delta)[target]
        return output

    def get_Xtrain(self, df, date, start="2013-11-01", target="activity"):
        import pandas as pd

        if type(start) == int:
            start = pd.to_datetime(date) + pd.Timedelta(days=start)
            start = (
                pd.to_datetime("2013-11-01")
                if start < pd.to_datetime("2013-11-01")
                else start
            )
        else:
            start = pd.to_datetime(start)
        end = pd.to_datetime(date) + pd.Timedelta(seconds=-1)
        return df.loc[start:end, df.columns != target]

    def get_ytrain(self, df, date, start="2013-11-01", target="activity"):
        import pandas as pd

        if type(start) == int:
            start = pd.to_datetime(date) + pd.Timedelta(days=start)
            start = (
                pd.to_datetime("2013-11-01")
                if start < pd.to_datetime("2013-11-01")
                else start
            )
        else:
            start = pd.to_datetime(start)
        end = pd.to_datetime(date) + pd.Timedelta(seconds=-1)
        return df.loc[start:end, target]

    def train_test_split(
        self, df, date, train_start="2013-11-01", test_delta="all", target="activity"
    ):
        X_train = self.get_Xtrain(df, date, start=train_start, target=target)
        y_train = self.get_ytrain(df, date, start=train_start, target=target)
        X_test = self.get_Xtest(df, date, time_delta=test_delta, target=target)
        y_test = self.get_ytest(df, date, time_delta=test_delta, target=target)
        return X_train, y_train, X_test, y_test

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

        if type(model) == statsmodels.discrete.discrete_model.BinaryResultsWrapper:
            y_hat = model.predict(X)
        else:
            raise InputError("Unknown model type.")
        return y_hat

    ################ML Models ##############################
    def skModels(self,model):
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

    def fit_skModels(self,dict_of_classifiers, model_type, X,y):
        if model_type in dict_of_classifers:
            fitted_model = dict_of_classifiers[model_type].fit(X,y)
            return fitted_model
        else:
            raise InputError('Unknown model type')
    
    def skModels_predict(self, model, X):
        types = [sk.neighbors._classification.KNeighborsClassifier,
                sk.svm._classes.SVC,
                sk.svm._classes.SVC,
                sk.gaussian_process._gpc.GaussianProcessClassifier,
                sk.tree._classes.DecisionTreeClassifier,
                sk.ensemble._forest.RandomForestClassifier,
                sk.neural_network._multilayer_perceptron.MLPClassifier,
                sk.ensemble._weight_boosting.AdaBoostClassifier,
                sk.naive_bayes.GaussianNB,
                sk.discriminant_analysis.QuadraticDiscriminantAnalysis]
        if type(model) in types:
            y_hat = model.predict(X)
            #y_hat = pd.Series(y_hat, index=X_test.index)
        else:
            raise InputError('Unknown model type.')
        return y_hat
#############################################################################################################################

    def auc(self, y_true, y_hat):
        import sklearn.metrics

        return sklearn.metrics.roc_auc_score(y_true, y_hat)

    def plot_model_performance(self, auc_train, auc_test, ylim="default"):
        import matplotlib.pyplot as plt

        plt.plot(list(auc_train.keys()), list(auc_train.values()))
        plt.plot(list(auc_train.keys()), list(auc_test.values()))
        plt.xticks(list(auc_train.keys()), " ")
        plt.ylim(ylim) if ylim != "default" else None

    def evaluate(
        self, df, model_type, split_params, predict_start="2014-01-01", predict_end=-1, return_errors=False
    ):
        import pandas as pd
        import numpy as np
        from tqdm import tqdm

        dates = (
            pd.DataFrame(df.index)
            .set_index(df.index)["Time"]
            .apply(lambda date: str(date)[:10])
            .drop_duplicates()
        )
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

        for date in tqdm(dates):
            errors = {}
            try:
                X_train, y_train, X_test, y_test = self.train_test_split(
                    df, date, **split_params
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

    # pipeline function: predicting user activity
    # -------------------------------------------------------------------------------------------
    def pipeline(self, df, date, model_type, split_params):
        # train test split
        X_train, y_train, X_test, y_test = self.train_test_split(
            df, date, **split_params
        )

        # fit model
        model = self.fit_skModels(X_train, y_train, model_type)

        # predict
        return self.skModels_predict(model, X_test)
