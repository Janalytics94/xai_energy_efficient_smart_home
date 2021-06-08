#!/usr/bin/env python3
import time
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from helper_functions import Helper

# More ML Models
import sklearn as sk 
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


# Preparation Agent
# ===============================================================================================
class Preparation_Agent:
    def __init__(self, REFIT_df):
        self.input = REFIT_df

    # stardard data preprocessing
    # -------------------------------------------------------------------------------------------
    def outlier_truncation(self, series, factor=1.5, verbose=0):
        

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr

        output = []
        counter = 0
        for item in (
            tqdm(series, desc=f"[outlier truncation: {series.name}]")
            if verbose != 0
            else series
        ):
            if item > upper_bound:
                output.append(int(upper_bound))
                counter += 1
            elif item < lower_bound:
                output.append(int(lower_bound))
                counter +=1
                
            else:
                output.append(item)
        print(
            f"[outlier truncation: {series.name}]: {counter} outliers were truncated."
        ) if verbose != 0 else None
        return output

    def truncate(self, df, features="all", factor=1.5, verbose=0):

        output = df.copy()
        features = (
            df.select_dtypes(include=["number"]).columns
            if features == "all"
            else features
        )

        for feature in features:
            time.sleep(0.2) if verbose != 0 else None
            row_nn = (
                df[feature] != 0
            )  # truncate only the values for which the device uses energy
            output.loc[row_nn, feature] = self.outlier_truncation(
                df.loc[row_nn, feature], factor=factor, verbose=verbose
            )  # Truncatation factor = 1.5 * IQR
            print("\n") if verbose != 0 else None
        return output

    def scale(self, df, features="all", kind="MinMax", verbose=0):
        output = df.copy()
        features = (
            df.select_dtypes(include=["number"]).columns
            if features == "all"
            else features
        )

        if kind == "MinMax":
            scaler = MinMaxScaler()
            output[features] = scaler.fit_transform(df[features])
            print("[MinMaxScaler] Finished scaling the data.") if verbose != 0 else None
        else:
            raise InputError("Chosen scaling method is not available.")
        return output

    # feature creation
    # -------------------------------------------------------------------------------------------
    def get_device_usage(self, df, device, threshold):
        return (df.loc[:, device] > threshold).astype("int")

    def get_last_usage(self, series):

        last_usage = []
        for idx in range(len(series)):
            shift = 1
            if pd.isna(series.shift(periods=1)[idx]):
                shift = None
            else:
                while series.shift(periods=shift)[idx] == 0:
                    shift += 1
            last_usage.append(shift)
        return last_usage

    def get_last_usages(self, df, features):

        output = pd.DataFrame()
        for feature in features:
            output["periods_since_last_" + feature] = self.get_last_usage(df[feature])
        output.set_index(df.index, inplace=True)
        return output

    def get_activity(self, df, active_appliances, threshold):

        active = pd.DataFrame(
            {appliance: df[appliance] > threshold for appliance in active_appliances}
        )
        return active.apply(any, axis=1).astype("int")

    def get_time_feature(self, df, features="all"):

        functions = {
            "hour": lambda df: df.index.hour,
            "day_of_week": lambda df: df.index.dayofweek,
            "day_name": lambda df: df.index.day_name().astype("category"),
            "month": lambda df: df.index.month,
            "month_name": lambda df: df.index.month_name().astype("category"),
        }
        if features == "all":
            output = pd.DataFrame(
                {function[0]: function[1](df) for function in functions.items()}
            )
        else:
            output = pd.DataFrame(
                {
                    function[0]: function[1](df)
                    for function in functions.items()
                    if function[0] in features
                }
            )
        output.set_index(df.index, inplace=True)
        return output

    def get_time_lags(self, df, features, lags):

        output = pd.DataFrame()
        for feature in features:
            for lag in lags:
                output[f"{feature}_lag_{lag}"] = df[feature].shift(periods=lag)
        return output

    # determining the optimal energy consumption threshold for target creation (usage, activity)
    # -------------------------------------------------------------------------------------------
    def visualize_threshold(self, df, threshold, appliances, figsize=(18, 5)):
        # data prep
        for appliance in appliances:
            df[appliance + "_usage"] = self.get_device_usage(df, appliance, threshold)
        df = df.join(self.get_time_feature(df))
        df["activity"] = self.get_activity(df, appliances, threshold)



        usage_cols = [column for column in df.columns if column.endswith("_usage")]
        columns = ["activity"] + usage_cols

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        # hour
        hour = df.groupby("hour").mean()[columns]
        hour.plot(ax=axes[0])
        axes[0].set_ylim(-0.1, 1.1)
        axes[0].set_title(f"[threshold: {round(threshold, 4)}] Activity ratio per hour")

        # week
        usage_cols = [column for column in df.columns if column.endswith("_usage")]
        week = df.groupby("day_name").mean()[columns]
        week = week.reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        week = week.rename(index=(lambda day: day[:3]))
        week.plot(ax=axes[1])
        axes[1].set_ylim(-0.1, 1.1)        
        axes[1].set_title(
            f"[threshold: {round(threshold, 4)}] Activity ratio per day of the week"
        )

        # month
        usage_cols = [column for column in df.columns if column.endswith("_usage")]
        month = df.groupby("month").mean()[columns]
        month.plot(ax=axes[2])
        axes[2].set_ylim(-0.1, 1.1)
        axes[2].set_title(
            f"[threshold: {round(threshold, 4)}] Activity ratio per month"
        )

    def validate_thresholds(self, df, thresholds, appliances, figsize=(18, 5)):

        for threshold in tqdm(thresholds):
            self.visualize_threshold(df, threshold, appliances, figsize)
        time.sleep(0.2)
        print("\n")
        
    # pipeline functions: preparing the input for the following agents
    # -------------------------------------------------------------------------------------------
    def pipeline_activity(self, df, params):
        
        helper = Helper()
        df = df.copy()
        output = pd.DataFrame()

        # Data cleaning
        df = self.truncate(
            df,
            **params["truncate"],
        )
        df = self.scale(df, **params["scale"])

        # Aggregate to hour level
        df = helper.aggregate(df, **params["aggregate"])

        # Activity feature
        output["activity"] = self.get_activity(df, **params["activity"])

        ## Time feature
        output = output.join(self.get_time_feature(df, **params["time"]))

        # Activity lags
        output = output.join(self.get_time_lags(output, **params["activity_lag"]))

        # Dummy coding
        output = pd.get_dummies(output, drop_first=True)

        return output

    def pipeline_load(self, df, params):
        from helper_functions import Helper
        import pandas as pd

        helper = Helper()
        df = df.copy()
        output = pd.DataFrame()

        # Data cleaning
        df = self.truncate(
            df,
            **params["truncate"],
        )
        scaled = self.scale(df, **params["scale"])

        # aggregate
        df = helper.aggregate(df, **params["aggregate"])
        scaled = helper.aggregate(scaled, **params["aggregate"])

        # Get device usage and transform to energy consumption
        for device in params["shiftable_devices"]:
            df[device + "_usage"] = self.get_device_usage(
                scaled, device, **params["device"]
            )
            output[device] = df.apply(
                lambda timestamp: timestamp[device] * timestamp[device + "_usage"],
                axis=1,
            )

        return output, scaled, df

    def pipeline_usage(self, df, params):
        helper = Helper()
        df = df.copy()
        output = pd.DataFrame()

        # Data cleaning
        df = self.truncate(
            df,
            **params["truncate"],
        )
        scaled = self.scale(df, **params["scale"])

        # Aggregate to hour level
        scaled = helper.aggregate(scaled, **params["aggregate_hour"])

        # Activity feature
        output["activity"] = self.get_activity(scaled, **params["activity"])

        # Get device usage and transform to energy consumption
        for device in params["shiftable_devices"]:
            output[device + "_usage"] = self.get_device_usage(
                scaled, device, **params["device"]
            )

        # aggregate and convert from mean to binary
        output = helper.aggregate(output, **params["aggregate_day"])
        output = output.apply(lambda x: (x > 0).astype("int"))

        # Last usage
        output = output.join(self.get_last_usages(output, output.columns))

        # Time features
        output = output.join(self.get_time_feature(output, **params["time"]))

        # lags
        output = output.join(
            self.get_time_lags(
                output,
                ["activity"]
                + [device + "_usage" for device in params["shiftable_devices"]],
                [1, 2, 3],
            )
        )
        output["active_last_2_days"] = (
            (output.activity_lag_1 == 1) | (output.activity_lag_2 == 1)
        ).astype("int")

        # dummy coding
        output = pd.get_dummies(output, drop_first=True)

        return output

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


# Load Agent
# ===============================================================================================
class Load_Agent:
    def __init__(self, load_input_df):
        self.input = load_input_df

    # selecting the correct data, identifying device runs, creating load profiles
    # -------------------------------------------------------------------------------------------
    def prove_start_end_date(self, df, date):
        import pandas as pd

        start_date = (df.index[0]).strftime("%Y-%m-%d")
        end_date = date

        if len(df[start_date]) < 24:
            start_date = (pd.to_datetime(start_date) + pd.Timedelta(days=1)).strftime(
                "%Y-%m-%d"
            )
            df = df[start_date:end_date]
        else:
            df = df[:end_date]

        if len(df[end_date]) < 24:
            end_new = (pd.to_datetime(end_date) - pd.Timedelta(days=1)).strftime(
                "%Y-%m-%d"
            )
            df = df[:end_new]
        else:
            df = df[:end_date]
        return df

    def df_yesterday_date(self, df, date):
        import pandas as pd

        yesterday = (pd.to_datetime(date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        return df[:yesterday]

    def load_profile_raw(self, df, shiftable_devices):
        import pandas as pd

        hours = []
        for hour in range(1, 25):
            hours.append("h" + str(hour))
        df_hours = {}

        for idx, appliance in enumerate(
            shiftable_devices
        ):  # delete enumerate if we do not need integers indexes of devices
            df_hours[appliance] = pd.DataFrame(index=None, columns=hours)
            column = df[appliance]

            for i in range(len(column)):

                if (i == 0) and (column[0] > 0):
                    df_hours[appliance].loc[0, "h" + str(1)] = column[0]

                elif (column[i - 1] == 0) and (column[i] > 0):
                    for j in range(0, 24):
                        if (i + j) < len(column):
                            if column[i + j] > 0:
                                df_hours[appliance].loc[i, "h" + str(j + 1)] = column[
                                    i + j
                                ]
        return df_hours

    def load_profile_cleaned(self, df_hours):
        import numpy as np

        for app in df_hours.keys():
            for i in df_hours[app].index:
                for j in df_hours[app].columns:
                    if np.isnan(df_hours[app].loc[i, j]):
                        df_hours[app].loc[i, j:] = 0
        return df_hours

    def load_profile(self, df_hours, shiftable_devices):
        import pandas as pd

        hours = df_hours[shiftable_devices[0]].columns
        loads = pd.DataFrame(columns=hours)

        for app in df_hours.keys():
            app_mean = df_hours[app].apply(lambda x: x.mean(), axis=0)
            for hour in app_mean.index:
                loads.loc[app, hour] = app_mean[hour]

        loads = loads.fillna(0)
        return loads

    # evaluating the performance of the load agent
    # -------------------------------------------------------------------------------------------
    def get_true_loads(self, shiftable_devices):
        true_loads = self.load_profile_raw(self.input, shiftable_devices)
        true_loads = self.load_profile_cleaned(true_loads)
        for device, loads in true_loads.items():
            true_loads[device].rename(
                index=dict(enumerate(self.input.index)), inplace=True
            )
        return true_loads

    def evaluate(self, shiftable_devices, metric="mse", aggregate=True, evaluation=False):
        from tqdm import tqdm
        import pandas as pd
        import numpy as np

        tqdm.pandas()

        # true_loads = self.get_true_loads(shiftable_devices)
        if metric == "mse":
            import sklearn.metrics

            metric = sklearn.metrics.mean_squared_error

        true_loads = self.get_true_loads(shiftable_devices)

        scores = {}
        if not evaluation:
            for device in shiftable_devices:
                scores[device] = true_loads[device].progress_apply(
                    lambda row: metric(
                        row.values,
                        self.pipeline(
                            self.input, str(row.name)[:10], [device]
                        ).values.reshape(
                            -1,
                        ),
                    ),
                    axis=1,
                )
        else:
            for device in shiftable_devices:
                scores[device] = {}
                for idx in tqdm(true_loads[device].index):
                    date = str(idx)[:10]
                    y_true = true_loads[device].loc[idx, :].values
                    try:
                        y_hat = (
                            evaluation[date]
                            .loc[device]
                            .values.reshape(
                                -1,
                            )
                        )
                    except KeyError:
                        try:
                            y_hat = self.pipeline(
                                self.input, date, [device]
                            ).values.reshape(
                                -1,
                            )
                        except:
                            y_hat = np.full(24, 0)
                    scores[device][idx] = metric(y_true, y_hat)
                scores[device] = pd.Series(scores[device])

        if aggregate:
            scores = {device: scores_df.mean() for device, scores_df in scores.items()}
        return scores    

    # pipeline function: creating typical load profiles
    # -------------------------------------------------------------------------------------------
    def pipeline(self, df, date, shiftable_devices):
        # kann mann dann self benutzen?
        df = self.prove_start_end_date(df, date)
        df = self.df_yesterday_date(df, date)
        df_hours = self.load_profile_raw(df, shiftable_devices)
        df_hours = self.load_profile_cleaned(df_hours)
        loads = self.load_profile(df_hours, shiftable_devices)
        return loads

# Price Agent
# ===============================================================================================================
class Price_Agent:
    def __init__(self, Prices_df):
        self.input = Prices_df

    # pipeline function: return day ahead prices
    # -------------------------------------------------------------------------------------------
    def return_day_ahead_prices(self, Date):
        import pandas as pd

        range = pd.date_range(start=Date, freq="H", periods=48)
        prices = self.input.loc[range]
        return prices

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

    

# Recommendation Agent
# ===============================================================================================
class Recommendation_Agent:
    def __init__(
        self, activity_input, usage_input, load_input, price_input, shiftable_devices
    ):
        self.activity_input = activity_input
        self.usage_input = usage_input
        self.load_input = load_input
        self.price_input = price_input
        self.shiftable_devices = shiftable_devices
        self.Activity_Agent = Activity_Agent(activity_input)
        # create dicionnary with Usage_Agent for each device
        self.Usage_Agent = {
            name: Usage_Agent(usage_input, name) for name in shiftable_devices
        }
        self.Load_Agent = Load_Agent(load_input)
        self.Price_Agent = Price_Agent(price_input)

    # calculating costs
    # -------------------------------------------------------------------------------------------
    def electricity_prices_from_start_time(self, date):
        import pandas as pd

        prices_48 = self.Price_Agent.return_day_ahead_prices(date)
        prices_from_start_time = pd.DataFrame()
        for i in range(24):
            prices_from_start_time["Price_at_H+" + str(i)] = prices_48.shift(-i)
        # delete last 24 hours
        prices_from_start_time = prices_from_start_time[:-24]
        return prices_from_start_time

    def cost_by_starting_time(self, date, device, evaluation=False):
        import numpy as np
        import pandas as pd

        # get electriciy prices following every device starting hour with previously defined function
        prices = self.electricity_prices_from_start_time(date)
        # build up table with typical load profile repeated for every hour (see Load_Agent)
        if not evaluation:
            device_load = self.Load_Agent.pipeline(
                self.load_input, date, self.shiftable_devices
            ).loc[device]
        else:
            # get device load for one date
            device_load = evaluation["load"][date].loc[device]
        device_load = pd.concat([device_load] * 24, axis=1)
        # multiply both tables and aggregate costs for each starting hour
        costs = np.array(prices) * np.array(device_load)
        costs = np.sum(costs, axis=0)
        # return an array of size 24 containing the total cost at each staring hour.
        return costs
    
    # creating recommendations
    # -------------------------------------------------------------------------------------------
    def recommend_by_device(
        self,
        date,
        device,
        activity_prob_threshold,
        usage_prob_threshold,
        evaluation=False,
    ):
        import numpy as np

        # add split params as input
        # IN PARTICULAR --> Specify date to start training
        split_params = {
            "train_start": "2013-11-01",
            "test_delta": {"days": 1, "seconds": -1},
            "target": "activity",
        }
        # compute costs by launching time:
        costs = self.cost_by_starting_time(date, device, evaluation=evaluation)
        # compute activity probabilities
        if not evaluation:
            activity_probs = self.Activity_Agent.pipeline(self.activity_input, date, "logit", split_params)
        else:
            # get activity probs for date
            activity_probs = evaluation["activity"][date]

        # set values above threshold to 1. Values below to Inf
        # (vector will be multiplied by costs, so that hours of little activity likelihood get cost = Inf)
        activity_probs = np.where(activity_probs >= activity_prob_threshold, 1, float("Inf"))

        # add a flag in case all hours have likelihood smaller than threshold
        no_recommend_flag_activity = 0
        if np.min(activity_probs) == float("Inf"):
            no_recommend_flag_activity = 1

        # compute cheapest hour from likely ones
        best_hour = np.argmin(np.array(costs) * np.array(activity_probs))

        # compute likelihood of usage:
        if not evaluation:
            usage_prob = self.Usage_Agent[device].pipeline(self.usage_input, date, "logit", split_params["train_start"])
        else:
            # get usage probs
            name = ("usage_"+ device.replace(" ", "_").replace("(", "").replace(")", "").lower())
            usage_prob = evaluation[name][date]

        no_recommend_flag_usage = 0
        if usage_prob < usage_prob_threshold:
            no_recommend_flag_usage = 1

        return {
            "recommendation_date": [date],
            "device": [device],
            "best_launch_hour": [best_hour],
            "no_recommend_flag_activity": [no_recommend_flag_activity],
            "no_recommend_flag_usage": [no_recommend_flag_usage],
            "recommendation": [
                best_hour
                if (no_recommend_flag_activity == 0 and no_recommend_flag_usage == 0)
                else np.nan
            ],
        }

    # vizualizing the recommendations
    # -------------------------------------------------------------------------------------------
    def recommendations_on_date_range(
        self, date_range, activity_prob_threshold=0.6, usage_prob_threshold=0.5
    ):
        import pandas as pd

        recommendations = []
        for date in date_range:
            recommendations.append(self.pipeline(date, activity_prob_threshold, usage_prob_threshold))
            output = pd.concat(recommendations)
        return output

    def visualize_recommendations_on_date_range(self, recs):
        import plotly.express as px
        import plotly.graph_objects as go

        fig = go.Figure()

        for device in recs["device"].unique():
            plot_device = recs[recs["device"] == device]
            fig.add_trace(
                go.Scatter(
                    x=plot_device["recommendation_date"],
                    y=plot_device["recommendation"],
                    mode="lines",
                    name=device,
                )
            )
        fig.show()

    def histogram_recommendation_hour(self, recs):
        import seaborn as sns

        ax = sns.displot(recs, x="recommendation", binwidth=1)
        ax.set(xlabel="Hour of Recommendation", ylabel="counts")
    
    # pipeline function: create recommendations
    # -------------------------------------------------------------------------------------------
    def pipeline(self, date, activity_prob_threshold, usage_prob_threshold, evaluation=False):
        import pandas as pd

        recommendations_by_device = self.recommend_by_device(
            date,
            self.shiftable_devices[0],
            activity_prob_threshold,
            usage_prob_threshold,
            evaluation=evaluation,
        )
        recommendations_table = pd.DataFrame.from_dict(recommendations_by_device)

        for device in self.shiftable_devices[1:]:
            recommendations_by_device = self.recommend_by_device(
                date,
                device,
                activity_prob_threshold,
                usage_prob_threshold,
                evaluation=evaluation,
            )
            recommendations_table = recommendations_table.append(
                pd.DataFrame.from_dict(recommendations_by_device)
            )
        return recommendations_table


# Evaluation Agent
# ===============================================================================================
class Evaluation_Agent:
    def __init__(self, DATA_PATH, config, load_data=True, load_files=None):
        import agents
        from helper_functions import Helper
        import pandas as pd

        helper = Helper()

        self.config = config
        self.preparation = (agents.Preparation_Agent(helper.load_household(DATA_PATH, config["data"]["household"]))
            if load_data
            else None
        )
        self.price = (
            agents.Price_Agent(helper.create_day_ahead_prices_df(DATA_PATH, "Day-ahead Prices_201501010000-201601010000.csv"))
            if load_data
            else None
        )
        self.activity = None
        self.load = None
        for device in self.config["user_input"]["shiftable_devices"]:
            name = ("usage_"+ device.replace(" ", "_").replace("(", "").replace(")", "").lower())
            exec(f"self.{name} = None")
        self.recommendation = None
        self.df = {}
        self.output = {}
        self.errors = {}
        self.agent_scores = {} 
        self.cold_start_scores = {}
        #self.true_loads = None
        self.results = {}
        self.cold_start_days = pd.DataFrame()
        if load_files != None:
            self.load_from_drive(load_files)

    # helper: loading and storing intermediary results and further helper
    # -------------------------------------------------------------------------------------------
    def _load_object(self, filename):
        import pickle
        import json
        import yaml

        # using a command dict as a if-list
        commands = {
            "pkl": f"pickle.load(open('{filename}', 'rb'))",
            "json": f"json.load(open('{filename}', 'r'))",
            "yaml": f"yaml.load(open('{filename}', 'r'), Loader = yaml.Loader)",
        }

        *_, name, ftype = filename.split(".")
        name = name[name.rfind("_") + 1 :]
        obj = eval(commands[ftype])
        self[name] = obj

    def load_from_drive(self, files):
        files = [files] if type(files) != list else files
        for filename in files:
            self._load_object(filename)

    def dump(self, EXPORT_PATH):
        import json
        import yaml
        import pickle

        # storing the current configuration
        json.dump(self.config, open(EXPORT_PATH + str(self.config["data"]["household"]) + "_config.json","w"), indent=4)

        # storing the prepared data
        if self.df != {}:
            pickle.dump(self.df, open(EXPORT_PATH + str(self.config["data"]["household"]) + "_df.pkl", "wb"))

        # storing the agents' output
        if self.output != {}:
            pickle.dump(self.output, open(EXPORT_PATH + str(self.config["data"]["household"]) + "_output.pkl", "wb"))
            
        # storing the results
        if self.results != {}:
            pickle.dump(self.results, open(EXPORT_PATH + str(self.config["data"]["household"]) + "_results.pkl", "wb"))
            
    def __getitem__(self, item):
        return eval(f"self.{item}")

    def __setitem__(self, key, value):
        exec(f"self.{key} = value")

    def _format_time(self, seconds):
        return "{:02.0f}".format(seconds // 60) + ":" + "{:02.0f}".format(seconds % 60)
    
    def _get_agent_names(self):
        devices = self.config["user_input"]["shiftable_devices"]
        names = ["activity", "load"] + ["usage_"+ str(device).replace(" ", "_").replace("(", "").replace(")", "").lower() for device in devices]
        return names

    
    # creating the default configuration
    # -------------------------------------------------------------------------------------------
    def get_default_config(self, agents):
        if type(agents) != list:
            agents = [agents]
        
        agents = [agent.lower() for agent in agents]
        for agent in agents:
            exec(f"self._get_default_{agent}_config()")     
            
    def _get_default_preparation_config(self):
        from copy import deepcopy

        # preparation
        self.config["preparation"] = {}
        ## preparation: activity agent
        self.config["preparation"]["activity"] = {
            "truncate": {"features": "all", "factor": 1.5, "verbose": 0},
            "scale": {"features": "all", "kind": "MinMax", "verbose": 0},
            "aggregate": {"resample_param": "60T"},
            "activity": {
                "active_appliances": deepcopy(self.config["user_input"]["active_appliances"]),
                "threshold": deepcopy(self.config["user_input"]["threshold"]),
            },
            "time": {"features": ["hour", "day_name"]},
            "activity_lag": {"features": ["activity"], "lags": [24, 48, 72]},
        }
        ## preparation: usage agent
        self.config["preparation"]["usage"] = {
            "truncate": {"features": "all", "factor": 1.5, "verbose": 0},
            "scale": {"features": "all", "kind": "MinMax", "verbose": 0},
            "activity": {
                "active_appliances": deepcopy(self.config["user_input"]["active_appliances"]),
                "threshold": deepcopy(self.config["user_input"]["threshold"]),
            },
            "aggregate_hour": {"resample_param": "60T"},
            "aggregate_day": {"resample_param": "24H"},
            "time": {"features": ["hour", "day_name"]},
            "shiftable_devices": deepcopy(self.config["user_input"]["shiftable_devices"]),
            "device": {"threshold": deepcopy(self.config["user_input"]["threshold"])},
        }
        ## preparation: load agent
        self.config["preparation"]["load"] = {
            "truncate": {"features": "all", "factor": 1.5, "verbose": 0},
            "scale": {"features": "all", "kind": "MinMax", "verbose": 0},
            "aggregate": {"resample_param": "60T"},
            "shiftable_devices": deepcopy(self.config["user_input"]["shiftable_devices"]),
            "device": {"threshold": deepcopy(self.config["user_input"]["threshold"])},
        }

    def _get_default_activity_config(self):
        from copy import deepcopy
        
        if (self.activity == None):
            self.init_agents()
        self._get_dates()
        self.config["activity"] = {
            "model_type": "logit",
            "split_params": {
                "train_start": deepcopy(self.config["data"]["start_dates"]["activity"]),
                "test_delta": {"days": 1, "seconds": -1},
                "target": "activity",
            },
        }
        
    def _get_default_load_config(self):
        from copy import deepcopy
        
        if (self.load == None):
            self.init_agents()
        self._get_dates()
        self.config["load"] = {
            "shiftable_devices": deepcopy(self.config["user_input"]["shiftable_devices"])
        }
        
    def _get_default_usage_config(self):
        from copy import deepcopy
        
        if (self.activity == None) | (self.load == None):
            self.init_agents()
        self._get_dates()
        self.config["usage"] = {
            "model_type": "logit",
            "train_start": deepcopy(self.config["data"]["start_dates"]["usage"]),
        }
        for device in self.config["user_input"]["shiftable_devices"]:
            name = ("usage_"+ device.replace(" ", "_").replace("(", "").replace(")", "").lower())
            self.config[name] = self.config["usage"]
            self.config["data"]["start_dates"][name] = self.config["data"]["start_dates"]["usage"]
            
    # extracting the available dates in the data
    def get_first_date(self, df):
        import pandas as pd

        first_data = df.index.to_series()[0]
        return (first_data + pd.Timedelta("1D")).replace(hour=0, minute=0, second=0)

    def get_last_date(self, df):
        import pandas as pd

        last_data = df.index.to_series()[-1]
        return (last_data - pd.Timedelta("1D")).replace(hour=23, minute=59, second=59)

    def get_min_start_date(self, df):
        df = df.dropna()
        return df.loc[df.index.hour == 0, :].index[0]

    def _get_dates(self):
        import numpy as np

        # first and last date in the data
        self.config["data"]["first_date"] = str(self.get_first_date(self.preparation.input))[:10]
        self.config["data"]["last_date"] = str(self.get_last_date(self.preparation.input))[:10]
        # start dates
        start_dates = {}
        for agent, data in self.df.items():
            start_dates[agent] = self.get_min_start_date(data)
        start_dates["combined"] = np.max(list(start_dates.values()))
        self.config["data"]["start_dates"] = {
            key: str(value)[:10] for key, value in start_dates.items()
        }


    # running the pipeline
    # -------------------------------------------------------------------------------------------
    def pipeline(self, agents, **kwargs):
        # converting single agent to list
        if type(agents) != list:
            agents = [agents]
 
        agents = [agent.lower() for agent in agents]
        
        if 'preparation' in agents:
            self._prepare(**kwargs)
        if 'activity' in agents:
            self._pipeline_activity_usage_load('activity', **kwargs)
        if 'usage' in agents:
            usage_agents = ["usage_"+ device.replace(" ", "_").replace("(", "").replace(")", "").lower() for device in self.config["user_input"]["shiftable_devices"]]
            for agent in usage_agents:
                self._pipeline_activity_usage_load(agent, **kwargs)
        if 'load' in agents:
            self._pipeline_activity_usage_load('load', **kwargs)
        if 'recommendation' in agents:
            self._get_recommendations(**kwargs)
            
    def init_agents(self):
        import agents

        # initialize the agents
        self.activity = agents.Activity_Agent(self.df["activity"])
        self.load = agents.Load_Agent(self.df["load"])

        # initialize usage agents for the shiftable devices: agent = usage_name
        for device in self.config["user_input"]["shiftable_devices"]:
            name = ("usage_"+ device.replace(" ", "_").replace("(", "").replace(")", "").lower())
            exec(f'self.{name} = Usage_Agent(self.df["usage"], "{device}")')
            self.df[name] = self.df["usage"]

        self.recommendation = agents.Recommendation_Agent(
            self.df["activity"],
            self.df["usage"],
            self.df["load"],
            self.price.input,
            self.config["user_input"]["shiftable_devices"],
        )
            
    def _prepare(self, agent="all"):
        lines = {
            "activity": 'self.df["activity"] = self.preparation.pipeline_activity(self.preparation.input, self.config["preparation"]["activity"])',
            "usage": 'self.df["usage"] = self.preparation.pipeline_usage(self.preparation.input, self.config["preparation"]["usage"])',
            "load": 'self.df["load"] ,_,_ = self.preparation.pipeline_load(self.preparation.input, self.config["preparation"]["load"])',
        }
        if agent == "all":
            for agent in ["activity", "usage", "load"]:
                exec(lines[agent])
                print(f"[evaluation agent] Finished preparing the data for the {agent} agent.")
        else:
            exec(lines[agent])
            print(f"[evaluation agent] Finished preparing the data for the {agent} agent.")

    def _pipeline_activity_usage_load(self, agent, verbose=1):
        import pandas as pd
        from IPython.display import clear_output
        import time

        self.output[agent] = {}
        self.errors[agent] = {}

        # init agents
        if (self.activity == None) | (self.load == None):
            self.init_agents()

        # determining the dates
        dates = self.df[agent].index.to_series()
        start = pd.to_datetime(self.config["data"]["start_dates"][agent])
        end = pd.to_datetime(self.config["data"]["last_date"]).replace(
            hour=23, minute=59, second=59
        )
        dates = dates[(dates >= start) & (dates <= end)].resample("1D").count()
        dates = [str(date)[:10] for date in list(dates.index)]

        # pipeline funtion
        start = time.time() if verbose >= 1 else None
        for date in dates:
            try:
                self.output[agent][date] = eval(f'self.{agent}.pipeline(self.{agent}.input, "{date}", **self.config["{agent}"])')
                # verbose
                if verbose >= 1:
                    clear_output(wait=True)
                    elapsed = time.time() - start
                    remaining = (elapsed / (len(dates)) * (len(dates) - (dates.index(date) + 1)))
                    print(f"agent:\t\t{agent}")
                    print(f"progress: \t{dates.index(date)+1}/{len(dates)}")
                    print(f"time:\t\t[{self._format_time(elapsed)}<{self._format_time(remaining)}]\n")
                    print(self.output[agent][date])
            except Exception as e:
                self.errors[agent][date] = type(e).__name__

    def _get_recommendations(
        self, activity_threshold, usage_threshold, dates: tuple = "all"
    ):
        import numpy as np
        from IPython.display import clear_output

        # determining dates
        start = (
            self.config["data"]["start_dates"]["combined"]
            if dates == "all"
            else dates[0]
        )
        end = self.config["data"]["last_date"] if dates == "all" else dates[1]
        dates = np.arange(
            np.datetime64(start),
            np.datetime64(end) + np.timedelta64(1, "D"),
            np.timedelta64(1, "D"),
        )
        dates = [str(date) for date in dates]

        # creating recommendations
        self.errors["recommendation"] = {}
        self.output["recommendation"] = {}
        for date in dates:
            try:
                self.output["recommendation"][date] = self.recommendation.pipeline(
                    date, activity_threshold, usage_threshold, evaluation=self.output
                )
            except Exception as e:
                self.errors["recommendation"][date] = e

        # merging the recommendations into one dataframe
        df = list(self.output["recommendation"].values())[0]

        for idx in range(1, len(self.output["recommendation"].values())):
            df = df.append(list(self.output["recommendation"].values())[idx])
        df.set_index("recommendation_date", inplace=True)
        self.output["recommendation"] = df
        clear_output()

        
    # individual agent scores
    # -------------------------------------------------------------------------------------------
    def get_agent_scores(self):
        scores = {}
        scores['activity_auc'] = None
        scores['usage_auc'] = {}
        scores['load_mse'] = {}

        agents = self._get_agent_names()
        for agent in agents:
            agent_type = agent.split('_')[0]
    
            if agent_type == 'activity':
                _, auc_test, _ = self[agent].evaluate(self[agent].input, **self.config[agent])
                scores['activity_auc'] = auc_test
            if agent_type == 'usage':
                _, auc_test, _ = self[agent].evaluate(self[agent].input, **self.config[agent])
                scores['usage_auc'][self[agent].device] = auc_test
            if agent_type == 'load':
                try:
                    scores['load_mse'] = self.load.evaluate(**self.config['load'], evaluation=self.output['load'])
                except KeyError:
                    scores['load_mse'] = self.load.evaluate(**self.config['load'])
        self.agent_scores = scores
        return scores
    
    def agent_scores_to_summary(self, scores='default'):
        import pandas as pd

        if scores == 'default':
            scores = self.agent_scores

        summary = {}
        summary['activity_auc'] = pd.DataFrame()
        summary['usage_auc'] = pd.DataFrame()
        summary['load_mse'] = pd.DataFrame()

        household_id = self.config['data']['household']
        devices = self.config['user_input']['shiftable_devices']

        # activity 
        summary['activity_auc'].loc[household_id, '-'] = scores['activity_auc']
        # usage
        i = 0
        for device in devices:
            summary['usage_auc'].loc[household_id, i] = scores['usage_auc'][device]
            i += 1
        #load
        i = 0
        for device in devices:
            summary['load_mse'].loc[household_id, i] = scores['load_mse'][device]
            i += 1
        
        summary['activity_auc'].index.name = 'household'
        summary['usage_auc'].index.name = 'household'
        summary['load_mse'].index.name = 'household'
        summary['usage_auc'].columns.name = 'device'
        summary['load_mse'].columns.name = 'device'
        return summary
    
    
    # cold start: predict on all data
    # -------------------------------------------------------------------------------------------
    def predict_all(self, agent, **kwargs):
        agent_type = agent.split("_")[0]
        return eval(f"self._predict_all_{agent_type}(agent, **kwargs)")
    
    def _predict_all_load(self, agent, device):
        y_hat = {
            date: profiles.loc[device, :]
            for date, profiles in self.output[agent].items()
        }
        return y_hat
    
    def _predict_all_activity(self, agent):
        return self._predict_all_activity_usage(agent)

    def _predict_all_usage(self, agent):
        return self._predict_all_activity_usage(agent)

    def _predict_all_activity_usage(self, agent):
        import pandas as pd
        import numpy as np

        y_hat = {}
        # intitializing the error dict
        try:
            self.errors["evaluation"]
        except KeyError:
            self.errors["evaluation"] = {}

        try:
            self.errors["evaluation"][agent]
        except KeyError:
            self.errors["evaluation"][agent] = {}

        # determining the dates
        dates = np.arange(
            np.datetime64(self.config["data"]["start_dates"][agent]),
            np.datetime64(self.config["data"]["last_date"]) + np.timedelta64(1, "D"),
            np.timedelta64(1, "D"),
        )
        start = dates[0]
        end = dates[-1] + pd.Timedelta(days=1, seconds=-1)

        # creating X_test
        X_test, _, _, _ = self[agent].train_test_split(
            self[agent].input,
            dates[-1] + np.timedelta64(1, "D"),
            train_start=self.config["data"]["start_dates"][agent],
        )

        # creating predictions
        for date in dates:
            X_train, y_train, _, _ = self[agent].train_test_split(
                self[agent].input,
                date,
                train_start=self.config["data"]["start_dates"][agent],
            )
            try:
                model = self[agent].fit(X_train, y_train, "logit")
                y_hat[date] = self[agent].predict(model, X_test)
            except Exception as e:
                self.errors["evaluation"][agent][date] = type(e).__name__
        return y_hat


    # cold start: calculate cold start scores
    # -------------------------------------------------------------------------------------------
    def get_cold_start_scores(self, fn: dict = "default"):
        from IPython.display import clear_output

        scores = {}
        fn = {} if fn == "default" else fn
        
        # activity-agent
        scores["activity"] = self._get_cold_start_score("activity", fn=fn.get("activity", "default"))
        clear_output()

        for device in self.config["user_input"]["shiftable_devices"]:
            name = device.replace(" ", "_").replace("(", "").replace(")", "").lower()
            # usage agent
            scores["usage_" + name] = self._get_cold_start_score("usage_" + name, fn=fn.get("usage", "default"))
            # load agent
            scores["load_" + name] = self._get_cold_start_score("load", fn=fn.get("load", "default"), device=device)
            clear_output()
        self.cold_start_scores = scores
    
    def _get_cold_start_score(self, agent, fn="default", **kwargs):
        import sklearn.metrics
        import numpy as np

        agent_type = agent.split("_")[0]
        # specifying the correct score function
        fn_dict = {
            "activity": f"self.{agent}.auc",
            "usage": f"self.{agent}.auc",
            "load": "sklearn.metrics.mean_squared_error",
        }
        fn = eval(fn_dict[agent_type]) if fn == "default" else fn

        # specifying the correct y_true, y_hat
        y_dict = {
            "activity": "self[agent].train_test_split(self[agent].input, date=np.datetime64(self.config['data']['last_date'])+np.timedelta64(1, 'D'), train_start=self.config['data']['start_dates'][agent])",
            "usage": "self[agent].train_test_split(self[agent].input, date=np.datetime64(self.config['data']['last_date'])+np.timedelta64(1, 'D'), train_start=self.config['data']['start_dates'][agent])",
            "load": "list(self.output['load'].values())[-1].loc[kwargs['device'], :]",
        }
        y_true = eval(y_dict[agent_type])
        y_true = y_true if agent_type == "load" else y_true[1]
        y_hat = self.predict_all(agent, **kwargs)

        # calculating the scores
        scores = {}
        for date, pred in y_hat.items():
            scores[date] = fn(y_true, pred)
        return scores

    def cold_start_scores_to_df(self):
        import pandas as pd
        import numpy as np

        scores_df = pd.DataFrame()
        # convert dicts into dataframe
        for key in self.cold_start_scores.keys():
            for date, score in self.cold_start_scores[key].items():
                scores_df.loc[str(date), key] = score

        # sort the dataframe
        cols = (
            ["activity"]
            + [col for col in scores_df if col.startswith("usage")]
            + [col for col in scores_df if col.startswith("load")]
        )
        scores_df.index = scores_df.index.map(np.datetime64)
        scores_df = scores_df[cols].sort_index()
        return scores_df
    
    def get_cold_start_days(self, tolerance_values):
        import pandas as pd

        self.cold_start_days = pd.DataFrame({"tolerance": []}).set_index("tolerance")
        scores_df = self.cold_start_scores_to_df()
        tolerance_fn = {
            "activity": "scores_df[agent].max() * (1 - tolerance[agent_type])",
            "usage": "scores_df[agent].max() * (1 - tolerance[agent_type])",
            "load": "tolerance['load']",
        }

        # agent coldstart days
        for tolerance in tolerance_values:
            tolerance = {"activity": tolerance, "usage": tolerance, "load": tolerance}

            for agent in scores_df.columns:
                agent_type = agent.split("_")[0]

                done = False
                day = 0
                while not done:
                    day += 1
                    tolerance_value = eval(tolerance_fn[agent_type])
                    if agent_type == "load":
                        done = all(scores_df[agent].values[day - 1 :] < tolerance_value)
                    else:
                        done = all(scores_df[agent].values[day - 1 :] > tolerance_value)
                self.cold_start_days.loc[tolerance[agent_type], agent] = day
        # framework cold start days
        self.cold_start_days['framework'] = self.cold_start_days.max(axis=1)
    
    
    def cold_start_to_summary(self, tolerance_values='all'):
        import pandas as pd
        
        if tolerance_values == 'all':
            tolerance_values = list(self.cold_start_days.index)
        
        household_id = self.config['data']['household']
        devices = self.config['user_input']['shiftable_devices']

        summary = {}
        summary['activity'] = {}
        summary['usage'] = {}
        summary['load'] = {}
        summary['framework'] = {}

        # activity agent
        summary['activity']['-'] = {}  # '-': placeholder for device
        summary['activity']['-'][household_id] = self.cold_start_days['activity'][tolerance_values].astype(int).to_list()
        # usage agent
        i = 0
        for device in devices:
            name = 'usage_' + device.replace(" ", "_").replace("(", "").replace(")", "").lower()
            summary['usage'][i] = {}
            summary['usage'][i][household_id] = self.cold_start_days[name][tolerance_values].astype(int).to_list()
            i += 1

        # load agent
        i = 0
        for device in devices:
            name = 'load_' + device.replace(" ", "_").replace("(", "").replace(")", "").lower()
            summary['load'][i] = {}
            summary['load'][i][household_id] = self.cold_start_days[name][tolerance_values].astype(int).to_list()
            i += 1
    
        # framework
        summary['framework']['-'] = {}  # '-': placeholder for device
        summary['framework']['-'][household_id] = self.cold_start_days['framework'][tolerance_values].astype(int).to_list()

        # converting the format
        for key, value in summary.items():
            summary[key] = pd.DataFrame(value)
            summary[key].columns.name = 'device'
            summary[key].index.name = 'household'
        return summary
    
    # cold start: visualizations
    # -------------------------------------------------------------------------------------------
    def _plot_axs(self, axs, y, x=None, legend=None, **kwargs):
        axs.plot(x, y) if x != None else axs.plot(y)
        axs.set(**kwargs)
        axs.legend(legend) if legend != None else None

    def visualize_cold_start(self, metrics_name: dict, tolerance: dict=None, figsize=(18, 5)):
        import matplotlib.pyplot as plt

        scores_df = self.cold_start_scores_to_df()
        fig, axs = plt.subplots(1, 3, figsize=figsize)

        # activity
        self._plot_axs(
            axs[0],
            x=range(1, scores_df.shape[0] + 1),
            y=scores_df["activity"],
            title=f"[activity] {metrics_name['activity']}",
        )
        legend = ['activity']
        if tolerance != None: 
            tolerance_value = scores_df["activity"].max() * (1 - tolerance["activity"])
            color = axs[0].lines[-1].get_color()
            axs[0].plot([tolerance_value] * scores_df.shape[0], "--", c=color)
            legend.append([f"tolerance@{tolerance['activity']}"])
        axs[0].legend(legend)
        axs[0].set_xlabel("days")

        # usage
        usage_agents = [agent for agent in scores_df.columns if agent.find("usage") != -1]
        legend = []
        for agent in usage_agents:
            self._plot_axs(axs[1],
                x=range(1, scores_df.shape[0] + 1),
                y=scores_df[agent],
                title=f"[usage] {metrics_name['usage']}",
            )
            legend += [agent]
            if tolerance != None:
                tolerance_value = scores_df[agent].max() * (1 - tolerance["usage"])
                color = axs[1].lines[-1].get_color()
                axs[1].plot([tolerance_value] * scores_df.shape[0], "--", c=color)
                legend += [f"tolerance_{agent.replace('usage_', '')}@{tolerance['usage']}"]
        axs[1].legend(legend)
        axs[1].set_xlabel("days")

        # load
        load_agents = [agent for agent in scores_df.columns if agent.find("load") != -1]
        legend = []
        for agent in load_agents:
            self._plot_axs(
                axs[2],
                x=range(1, scores_df.shape[0] + 1),
                y=scores_df[agent],
                title=f"[load] {metrics_name['load']}",
            )
            legend += [agent]
        if tolerance != None:
            axs[2].plot([tolerance["load"]] * scores_df.shape[0], "--", c="black")
            legend += [f"tolerance@{tolerance['load']}"]
        axs[2].legend(legend)
        axs[2].set_xlabel("days")

    
    # evaluation: calculate costs per device run
    # -------------------------------------------------------------------------------------------
    def calculate_cost(self, date, hour, load):
        import numpy as np

        if np.isnan(hour):
            return np.nan
        else:
            price_idx = self.price.input.index.values
            prices = self.price.input.values

            dt = np.datetime64(date) + np.timedelta64(int(hour), "h")
            # getting the correct position for the load in the load array
            i = np.where(price_idx == dt)[0][0]

            # reshaping the load array and calculating the costs
            before = np.zeros(i)
            after = np.zeros(prices.shape[0] - load.shape[0] - before.shape[0])
            load = np.hstack([before, load, after])
            return np.dot(load, prices)

    def _get_usage(self, device, date):
        return self.df["usage"].loc[date, device + "_usage"]

    def _get_activity(self, date, hour):
        import numpy as np

        if np.isnan(hour):
            return np.nan
        else:
            dt = np.datetime64(date) + np.timedelta64(int(hour), "h")
            return self.activity.input.loc[dt, "activity"]

    def _get_starting_times(self, device):
        import numpy as np

        # extracts hours in which the device is turned on,
        # conditional on that the device was turned off the hour before
        times = self.df["load"][device].index.to_numpy()
        hour = self.df["load"][device].values
        before = np.insert(hour, 0, 0)[:-1]
        return times[(before == 0) & (hour != 0)]

    def _get_starting_hours(self, device, date):
        import numpy as np
        import pandas as pd

        times = self._get_starting_times(device)
        date = np.datetime64(date) if type(date) != np.datetime64 else date
        times = times[(times >= date) & (times < date + np.timedelta64(1, "D"))]
        hours = (
            pd.Series(times).apply(lambda x: x.hour).to_numpy()
            if times.shape[0] != 0
            else np.nan
        )
        return hours

    def _get_load(self, true_loads, device, date, hour):
        import numpy as np

        try:
            dt = np.datetime64(date) + np.timedelta64(hour, "h")
        # if hour == NaN, return zero load profile
        except ValueError:
            return np.zeros(24)
        try:
            return true_loads[device].loc[dt].values
        except KeyError as ke:
            # return a zero load profile if the datetime index was not found
            if str(ke).split("(")[0] == "numpy.datetime64":
                return np.zeros(24)
            # in any other case raise the key error
            else:
                raise ke


    # evaluation: performance metrics
    # -------------------------------------------------------------------------------------------
    def evaluate(self, activity_threshold, usage_threshold):
        name = f"activity: {activity_threshold}; usage: {usage_threshold}"
        #self._get_recommendations(activity_threshold, usage_threshold)
        self.pipeline('recommendation', activity_threshold=activity_threshold, usage_threshold=usage_threshold, dates='all')
        self.results[name] = self._evaluate()

    def _evaluate(self):
        import numpy as np

        df = self.output["recommendation"].copy()
        
        # usage and activity target
        df["usage_true"] = df.apply(lambda row: self._get_usage(row["device"], row.name), axis=1)
        df["activity_true"] = df.apply(lambda row: self._get_activity(row.name, row["recommendation"]), axis=1)
        df["acceptable"] = df["usage_true"] * df["activity_true"]

        # starting times
        df["starting_times"] = df.apply(
            lambda row: self._get_starting_hours(row["device"], row.name), axis=1
        )
        df["relevant_start"] = abs(df["starting_times"] - df["recommendation"])
        df.loc[df["starting_times"].notna(), "relevant_start"] = df[
            df["starting_times"].notna()
        ].apply(
            lambda row: row["starting_times"][np.argmin(row["relevant_start"])], axis=1
        )

        # actual loads
        true_loads = self.load.get_true_loads(self.config["user_input"]["shiftable_devices"])
        df["load"] = df.apply(lambda row: self._get_load(true_loads, row["device"], row.name, row["relevant_start"]), axis=1)

        # calculating costs
        df["cost_no_recommendation"] = df.apply(lambda row: self.calculate_cost(row.name, row["relevant_start"], row["load"]), axis=1)
        df["cost_recommendation"] = df.apply(lambda row: self.calculate_cost(row.name, row["recommendation"], row["load"]),axis=1)
        df["savings"] = df["cost_no_recommendation"] - df["cost_recommendation"]
        df["relative_savings"] = df["savings"] / df["cost_no_recommendation"]

        return df[
            [
                "device",
                "recommendation",
                "acceptable",
                "relevant_start",
                "cost_no_recommendation",
                "cost_recommendation",
                "savings",
                "relative_savings",
            ]
        ]

    def _result_to_summary(self, result):
        return {
            "n_recommendations": result["recommendation"].count(),
            "acceptable": result["acceptable"].mean(),
            "total_savings": (result["acceptable"] * result["savings"]).sum(),
            "relative_savings_mean": result["relative_savings"].mean(),
            "relative_savings_median": result["relative_savings"].median(),
        }

    def results_to_summary(self):
        import pandas as pd

        summary = {
            name: self._result_to_summary(result)
            for name, result in self.results.items()
        }
        return pd.DataFrame.from_dict(summary, orient="index")
    

    # evaluation: grid search and sensitivity
    # -------------------------------------------------------------------------------------------
    def grid_search(self, activity_thresholds, usage_thresholds):
        import itertools
        from tqdm import tqdm
        
        # updating the config 
        try:
            self.config['evaluation']
        except:
            self.config['evaluation'] = {}
        
        self.config['evaluation']['grid_search'] = {}
        self.config['evaluation']['grid_search']['activity_thresholds'] = list(activity_thresholds)
        self.config['evaluation']['grid_search']['usage_thresholds'] = list(usage_thresholds)
        
        # testing candidate thresholds
        iterator = itertools.product(activity_thresholds, usage_thresholds)
        for thresholds in tqdm(list(iterator)):
            self.evaluate(thresholds[0], thresholds[1])

    def get_sensitivity(self, target):
        import pandas as pd

        df = self.results_to_summary()
        sensitivity = pd.DataFrame()
        for threshold_name in df.index:
            thresholds = threshold_name.split("; ")
            activity_threshold, usage_threshold = [th.split(": ")[1] for th in thresholds]
            sensitivity.loc[activity_threshold, usage_threshold] = df.loc[threshold_name, target]
        # sort and name rows and columns
        sensitivity = sensitivity.loc[sorted(sensitivity.index), :]
        sensitivity = sensitivity.loc[:, sorted(sensitivity.columns)]
        sensitivity.index.name = "activity_threshold"
        sensitivity.columns.name = "usage_threshold"
        return sensitivity
    
    def get_optimal_thresholds(self):
        df = self.results_to_summary()
        result = df.sort_values(by='total_savings').iloc[-1, :]
        thresholds = result.name.split('; ')
        thresholds = [threshold.split(': ') for threshold in thresholds]
        thresholds = {f"{threshold}_threshold": value for threshold, value in thresholds}
        self.config['evaluation']['grid_search']['optimal_thresholds'] = thresholds
        return thresholds
        
    def thresholds_to_index(self, activity_threshold='optimal', usage_threshold='optimal'):
        if activity_threshold == 'optimal':
            activity_threshold = self.config['evaluation']['grid_search']['optimal_thresholds']['activity_threshold']
        if usage_threshold == 'optimal':
            usage_threshold = self.config['evaluation']['grid_search']['optimal_thresholds']['usage_threshold']
        return f"activity: {activity_threshold}; usage: {usage_threshold}"
    
    def optimal_result_to_summary(self):
        import pandas as pd
        optimal_thresholds = self.get_optimal_thresholds()
        optimal_thresholds_index = self.thresholds_to_index()
        result = self.results_to_summary().loc[optimal_thresholds_index,:]
        result = result.append(pd.Series(optimal_thresholds))
        result.name = self.config['data']['household']
        return result