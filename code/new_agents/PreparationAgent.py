#! usr/bin/env python3

from helper_functions import Helper
import pandas as pd



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