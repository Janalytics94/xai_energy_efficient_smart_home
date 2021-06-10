#!/usr/bin/env python3
import pandas as pd
import numpy as np
from tqdm import tqdm

# Load Agent
# ===============================================================================================
class Load_Agent:
    def __init__(self, load_input_df):
        self.input = load_input_df

    # selecting the correct data, identifying device runs, creating load profiles
    # -------------------------------------------------------------------------------------------
    def prove_start_end_date(self, df, date):
        

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

        yesterday = (pd.to_datetime(date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        return df[:yesterday]

    def load_profile_raw(self, df, shiftable_devices):

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

        for app in df_hours.keys():
            for i in df_hours[app].index:
                for j in df_hours[app].columns:
                    if np.isnan(df_hours[app].loc[i, j]):
                        df_hours[app].loc[i, j:] = 0
        return df_hours

    def load_profile(self, df_hours, shiftable_devices):

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

