#! /usr/bin/env python3 

import pandas as pd
import pickle
import json
import yaml
import itertools
import numpy as np
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output
from helper_functions import Helper
from copy import deepcopy
from tqdm import tqdm

from ActivityAgent import Activity_Agent
from LoadAgent import Load_Agent
from PreparationAgent import Preparation_Agent
from UsageAgent import Usage_Agent
from PriceAgent import Price_Agent
from RecommendationAgent import Recommendation_Agent


# Evaluation Agent
# ===============================================================================================
class Evaluation_Agent:
    def __init__(self, DATA_PATH, config, load_data=True, load_files=None):
       

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
        
        if (self.load == None):
            self.init_agents()
        self._get_dates()
        self.config["load"] = {
            "shiftable_devices": deepcopy(self.config["user_input"]["shiftable_devices"])
        }
        
    def _get_default_usage_config(self):
        
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

        first_data = df.index.to_series()[0]
        return (first_data + pd.Timedelta("1D")).replace(hour=0, minute=0, second=0)

    def get_last_date(self, df):

        last_data = df.index.to_series()[-1]
        return (last_data - pd.Timedelta("1D")).replace(hour=23, minute=59, second=59)

    def get_min_start_date(self, df):
        df = df.dropna()
        return df.loc[df.index.hour == 0, :].index[0]

    def _get_dates(self):

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
        if np.isnan(hour):
            return np.nan
        else:
            dt = np.datetime64(date) + np.timedelta64(int(hour), "h")
            return self.activity.input.loc[dt, "activity"]

    def _get_starting_times(self, device):

        # extracts hours in which the device is turned on,
        # conditional on that the device was turned off the hour before
        times = self.df["load"][device].index.to_numpy()
        hour = self.df["load"][device].values
        before = np.insert(hour, 0, 0)[:-1]
        return times[(before == 0) & (hour != 0)]

    def _get_starting_hours(self, device, date):
  
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
        summary = {
            name: self._result_to_summary(result)
            for name, result in self.results.items()
        }
        return pd.DataFrame.from_dict(summary, orient="index")
    

    # evaluation: grid search and sensitivity
    # -------------------------------------------------------------------------------------------
    def grid_search(self, activity_thresholds, usage_thresholds):
        
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
        optimal_thresholds = self.get_optimal_thresholds()
        optimal_thresholds_index = self.thresholds_to_index()
        result = self.results_to_summary().loc[optimal_thresholds_index,:]
        result = result.append(pd.Series(optimal_thresholds))
        result.name = self.config['data']['household']
        return result