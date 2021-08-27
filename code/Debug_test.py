
# loading necessary libraries
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import os

from helper_functions import Helper
from agents import Preparation_Agent
from datetime import datetime

helper = Helper()

DATA_PATH = '../data/'

# load household data for Household 3
household = helper.load_household(DATA_PATH, 3, weather_sel=True)

threshold = 0.01
active_appliances = ['Toaster', 'Tumble Dryer', 'Dishwasher', 'Washing Machine','Television', 'Microwave', 'Kettle']
shiftable_devices = ['Tumble Dryer', 'Washing Machine', 'Dishwasher']

#activity params
truncation_params = {
    'features': 'all',
    'factor': 1.5,
    'verbose': 0
}

scale_params = {
    'features': 'all',
    'kind': 'MinMax',
    'verbose': 0
}

aggregate_params = {
    'resample_param': '60T'
}

activity_params = {
    'active_appliances': active_appliances,
    'threshold': threshold
}

time_params = {
    'features': ['hour', 'day_name']
}

activity_lag_params = {
    'features': ['activity'],
    'lags': [24, 48, 72]
}

activity_pipe_params = {
    'truncate': truncation_params,
    'scale': scale_params,
    'aggregate': aggregate_params,
    'activity': activity_params,
    'time': time_params,
    'activity_lag': activity_lag_params
}

#load agent
device_params = {
    'threshold': threshold
}

load_pipe_params = {
    'truncate': truncation_params,
    'scale': scale_params,
    'aggregate': aggregate_params,
    'shiftable_devices': shiftable_devices,
    'device': device_params
}

#usage agent

device = {
    'threshold' : threshold}

aggregate_params24_H = {
    'resample_param': '24H'
}

usage_pipe_params = {
    'truncate': truncation_params,
    'scale': scale_params,
    'activity': activity_params,
    'aggregate_hour': aggregate_params,
    'aggregate_day': aggregate_params24_H,
    'time': time_params,
    'activity_lag': activity_lag_params,
    'shiftable_devices' : shiftable_devices,
    'device': device
}

#%%

# calling the preparation pipeline
#prep = Preparation_Agent(household)
#activity_df = prep.pipeline_activity(household, activity_pipe_params)
#load_df, _, _ = prep.pipeline_load(household, load_pipe_params)
#usage_df = prep.pipeline_usage(household, usage_pipe_params)

# load price data
# FILE_PATH = '/content/drive/MyDrive/T4_Recommendation-system-for-demand-response-and-load-shifting/02_data/'
#price_df = helper.create_day_ahead_prices_df(DATA_PATH, 'Day-ahead Prices_201501010000-201601010000.csv')


#%%

#activity_df.to_pickle('../data/processed_pickle/activity_df.pkl')
#load_df,_, _ .to_pickle('../data/processed_pickle/load_df.pkl')
#usage_df.to_pickle('../data/processed_pickle/usage_df.pkl')
#price_df.to_pickle('../data/processed_pickle/price_df.pkl')

#%%

# Load pickle data
activity_df = pd.read_pickle('../data/processed_pickle/activity_df.pkl')
load_df = pd.read_pickle('../data/processed_pickle/load_df.pkl')
usage_df = pd.read_pickle('../data/processed_pickle/usage_df.pkl')
price_df = pd.read_pickle('../data/processed_pickle/price_df.pkl')

#%%

from agents import Activity_Agent, Usage_Agent, Load_Agent, Price_Agent, Recommendation_Agent, Explainability_Agent

from agents import Performance_Evaluation_Agent
import json
household_id = 3
model_type = "logit"
weather_sel = True
EXPORT_PATH = '../export/'
config = json.load(open(EXPORT_PATH + str(household_id) + '_' + str(model_type) + '_' + str(model_type) +'_' + str(weather_sel) +'_config.json', 'r'))
files = ['df.pkl', 'output.pkl']
files = [f"{EXPORT_PATH}{household_id}_{model_type}_{model_type}_{weather_sel}_{file}" for file in files]

# initializing the agent
evaluation = Performance_Evaluation_Agent(DATA_PATH,model_type='logit' ,config=config, load_data=True, load_files=files, weather_sel=weather_sel)
evaluation.init_agents()
#performance = Performance_Evaluation_Agent(DATA_PATH, "logit", config)


activity_threshold = 0.625
usage_threshold = 0.125
#df_rec = evaluation._get_recommendations(activity_threshold,usage_threshold)
#%%

df = evaluation.evaluate(activity_threshold=activity_threshold, usage_threshold=usage_threshold)
df
