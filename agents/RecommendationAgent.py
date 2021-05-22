# Load in all the Agents
import numpy as np
import pandas as pd

from agents.LoadAgent import Load_Agent
from agents.ActivityAgent import Activity_Agent 
from agents.UsageAgent import Usage_Agent 
from agents.PriceAgent import Price_Agent

#from agents import Activity_Agent, Usage_Agent, Load_Agent, Price_Agent
class Recommendation_Agent:

    def __init__(self, activity_input, usage_input, load_input, price_input, shiftable_devices):
        self.activity_input = activity_input
        self.usage_input = usage_input
        self.load_input = load_input
        self.price_input = price_input
        self.shiftable_devices = shiftable_devices

        self.Activity_Agent = Activity_Agent(activity_input)

        #create dicionnary with Usage_Agent for each device
        self.Usage_Agent = {name: Usage_Agent(usage_input , name)  for name in shiftable_devices}

        self.Load_Agent = Load_Agent(load_input)
        self.Price_Agent = Price_Agent(price_input)

    def electricity_prices_from_start_time(self, date):
        prices_48 = self.Price_Agent.return_day_ahead_prices(date)
        prices_from_start_time = pd.DataFrame()

        for i in range(24):
          prices_from_start_time["Price_at_H+"+ str(i)] = prices_48.shift(-i)

        #delete last 24 hours
        prices_from_start_time = prices_from_start_time[:-24]
        return prices_from_start_time

    def cost_by_starting_time(self, date, device):
        #get electriciy prices following every device starting hour with previously defined function
        prices = self.electricity_prices_from_start_time(date)

        #build up table with typical load profile repeated for every hour (see Load_Agent)
        device_load = self.Load_Agent.pipeline(self.load_input, date, self.shiftable_devices).loc[device]
        device_load = pd.concat([device_load] * 24, axis= 1)

        #multiply both tables and aggregate costs for each starting hour
        costs = np.array(prices)*np.array(device_load)
        costs = np.sum(costs, axis = 0)

        #return an array of size 24 containing the total cost at each staring hour.
        return costs

    def recommend_by_device(self, date, device, activity_prob_threshold, usage_prob_threshold):

        #add split params as input
        # IN PARTICULAR --> Specify date to start training
        split_params =  {'train_start': '2013-11-01', 'test_delta': {'days':1, 'seconds':-1}, 'target': 'activity'}

        #compute costs by launching time:
        costs = self.cost_by_starting_time(date, device)

        #compute activity
        activity_probs = self.Activity_Agent.pipeline(self.activity_input, date, 'logit', split_params)

        #set values above threshold to 1. Values below to Inf 
        #(vector will be multiplied by costs, so that hours of little activity likelihood get cost = Inf)
        activity_probs = np.where(activity_probs>= activity_prob_threshold, 1, float("Inf"))

        #add a flag in case all hours have likelihood smaller than threshold
        no_recommend_flag_activity = 0
        if np.min(activity_probs) == float("Inf"):
          no_recommend_flag_activity = 1

        # compute cheapest hour from likely ones
        best_hour = np.argmin(np.array(costs) * np.array(activity_probs))

        # compute likelihood of usage:
        usage_prob = self.Usage_Agent[device].pipeline(self.usage_input , date, "logit", split_params["train_start"]) 
        no_recommend_flag_usage = 0
        if usage_prob < usage_prob_threshold :
          no_recommend_flag_usage = 1

        return {"recommendation_date": [date], "device": [device] ,"best_launch_hour": [best_hour] , "no_recommend_flag_activity" : [no_recommend_flag_activity], "no_recommend_flag_usage" : [no_recommend_flag_usage] }

    def pipeline(self, date, activity_prob_threshold, usage_prob_threshold):
      recommendations_by_device = self.recommend_by_device(date, self.shiftable_devices[0], activity_prob_threshold, usage_prob_threshold)
      recommendations_table = pd.DataFrame.from_dict(recommendations_by_device)

      for device in self.shiftable_devices[1:]:
        recommendations_by_device = self.recommend_by_device(date, device, activity_prob_threshold, usage_prob_threshold)
        recommendations_table = recommendations_table.append(pd.DataFrame.from_dict(recommendations_by_device))
      return recommendations_table