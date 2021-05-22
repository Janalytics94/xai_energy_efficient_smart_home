#! usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt

class Helper:
    # source: http://stackoverflow.com/questions/18256363/how-do-i-print-the-content-of-a-txt-file-in-python
    @staticmethod
    def read_txt(filename):
        fl = open(filename, 'r')
        print(fl.read())
        fl.close


    def get_timespan(self, df, start, timedelta_params):

        start = pd.to_datetime(start) if type(start) != type(pd.to_datetime('1970-01-01')) else start 
        end = start + pd.Timedelta(**timedelta_params)
        return df[start:end]


    @staticmethod
    def load_txt(filename):
        fl = open(filename, 'r')
        output = fl.read()
        fl.close
        return output


    def get_column_labels(self, filename):
        columns = {}
        readme = self.load_txt(filename)
        temp = readme[readme.find('\nHouse'):]

        for house in range(1, 22):
            cols = {}
            temp = readme[readme.find('\nHouse '+str(house)):]
    
            for idx in range(10):
                start = temp.find(str(idx)+'.')+2
                stop = temp.find(',') if temp.find(',') < temp.find('\n\t') else temp.find('\n\t')
                cols.update({'Appliance'+str(idx):temp[start:stop]})
                temp = temp[stop+1:]

            columns.update({house: cols})
        return columns


    def load_household(self, REFIT_dir, house_id):

        data_sets = {id:f'CLEAN_House{id}.csv' for id in range(1,22)}
        filename = REFIT_dir + data_sets[house_id]

        readme = REFIT_dir + 'REFIT_Readme.txt'
        columns = self.get_column_labels(readme)

        house = pd.read_csv(filename)
        house.rename(columns=columns[house_id], inplace=True)
        house.set_index(pd.DatetimeIndex(house['Time']), inplace=True)
        return house


    def aggregate(self, df, resample_param):
        return df.resample(resample_param).mean().copy()


    def plot_consumption(self, df, features='all', figsize='default', threshold=None, title='Consumption'):

        df = df.copy()
        features = [column for column in df.columns if column not in ['Unix', 'Issues']] if features == 'all' else features

        fig, ax = plt.subplots(figsize=figsize) if figsize != 'default' else plt.subplots()
        if threshold != None:
            df['threshold'] = [threshold]*df.shape[0]
            ax.plot(df['threshold'], color = 'tab:red')
        for feature in features:
            ax.plot(df[feature])
        ax.legend(['threshold'] + features) if threshold != None else ax.legend(features)
        ax.set_title(title);

    def create_day_ahead_prices_df(self, FILE_PATH, filename):
     
      electricity_prices1 = pd.read_csv(FILE_PATH + filename)
      electricity_prices1["MTU (UTC)"] = electricity_prices1["MTU (UTC)"].str.split(pat = "-", n = 0).str[0]
      electricity_prices1["MTU (UTC)"] = electricity_prices1["MTU (UTC)"].str.replace("2015", "2013")

      electricity_prices2 = pd.read_csv(FILE_PATH + filename)
      electricity_prices2["MTU (UTC)"] = electricity_prices2["MTU (UTC)"].str.split(pat = "-", n = 0).str[0]
      electricity_prices2["MTU (UTC)"] = electricity_prices2["MTU (UTC)"].str.replace("2015", "2014")

      electricity_prices3 = pd.read_csv(FILE_PATH + filename)
      electricity_prices3["MTU (UTC)"] =  electricity_prices3["MTU (UTC)"].str.split(pat = "-", n = 0).str[0]

      electricity_prices = pd.concat([electricity_prices1, electricity_prices2, electricity_prices3])
      electricity_prices.columns = ["Time", "Price"]
      electricity_prices = electricity_prices.set_index(pd.DatetimeIndex(electricity_prices['Time']), drop = True)
      electricity_prices = electricity_prices["Price"]
      
      return electricity_prices