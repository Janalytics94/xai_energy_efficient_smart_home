import pandas as pd
# Price Agent
class Price_Agent:
  def __init__(self, Prices_df):
    self.input = Prices_df

  def return_day_ahead_prices(self, Date):
    range = pd.date_range(start=Date, freq = "H", periods=48)
    prices = self.input.loc[range]
    
    return prices