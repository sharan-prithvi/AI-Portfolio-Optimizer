import pandas as pd

def compute_returns(prices_df):
    returns = prices_df.pct_change().dropna()
    return returns