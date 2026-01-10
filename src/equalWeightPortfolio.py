import numpy as np 
import pandas as pd

def equal_weight_port(returns_df):
    n_assets = returns_df.shape[1]
    weights = np.array([1/n_assets] * n_assets)
    portfolio_returns = returns_df.dot(weights)
    return portfolio_returns,weights