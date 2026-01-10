import numpy as np    
def mean_historical_returns(returns_df, periods=252):
    return returns_df.mean() * periods