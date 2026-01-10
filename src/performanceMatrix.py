import numpy as np

def annualized_returns(returns, period = 252):
    return np.mean(returns) * period

def annualized_volatility(returns, period = 252):
    return np.std(returns) * np.sqrt(period)

def sharpe_ratio(returns, risk_free_ratio = 0.05):
    excess = returns - risk_free_ratio/252
    return np.mean(excess) / np.std(excess) * np.sqrt(252)