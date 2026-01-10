import numpy as np

def annualized_returns(returns, period = 252):
    return np.mean(returns) * period

def annualized_volatility(returns, period = 252):
    return np.std(returns) * np.sqrt(period)

def sharpe_ratio(returns, risk_free_ratio = 0.05):
    excess = returns - risk_free_ratio/252
    return np.mean(excess) / np.std(excess) * np.sqrt(252)

def cumulative_returns(returns):
    return (1 + returns).cumprod()

def drawdown(cumulative_returns):
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    return drawdown

def max_drawdown(drawdown_series):
    return drawdown_series.min()

def downside_deviation(returns, risk_free_ratio = 0.05):
    rf_daily = risk_free_ratio/252
    downside_returns = returns[returns < rf_daily]
    return np.std(downside_returns) * np.sqrt(252)

def sortino_ratio(returns, risk_free_ratio =0.05):
    downside_dev = downside_deviation(returns, risk_free_ratio)
    excess_return = np.mean(returns - risk_free_ratio / 252) *252
    return excess_return / downside_dev