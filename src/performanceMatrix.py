import numpy as np
import pandas as pd

def annualized_returns(returns, period = 252):
    return np.mean(returns) * period

def annualized_volatility(returns, period = 252):
    return np.std(returns) * np.sqrt(period)

def sharpe_ratio(returns, risk_free_ratio = 0.05):
    excess = returns - risk_free_ratio/252
    return np.mean(excess) / np.std(excess) * np.sqrt(252)

def cumulative_returns(returns: pd.Series) -> pd.Series:
    return (1 + returns).cumprod()

def drawdown(cumulative_returns: pd.Series) -> pd.Series:
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    return drawdown

def max_drawdown(drawdown_series):
    return drawdown_series.min()

def max_drawdown_CC(returns: pd.Series) -> float:
    cum_returns = cumulative_returns(returns)
    dd = drawdown(cum_returns)
    return dd.min()

def downside_deviation(returns, risk_free_ratio = 0.05):
    rf_daily = risk_free_ratio/252
    downside_returns = returns[returns < rf_daily]
    return np.std(downside_returns) * np.sqrt(252)

def downside_deviation_CC(returns: pd.Series, target: float =0.0) -> float:
    downside = np.minimum(returns - target, 0)
    return np.sqrt(np.mean(downside ** 2))

def sortino_ratio(returns, risk_free_ratio =0.05):
    downside_dev = downside_deviation(returns, risk_free_ratio)
    excess_return = np.mean(returns - risk_free_ratio / 252) *252
    return excess_return / downside_dev


def value_at_risk(
    returns: pd.Series,
    confidence: float = 0.95
) -> float:
    return np.percentile(returns, (1-confidence) * 100)

def conditional_var(
    returns: pd.Series,
    confidence: float = 0.95
) -> float:
    var = value_at_risk(returns,confidence)
    tail_loss = returns[returns <= var]
    return tail_loss.min()
