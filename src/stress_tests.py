import pandas as pd
from performanceMatrix import max_drawdown_CC, conditional_var

def historical_stress_test(
    returns: pd.DataFrame,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    return returns.loc[start_date:end_date]

def stress_metrics(returns: pd.Series) -> dict:
    return {
        "max_drawdown":max_drawdown_CC(returns),
        "cvar_95": conditional_var(returns,0.95),
        "mean_return": returns.mean()
    }

def compare_under_stress(returns_df, sharpe_weights, cvar_weights):
    sharpe_returns = returns_df @ sharpe_weights
    cvar_returns = returns_df @ cvar_weights
    return {
        "sharpe": stress_metrics(sharpe_returns),
        "cvar": stress_metrics(cvar_returns)
    }