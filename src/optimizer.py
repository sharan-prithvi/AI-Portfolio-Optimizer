from pypfopt import EfficientFrontier
import pandas as pd

def markowitz_opt(mu, cov, risk_free_ratio = 0.05):
    ef = EfficientFrontier(mu, cov)
    ef.max_sharpe(risk_free_rate=risk_free_ratio)
    # clean_weights = ef.clean_weights()
    return ef

def opt_portfolio(mu, cov, risk_free_ratio=0.0):
    ef = EfficientFrontier(mu,cov)
    try:
        ef.max_sharpe(risk_free_rate=risk_free_ratio)
    except ValueError:
        print("Warning: All returns are negative. Falling back to Min Volatility")
        ef.min_volatility()
    return pd.Series(ef.clean_weights())