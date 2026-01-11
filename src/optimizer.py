from pypfopt import EfficientFrontier

def markowitz_opt(mu, cov, risk_free_ratio = 0.05):
    ef = EfficientFrontier(mu, cov)
    ef.max_sharpe(risk_free_rate=risk_free_ratio)
    # clean_weights = ef.clean_weights()
    return ef

def opt_portfolio(mu, cov):
    ef = EfficientFrontier(mu,cov)
    ef.max_sharpe()
    return ef.clean_weights()