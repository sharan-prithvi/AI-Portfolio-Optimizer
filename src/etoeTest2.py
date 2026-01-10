from dataloader import fetch_prices
from returns import compute_returns
from equalWeightPortfolio import equal_weight_port
from performanceMatrix import sharpe_ratio
import datetime
from expected_returns import mean_historical_returns
from optimizer import markowitz_opt
from risk_models import sample_cov
from visualization import plot_eff_frontier

tickers = ['RELIANCE.NS','TCS.NS','INFY.NS']
print(f"Collecting data for Tickers:- {tickers} for {(datetime.datetime.now().date() - datetime.date(2015,1,1)).days} days")
prices = fetch_prices(tickers=tickers, start_date=datetime.date(2015,1,1),end_date=datetime.datetime.now().date())
returns = compute_returns(prices)
portfolio_returns,weights = equal_weight_port(returns)

mu = mean_historical_returns(returns)
cov = sample_cov(returns)

ef = markowitz_opt(mu, cov)

print("Optimized Weights")
print(ef.clean_weights())

plot_eff_frontier(mu, cov)