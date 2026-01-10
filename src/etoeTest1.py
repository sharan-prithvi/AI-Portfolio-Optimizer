from dataloader import fetch_prices
from returns import compute_returns
from equalWeightPortfolio import equal_weight_port
from performanceMatrix import sharpe_ratio
import datetime
tickers = ['RELIANCE.NS','TCS.NS','INFY.NS']
prices = fetch_prices(tickers=tickers, start_date="2020-01-01",end_date=datetime.datetime.now())
returns = compute_returns(prices)
portfolio_returns,weights = equal_weight_port(returns)

print("Weights:",weights)
print("Portfolio Returns:-",portfolio_returns)
print("Sharpe:-", sharpe_ratio(portfolio_returns))