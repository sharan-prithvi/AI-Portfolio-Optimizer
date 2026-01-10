from dataloader import fetch_prices
from returns import compute_returns
from equalWeightPortfolio import equal_weight_port
import datetime
from expected_returns import mean_historical_returns
from optimizer import markowitz_opt
from risk_models import sample_cov
from performanceMatrix import (
    cumulative_returns,
    drawdown,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio
)
from visualization import plot_drawdown
from matplotlib import pyplot as plt

tickers = ['RELIANCE.NS','TCS.NS','INFY.NS']
start_date = datetime.date(2000,1,1)
end_date = datetime.datetime.now().date()
print(f"Collecting data for Tickers:- {tickers} for {(end_date - start_date).days} days")
prices = fetch_prices(tickers=tickers, start_date=start_date,end_date=end_date)
returns = compute_returns(prices)

# PHASE 1
eq_return,eq_weights = equal_weight_port(returns)

# PHASE 2
mu = mean_historical_returns(returns)
cov = sample_cov(returns)
ef = markowitz_opt(mu, cov)
opt_weights = ef.clean_weights()
opt_returns = returns.dot(list(opt_weights.values()))

# PHASE 3
eq_cumulative = cumulative_returns(eq_return)
opt_cumulative = cumulative_returns(opt_returns)

eq_dd = drawdown(eq_cumulative)
opt_dd = drawdown(opt_cumulative)

# Metrics Comparision
print("\n📊 Portfolio Comparison\n")
print("Equal Weight:")
print("Sharpe:", round(sharpe_ratio(eq_return), 3))
print("Sortino:", round(sortino_ratio(eq_return), 3))
print("Max Drawdown:", round(max_drawdown(eq_dd), 3))

print("\nOptimized Portfolio:")
print("Sharpe:", round(sharpe_ratio(opt_returns), 3))
print("Sortino:", round(sortino_ratio(opt_returns), 3))
print("Max Drawdown:", round(max_drawdown(opt_dd), 3))

# VISUALIZATION
plt.figure(figsize=(10, 5))
eq_cumulative.plot(label="Equal Weights")
opt_cumulative.plot(label="Optimized")
plt.title("Cumulative Returns Comparision")
plt.legend()
plt.show()

plot_drawdown(eq_dd, "Equal Weight Drawdown")
plot_drawdown(opt_dd, "Optimized Portfolio Drawdown")