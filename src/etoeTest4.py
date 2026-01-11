import pandas as pd
import matplotlib.pyplot as plt
import datetime
from dataloader import fetch_prices
from returns import compute_returns
from expected_returns import mean_historical_returns
from risk_models import ewma_cov, shrinkage_cov, sample_cov
from optimizer import opt_portfolio
from portfolioReturns import port_returns
from performanceMatrix import drawdown

TICKERS=["RELIANCE.NS","TCS.NS","INFY.NS"]
start_date=datetime.date(2000,1,1)
end_date=datetime.datetime.today().date()
LAMBDA_EWMA = 0.94
SHRINKAGE_DELTA = 0.2

prices = fetch_prices(TICKERS, start_date, end_date)
returns_df = compute_returns(prices)

mu = mean_historical_returns(returns_df)

# Risk Models
sample_cov = sample_cov(returns_df)
ewma_cov = ewma_cov(returns_df, lambda_=LAMBDA_EWMA)
shrink_cov = shrinkage_cov(sample_cov, delta=SHRINKAGE_DELTA)

# Portfolio Opt
weights_sample = opt_portfolio(mu, sample_cov)
weights_ewma = opt_portfolio(mu, ewma_cov)
weights_shrink = opt_portfolio(mu, shrink_cov)

# Portfolio Return series
ret_sample = port_returns(returns_df, weights_sample)
ret_ewma = port_returns(returns_df, weights_ewma)
ret_shrink = port_returns(returns_df, weights_shrink)

# Cumulative Returns
cum_sample = (1 + ret_sample).cumprod()
cum_ewma = (1 + ret_ewma).cumprod()
cum_shrink = (1 + ret_shrink).cumprod()

# PLOT
plt.figure()
plt.plot(cum_sample, label="Sample Covariance")
plt.plot(cum_ewma,label="EWMA Covariance")
plt.plot(cum_shrink,label="Shrinkage Covariance")
plt.legend()
plt.title("Cumulative returns - risk model comparision")
plt.show()

# Drawdown Analysis
dd_sample = drawdown(cum_sample)
dd_ewma = drawdown(cum_ewma)
dd_shrink = drawdown(cum_shrink)

# PLOT
plt.figure()
plt.plot(dd_sample, label="Sample Covariance")
plt.plot(dd_ewma, label="EWMA Covariance")
plt.plot(dd_shrink, label="Shrinkage Covariance")
plt.legend()
plt.title("Drawdown Comparision - Risk Stability")
plt.show()

# Summary
summary = pd.DataFrame({
    "Sample": [
        dd_sample.min(),
        ret_sample.std() * (252 ** 0.5)
    ],
    "EWMA": [
        dd_ewma.min(),
        ret_ewma.std() * (252 ** 0.5)
    ],
    "Shrinkage": [
        dd_shrink.min(),
        ret_shrink.std() * (252 ** 0.5)
    ]
}, index = ["MAX Drawdown", "Annualized Volatility"])

print("\nPHASE 4 - Risk model Comparision")
print(summary)