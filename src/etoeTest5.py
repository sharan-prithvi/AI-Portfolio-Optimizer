import pandas as pd
import matplotlib.pyplot as plt
from optimizer import opt_portfolio
from cvar_optimizer import CVaROpt
from performanceMatrix import (
    max_drawdown_CC,
    conditional_var,
    cumulative_returns
)
import datetime
from dataloader import fetch_prices
from returns import compute_returns
from equalWeightPortfolio import equal_weight_port
from expected_returns import mean_historical_returns
from risk_models import sample_cov

# load data
TICKERS=["RELIANCE.NS","TCS.NS","INFY.NS"]

#COVID Window
COVID_START = "2020-02-15"
COVID_END   = "2020-04-15"
# start_date=datetime.date(2000,1,1)
# end_date=datetime.datetime.today().date()
LAMBDA_EWMA = 0.94
SHRINKAGE_DELTA = 0.2

prices = fetch_prices(TICKERS, COVID_START, COVID_END)
returns_df = compute_returns(prices)

portfolio_returns,weights = equal_weight_port(returns_df)

mu = mean_historical_returns(returns_df)
cov = sample_cov(returns_df)

ef = opt_portfolio(mu, cov,0.0)

cvar_opt = CVaROpt(returns_df, confidence=0.95)
cvar_weights = cvar_opt.optimize()

assert abs(ef.sum() - 1) < 1e-6
assert abs(cvar_weights.sum() - 1) < 1e-6

covid_returns = returns_df.loc[COVID_START:COVID_END]

sharpe_covid = covid_returns @ ef
cvar_covid = covid_returns @ cvar_weights

# Stress Metrics
def stress_report(returns):
    return {
        "mean_return":returns.mean(),
        "max_drawdown": max_drawdown_CC(returns),
        "cvar_95": conditional_var(returns,0.95)
    }

print("\n-- COVID STRESS TEST --")
print("Sharpe:", stress_report(sharpe_covid))
print("CVaR:", stress_report(cvar_covid))

def recovery_time(returns):
    cum = cumulative_returns(returns)
    peak = cum.iloc[0]
    recovered = cum[cum >= peak]
    return recovered.index[0] if not recovered.empty else None

print("\nRecovery Date:")
print("Sharpe:", recovery_time(sharpe_covid))
print("CVaR  :", recovery_time(cvar_covid))

# --------------------------------------------------
# STEP 6 — Plot COVID Cumulative Returns
# --------------------------------------------------

plt.figure(figsize=(10, 5))
cumulative_returns(sharpe_covid).plot(label="Sharpe Portfolio")
cumulative_returns(cvar_covid).plot(label="CVaR Portfolio")
plt.title("COVID Crash Stress Test (Feb–Apr 2020)")
plt.legend()
plt.grid(True)
plt.show()

# --------------------------------------------------
# STEP 7 — Synthetic Volatility Spike Test
# --------------------------------------------------

def volatility_spike(returns, scale=2.5):
    return returns * scale

shocked_returns = volatility_spike(returns_df)

sharpe_shock = shocked_returns @ ef
cvar_shock   = shocked_returns @ cvar_weights

print("\n--- VOLATILITY SPIKE STRESS TEST ---")
print("Sharpe:", stress_report(sharpe_shock))
print("CVaR  :", stress_report(cvar_shock))

# --------------------------------------------------
# STEP 8 — Assertions (Behavioral Guarantees)
# --------------------------------------------------

assert max_drawdown_CC(cvar_covid) > max_drawdown_CC(sharpe_covid), \
    "CVaR drawdown should be smaller than Sharpe"

assert conditional_var(cvar_covid, 0.95) > conditional_var(sharpe_covid, 0.95), \
    "CVaR tail loss should be less severe"

print("\n✅ Phase 5C E2E stress test PASSED")