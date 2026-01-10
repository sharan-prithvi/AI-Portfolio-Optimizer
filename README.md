# 🎯 Project Objective

- Build a robust financial data pipeline
- Implement classical portfolio theory correctly
- Avoid common quant & ML pitfalls (look-ahead bias, overfitting)
- Demonstrate recruiter-grade understanding of AI + Finance + Systems

# 🧠 Why This Project Matters

Most portfolio projects jump directly to ML models.
This project:

- Starts with baselines
- Builds mathematical intuition first
- Adds AI only where it is justified

This mirrors how real quant & AI teams work.

# PHASE 1 - Data Pipeline & Baseline Portfolio 
## Goal

Establish a clean, reproducible foundation without prediction or optimization bias.

## ✅ What was built

1. Financial Data ingestion
- Historical stock prices fetched from yahoo finance
- Close prices used to account for dividends and splits

2. Return Computation
Prices are non stationary. All modeling is performed on returns.
$$r_t = \frac{P_t - P_{t-1}}{P_{t-1}}$$

3. Correlation Analysis
- Asset correlations computed to understand diversification
- Highlights why portfolio risk ≠ sum of individual risks

`returns_df.corr()`

4. Equal-Weight Portfolio (Baseline)
Each asset is assigned equal capital allocation.

Why this matters:
- No assumptions
- No optimization
- Serves as a benchmark for all future models

5. Portfolio Performance Metrics
- Annualized returns
- Annualized volatility
- Sharpe Ratio

## 🧠 Key Learnings (Phase 1)

- Why returns matter more than prices
- How diversification reduces risk
- Importance of baselines before AI/ML
- Clean modular financial engineering

## ⚠️ Limitations (Phase 1)
- No optimization
- No predictive modeling
- No transaction costs
- Static portfolio
These limitations are intentional.

# 📌 PHASE 2 — Mean–Variance Portfolio Optimization (Markowitz)
## 🎯 Goal
Implement classical portfolio theory using mathematically correct optimization.

## 🧠 Theory Overview
Mean–Variance Optimization seeks to:
- Maximize expected return
- Minimize portfolio variance

$$𝜎_p^2 = w^TΣw$$


Where:
- w: portfolio weights
- Σ: covariance matrix

## ✅ What Was Built
1. Expected Returns Estimation : Using historical mean returns (⚠️ Known to be noisy — acknowledged explicitly.)
2. Covariance-Based Risk Model: Simple covariance matrix
3. Mean–Variance Optimizer
- Objective: Maximize Sharpe Ratio
- Constraints:
    - Long-only (no short selling)
    - Fully invested portfolio (Σw = 1)
4. Efficient Frontier Visualization

## ⚠️ Limitations (Phase 2)
- Expected returns are backward-looking
- Covariance assumes stationarity
- Gaussian return assumption
- Not suitable alone for retail investors
These are known and documented weaknesses of Markowitz optimization.

# 📌 PHASE 3 — Risk Beyond Variance (Drawdowns & Downside Risk)
## 🎯 Goal
Extend the portfolio optimizer to model **real investor risk**, not just mathematical volatility.

Phase 3 focuses on **path-dependent risk and downside risk**, which are critical in real-world portfolio management but often ignored in academic implementations.

## Why Variance Is Not Enough ❓
Variance treats:
- Upside volatility ❌
- Downside volatility ❌
as equally bad.

However, investors do not fear upside volatility — they fear losses and deep drawdowns.
Two portfolios can have identical volatility but vastly different investor experiences.

## 🧠 Concepts Introduced in Phase 3
1. Cumulative Returns
Tracks portfolio value over time:
$$V_t​=∏_{i=1}^t​(1+r_i​)$$

2. Drawdown
Measures loss from the most recent peak:
$$Drawdown_t=\frac{V_t - max(V_{0:t})}{max(V_{0:t})}$$
This directly models *investor pain*.

3. Maximum Drawdown
Worst peak-to-trough loss during the period.
Widely used by:
- Hedge funds
- PMS Desks
- Risk Committees

4. Downside Drawdown
Volatility calculated **only for negative returns**.
Unlike standard deviation, it **does not penalize upside volatility**.

5. Sortino Ratio
Improves upon Sharpe Ratio: 
$$Sortino=\frac{R_p - R_f}{𝜎_{downside}}$$
Focuses purely on downside risk.

## 📊 Visualizations
- Cumulative return comparison
- Drawdown curves for:
    - Equal-weight portfolio
    - Optimized (Phase-2) portfolio
These plots clearly show why volatility alone is misleading.

| Metric        | Equal Weight | Optimized    |
| ------------- | ------------ | ------------ |
| Volatility    | Lower        | Higher       |
| Sharpe Ratio  | Baseline     | Higher       |
| Sortino Ratio | Lower        | Higher       |
| Max Drawdown  | Often Lower  | Often Higher |

## **📌 Key Insight:**
Optimizing for Sharpe can increase drawdown risk.
This is a real institutional trade-off.

## ⚠️ Limitations (Intentional)
- No transaction costs
- No turnover penalty
- No behavioral constraints
- No tail-risk modeling (CVaR yet)

# 📌 Disclaimer
This project is for *educational and research purposes only*.
It does **not** constitute financial or investment advice.

# 👨‍💻 Author
**Prithvi Sharan**
*Full Stack Developer for State Street Corporation and ML/AI Enthusiast*