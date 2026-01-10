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
