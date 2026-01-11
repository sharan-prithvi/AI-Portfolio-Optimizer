import matplotlib.pyplot as plt
import numpy as np
from pypfopt import EfficientFrontier,plotting
from risk_models import ewma_cov

def plot_cummulative_returns(returns, title):
    cumulative = (1 + returns).cumprod()
    cumulative.plot(title= title)
    plt.show()
    
def plot_eff_frontier(mu, cov):
    ef = EfficientFrontier(mu, cov)
    fig,ax = plt.subplots(figsize=(8,6))
    # ef.plot_efficient_frontier(ax=ax,show_assets=True)
    plotting.plot_efficient_frontier(ef, ax=ax, show_assets=True)
    plt.title("Efficient Frontier")
    plt.show()
    
def plot_drawdown(drawdown_series, title):
    drawdown_series.plot(figsize=(10,4))
    plt.title(title)
    plt.ylabel("Drawdown")
    plt.show()
    
def ewma_plot(returns_df):
    asset = returns_df.columns[0]
    rolling_vol = returns_df[asset].rolling(60).std() * np.sqrt(252)
    ewma_vol = np.sqrt(ewma_cov(returns_df=returns_df)[asset][asset])
    plt.plot(rolling_vol, label="Rolling Vol (60 Days)")
    plt.axline(ewma_vol, color="red",label="EWMA Vol")
    plt.legend()
    plt.title(f"Volatility Comparision: {asset}")
    plt.show()

def cum_plot_comparision(cum_sample, cum_ewma):
    plt.plot(cum_sample,label="Sample Covariance")
    plt.plot(cum_ewma,label="EWMA Covariance")
    plt.legend()
    plt.title("Cumulative Returns Comparision")
    plt.show()
    
def dd_comparision(dd_sample, dd_ewma):
    plt.plot(dd_sample, label="Sample Comparision")
    plt.plot(dd_ewma, label="EWMA Covariance")
    plt.legend()
    plt.title("Drawdown Comparision")
    plt.show()