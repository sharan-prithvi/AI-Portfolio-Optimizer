import matplotlib.pyplot as plt
from pypfopt import EfficientFrontier,plotting

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