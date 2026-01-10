import matplotlib.pyplot as plt

def plot_cummulative_returns(returns, title):
    cumulative = (1 + returns).cumprod()
    cumulative.plot(title= title)
    plt.show()