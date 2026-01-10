import yfinance as yf
import pandas as pd

def fetch_prices(tickers, start_date, end_date):
    data = yf.download(tickers=tickers, start=start_date, end=end_date)['Close']
    data = data.dropna()
    return data