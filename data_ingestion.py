import sqlite3
import numpy as np
import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "SPY", "TLT", "GLD"]
START_DATE = "2018-01-01"
END_DATE = "2026-01-01"
DB_NAME = "portfolio_data.db"

def fetch_market_data(tickers, start_date, end_date):
    print(f"[INFO] Fetching data for {tickers} from {start_date} to {end_date}...")
    data = yf.download(tickers, start=start_date, end=end_date)
    
    # Extract Adjusted Close cleanly
    if "Adj Close" in data:
        prices = data["Adj Close"]
    elif "Close" in data:
        prices = data["Close"]
    else:
        prices = data

    prices = prices.ffill().bfill()
    print("[SUCCESS] Market data downloaded successfully.")
    return prices

def process_financial_data(df_prices):
    print("[INFO] Computing daily log returns...")
    df_returns = np.log(df_prices / df_prices.shift(1)).dropna()

    df_prices_long = df_prices.reset_index().melt(
        id_vars=["Date"], var_name="Ticker", value_name="Adj_Close"
    )
    df_returns_long = df_returns.reset_index().melt(
        id_vars=["Date"], var_name="Ticker", value_name="Log_Return"
    )

    # Ensure clean string formatting for columns
    df_prices_long["Ticker"] = df_prices_long["Ticker"].astype(str)
    df_returns_long["Ticker"] = df_returns_long["Ticker"].astype(str)

    return df_prices_long, df_returns_long

def store_in_sqlite(df_prices_long, df_returns_long, db_name):
    print(f"[INFO] Saving data to database '{db_name}'...")
    engine = create_engine(f"sqlite:///{db_name}")
    df_prices_long.to_sql("asset_prices", engine, if_exists="replace", index=False)
    df_returns_long.to_sql("asset_returns", engine, if_exists="replace", index=False)
    print("[SUCCESS] Data successfully written to SQLite database!")

if __name__ == "__main__":
    raw_prices = fetch_market_data(TICKERS, START_DATE, END_DATE)
    prices_long, returns_long = process_financial_data(raw_prices)
    store_in_sqlite(prices_long, returns_long, DB_NAME)