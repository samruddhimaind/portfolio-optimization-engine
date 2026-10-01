import sqlite3
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ---------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------
DB_NAME = "portfolio_data.db"
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.04  # 4% annual risk-free rate (e.g., 10-Yr US Treasury)


# ---------------------------------------------------------
# 1. Load Returns Matrix from Database
# ---------------------------------------------------------
def load_returns_data(db_name):
    """Loads log returns from SQLite and pivots into a (Date x Ticker) matrix."""
    conn = sqlite3.connect(db_name)
    df = pd.read_sql("SELECT * FROM asset_returns;", conn)
    conn.close()

    # Convert all column names to lowercase for consistency
    df.columns = [str(col).lower() for col in df.columns]

    # Find the matching columns
    date_col = [c for c in df.columns if 'date' in c][0]
    ticker_col = [c for c in df.columns if 'ticker' in c][0]
    return_col = [c for c in df.columns if 'return' in c or 'log' in c][0]

    # Clean ticker string column values
    df[ticker_col] = df[ticker_col].astype(str)

    # Pivot into Wide Matrix
    df_pivot = df.pivot(index=date_col, columns=ticker_col, values=return_col)
    
    # Drop empty columns/rows if any
    df_pivot = df_pivot.dropna(how='all', axis=1).dropna(how='all', axis=0)
    
    return df_pivot


# ---------------------------------------------------------
# 2. Portfolio Statistics Calculator
# ---------------------------------------------------------
def calculate_portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate=0.04):
    """
    Calculates annualized return, volatility, and Sharpe ratio for given weights.
    """
    # Annualized portfolio return
    portfolio_return = np.sum(mean_returns * weights) * TRADING_DAYS_PER_YEAR
    
    # Annualized portfolio volatility (standard deviation)
    portfolio_volatility = np.sqrt(
        np.dot(weights.T, np.dot(cov_matrix * TRADING_DAYS_PER_YEAR, weights))
    )
    
    # Sharpe Ratio
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
    
    return portfolio_return, portfolio_volatility, sharpe_ratio


# ---------------------------------------------------------
# 3. Optimization Functions
# ---------------------------------------------------------
def optimize_max_sharpe(mean_returns, cov_matrix, risk_free_rate=0.04):
    """Finds asset weights that maximize the Sharpe Ratio."""
    num_assets = len(mean_returns)
    
    # Objective function to MINIMIZE: Negative Sharpe Ratio
    def negative_sharpe(weights):
        _, _, sharpe = calculate_portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)
        return -sharpe

    # Constraints: Sum of weights = 1
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    
    # Bounds: Long-only portfolio (0 <= weight <= 1)
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    
    # Initial guess: Equal weights
    initial_weights = num_assets * [1.0 / num_assets]
    
    result = minimize(
        negative_sharpe, 
        initial_weights, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints
    )
    
    return result.x


def optimize_min_variance(mean_returns, cov_matrix):
    """Finds asset weights that minimize total portfolio volatility."""
    num_assets = len(mean_returns)
    
    # Objective function to MINIMIZE: Portfolio Volatility
    def portfolio_vol(weights):
        _, volatility, _ = calculate_portfolio_performance(weights, mean_returns, cov_matrix)
        return volatility

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds = tuple((0.0, 1.0) for _ in range(num_assets))
    initial_weights = num_assets * [1.0 / num_assets]
    
    result = minimize(
        portfolio_vol, 
        initial_weights, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints
    )
    
    return result.x


def generate_efficient_frontier(mean_returns, cov_matrix, num_portfolios=50):
    """Generates optimal portfolio points across target returns for the Efficient Frontier."""
    num_assets = len(mean_returns)
    
    # Get return range from min variance to max achievable return
    min_vol_weights = optimize_min_variance(mean_returns, cov_matrix)
    min_ret, _, _ = calculate_portfolio_performance(min_vol_weights, mean_returns, cov_matrix)
    
    max_ret = np.max(mean_returns) * TRADING_DAYS_PER_YEAR
    target_returns = np.linspace(min_ret, max_ret, num_portfolios)
    
    frontier_volatilities = []
    frontier_weights = []

    for target in target_returns:
        def portfolio_vol(weights):
            return calculate_portfolio_performance(weights, mean_returns, cov_matrix)[1]

        # Constraints: Sum of weights = 1 AND Portfolio Return = Target Return
        constraints = (
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
            {'type': 'eq', 'fun': lambda w: calculate_portfolio_performance(w, mean_returns, cov_matrix)[0] - target}
        )
        bounds = tuple((0.0, 1.0) for _ in range(num_assets))
        initial_weights = num_assets * [1.0 / num_assets]

        res = minimize(
            portfolio_vol, 
            initial_weights, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints
        )
        
        if res.success:
            frontier_volatilities.append(res.fun)
            frontier_weights.append(res.x)

    return target_returns[:len(frontier_volatilities)], np.array(frontier_volatilities), frontier_weights


# ---------------------------------------------------------
# Execution & Testing
# ---------------------------------------------------------
if __name__ == "__main__":
    print("[INFO] Reading returns matrix from database...")
    df_returns = load_returns_data(DB_NAME)
    
    tickers = df_returns.columns.tolist()
    mean_returns = df_returns.mean()
    cov_matrix = df_returns.cov()

    print(f"\n[INFO] Assets loaded: {tickers}")

    # 1. Calculate Maximum Sharpe Ratio Portfolio
    max_sharpe_weights = optimize_max_sharpe(mean_returns, cov_matrix, RISK_FREE_RATE)
    ret_ms, vol_ms, sharpe_ms = calculate_portfolio_performance(max_sharpe_weights, mean_returns, cov_matrix, RISK_FREE_RATE)

    print("\n" + "="*50)
    print("      MAX SHARPE RATIO PORTFOLIO      ")
    print("="*50)
    print(f"Expected Annual Return : {ret_ms * 100:.2f}%")
    print(f"Annualized Volatility  : {vol_ms * 100:.2f}%")
    print(f"Sharpe Ratio           : {sharpe_ms:.2f}")
    print("\nOptimal Weights Allocation:")
    for ticker, weight in zip(tickers, max_sharpe_weights):
        print(f"  - {ticker:6s}: {weight * 100:.2f}%")

    # 2. Calculate Minimum Variance Portfolio
    min_var_weights = optimize_min_variance(mean_returns, cov_matrix)
    ret_mv, vol_mv, sharpe_mv = calculate_portfolio_performance(min_var_weights, mean_returns, cov_matrix, RISK_FREE_RATE)

    print("\n" + "="*50)
    print("     MINIMUM VARIANCE PORTFOLIO     ")
    print("="*50)
    print(f"Expected Annual Return : {ret_mv * 100:.2f}%")
    print(f"Annualized Volatility  : {vol_mv * 100:.2f}%")
    print(f"Sharpe Ratio           : {sharpe_mv:.2f}")
    print("\nOptimal Weights Allocation:")
    for ticker, weight in zip(tickers, min_var_weights):
        print(f"  - {ticker:6s}: {weight * 100:.2f}%")