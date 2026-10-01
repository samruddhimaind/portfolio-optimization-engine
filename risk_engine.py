import sqlite3
import numpy as np
import pandas as pd
from portfolio_optimizer import load_returns_data, optimize_max_sharpe, calculate_portfolio_performance

# ---------------------------------------------------------
# Configuration & Parameters
# ---------------------------------------------------------
DB_NAME = "portfolio_data.db"
INITIAL_PORTFOLIO_VALUE = 100_000  # $100,000 investment
SIMULATION_DAYS = 252              # 1-year trading horizon
NUM_SIMULATIONS = 10_000           # 10,000 Monte Carlo runs
RISK_FREE_RATE = 0.04


# ---------------------------------------------------------
# 1. Historical Value at Risk (VaR) & CVaR
# ---------------------------------------------------------
def calculate_var_cvar(portfolio_returns, confidence_level=0.95):
    """
    Calculates Historical Value at Risk (VaR) and Conditional Value at Risk (CVaR).
    """
    # Sort returns from worst to best
    sorted_returns = np.sort(portfolio_returns)
    
    # Calculate index percentile
    index = int((1 - confidence_level) * len(sorted_returns))
    
    # Historical VaR
    var = -sorted_returns[index]
    
    # Historical CVaR (Expected Shortfall): Average of losses worse than VaR
    cvar = -sorted_returns[:index].mean()
    
    return var, cvar


# ---------------------------------------------------------
# 2. Monte Carlo Simulation Engine
# ---------------------------------------------------------
def run_monte_carlo_simulation(mean_returns, cov_matrix, weights, initial_value, days, num_sims):
    """
    Runs Monte Carlo simulation for portfolio trajectories using Cholesky Decomposition
    to maintain correlated asset price movements.
    """
    num_assets = len(weights)
    
    # Compute Cholesky Decomposition L where L * L^T = Covariance Matrix
    # Add a small regularization term to ensure positive semi-definiteness
    cholesky_matrix = np.linalg.cholesky(cov_matrix + 1e-8 * np.eye(num_assets))
    
    # Generate random standard normal variables: Shape (days, num_sims, num_assets)
    random_shocks = np.random.normal(size=(days, num_sims, num_assets))
    
    # Correlate shocks across assets
    correlated_shocks = np.einsum('ij,tkj->tki', cholesky_matrix, random_shocks)
    
    # Daily drift (expected return per day)
    daily_drift = mean_returns.values - 0.5 * np.diag(cov_matrix)
    
    # Calculate daily simulated returns for each asset
    simulated_daily_returns = np.exp(daily_drift + correlated_shocks)
    
    # Calculate weighted daily return across the portfolio
    portfolio_daily_growth = np.dot(simulated_daily_returns, weights)
    
    # Calculate cumulative portfolio paths starting from initial_value
    portfolio_paths = np.vstack([
        np.full(num_sims, initial_value),
        initial_value * np.cumprod(portfolio_daily_growth, axis=0)
    ])
    
    return portfolio_paths


# ---------------------------------------------------------
# Execution & Testing
# ---------------------------------------------------------
if __name__ == "__main__":
    print("[INFO] Loading returns and computing optimal Max Sharpe portfolio...")
    df_returns = load_returns_data(DB_NAME)
    mean_returns = df_returns.mean()
    cov_matrix = df_returns.cov()

    # Get Max Sharpe optimal weights
    weights = optimize_max_sharpe(mean_returns, cov_matrix, RISK_FREE_RATE)
    
    # Calculate historical daily portfolio returns
    portfolio_daily_returns = df_returns.dot(weights)

    # 1. Historical VaR & CVaR
    var_95, cvar_95 = calculate_var_cvar(portfolio_daily_returns, confidence_level=0.95)
    var_99, cvar_99 = calculate_var_cvar(portfolio_daily_returns, confidence_level=0.99)

    print("\n" + "="*55)
    print("      HISTORICAL TAIL RISK METRICS ($100,000 PORTFOLIO)      ")
    print("="*55)
    print(f"Daily 95% Value at Risk (VaR)  : ${var_95 * INITIAL_PORTFOLIO_VALUE:,.2f} ({var_95*100:.2f}%)")
    print(f"Daily 95% Conditional VaR (CVaR): ${cvar_95 * INITIAL_PORTFOLIO_VALUE:,.2f} ({cvar_95*100:.2f}%)")
    print(f"Daily 99% Value at Risk (VaR)  : ${var_99 * INITIAL_PORTFOLIO_VALUE:,.2f} ({var_99*100:.2f}%)")
    print(f"Daily 99% Conditional VaR (CVaR): ${cvar_99 * INITIAL_PORTFOLIO_VALUE:,.2f} ({cvar_99*100:.2f}%)")

    # 2. Monte Carlo Simulation
    print(f"\n[INFO] Running {NUM_SIMULATIONS:,} Monte Carlo simulations over {SIMULATION_DAYS} trading days...")
    paths = run_monte_carlo_simulation(
        mean_returns, cov_matrix, weights, INITIAL_PORTFOLIO_VALUE, SIMULATION_DAYS, NUM_SIMULATIONS
    )

    final_portfolio_values = paths[-1]
    
    print("\n" + "="*55)
    print("      1-YEAR MONTE CARLO PROJECTION RESULTS      ")
    print("="*55)
    print(f"Expected Portfolio Value (Median) : ${np.median(final_portfolio_values):,.2f}")
    print(f"5th Percentile (Worst 5% Case)    : ${np.percentile(final_portfolio_values, 5):,.2f}")
    print(f"95th Percentile (Best 5% Case)    : ${np.percentile(final_portfolio_values, 95):,.2f}")