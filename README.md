# 📈 Quantitative Portfolio Optimization & Stress Testing Engine

An enterprise-grade financial analytics application that combines **Markowitz Mean-Variance Optimization (MVO)**, **Monte Carlo Stress Testing**, and **Tail-Risk Analysis (VaR/CVaR)** into an interactive web dashboard.

---

## 🌟 Key Features

- **Automated Data Ingestion:** Fetches historical market pricing, computes log returns, and stores cleaned data in a local SQLite database.
- **Markowitz Efficient Frontier:** Uses quadratic programming (`scipy.optimize`) to calculate optimal asset weight allocations for the **Maximum Sharpe Ratio** and **Minimum Variance** portfolios.
- **Monte Carlo Simulations:** Simulates 10,000 portfolio price trajectories over a 252-day horizon using **Cholesky Decomposition** to preserve cross-asset covariance structures.
- **Tail-Risk Analytics:** Calculates Historical **Value at Risk (VaR)** and **Conditional VaR (CVaR / Expected Shortfall)** at 95% and 99% confidence levels.
- **Interactive Web Dashboard:** Built with Streamlit and Plotly for dynamic strategy visualization and parameter adjustments.

---

## 🛠 Tech Stack & Architecture

- **Language:** Python 3.10+
- **Data & Storage:** `yfinance`, `pandas`, `numpy`, `SQLAlchemy`, `sqlite3`
- **Quantitative Modeling:** `scipy.optimize`, `numpy.linalg`
- **Visualization & UI:** `streamlit`, `plotly`

### Project Structure
```text
portfolio_optimization_engine/
├── data_ingestion.py     # Market data ETL & SQLite pipeline
├── portfolio_optimizer.py # Markowitz MVO & Efficient Frontier solver
├── risk_engine.py         # Monte Carlo simulation & VaR/CVaR calculations
├── app.py                 # Streamlit web dashboard
├── portfolio_data.db      # SQLite database (auto-generated)
└── requirements.txt       # Project dependencies