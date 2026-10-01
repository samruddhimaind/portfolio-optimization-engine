import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from portfolio_optimizer import (
    calculate_portfolio_performance,
    generate_efficient_frontier,
    load_returns_data,
    optimize_max_sharpe,
    optimize_min_variance,
)
from risk_engine import calculate_var_cvar, run_monte_carlo_simulation

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Quant Portfolio Optimizer & Risk Engine",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Quantitative Portfolio Optimization & Stress Testing Engine")
st.markdown(
    "An enterprise risk dashboard powered by **Markowitz Mean-Variance Optimization** and **Monte Carlo Simulations**."
)

DB_NAME = "portfolio_data.db"

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.header("⚙️ Simulation Settings")

portfolio_val = st.sidebar.number_input(
    "Initial Portfolio Value ($)", value=100000, step=10000
)
risk_free_rate = (
    st.sidebar.slider(
        "Risk-Free Rate (%)", min_value=0.0, max_value=10.0, value=4.0, step=0.25
    )
    / 100.0
)
num_simulations = st.sidebar.selectbox(
    "Monte Carlo Iterations", [1000, 5000, 10000], index=1
)

# ---------------------------------------------------------
# Data Loading
# ---------------------------------------------------------
@st.cache_data
def get_data():
    df = load_returns_data(DB_NAME)
    mean_ret = df.mean()
    cov = df.cov()
    return df, mean_ret, cov


df_returns, mean_returns, cov_matrix = get_data()
tickers = df_returns.columns.tolist()

# Compute Optimizations
max_sharpe_w = optimize_max_sharpe(
    mean_returns, cov_matrix, risk_free_rate=risk_free_rate
)
ret_ms, vol_ms, sharpe_ms = calculate_portfolio_performance(
    max_sharpe_w, mean_returns, cov_matrix, risk_free_rate=risk_free_rate
)

min_var_w = optimize_min_variance(mean_returns, cov_matrix)
ret_mv, vol_mv, sharpe_mv = calculate_portfolio_performance(
    min_var_w, mean_returns, cov_matrix, risk_free_rate=risk_free_rate
)

# ---------------------------------------------------------
# Key Performance Metrics Cards
# ---------------------------------------------------------
st.subheader("📌 Key Portfolio Strategies")
col1, col2, col3, col4 = st.columns(4)

col1.metric("Max Sharpe Expected Return", f"{ret_ms * 100:.2f}%")
col2.metric("Max Sharpe Volatility (Risk)", f"{vol_ms * 100:.2f}%")
col3.metric("Max Sharpe Ratio", f"{sharpe_ms:.2f}")
col4.metric("Min Variance Volatility", f"{vol_mv * 100:.2f}%")

st.markdown("---")

# ---------------------------------------------------------
# Tab Layout
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    ["📊 Efficient Frontier", "🎲 Monte Carlo Simulation", "🛡️ Tail Risk (VaR/CVaR)"]
)

# --- TAB 1: EFFICIENT FRONTIER ---
with tab1:
    st.subheader("Markowitz Efficient Frontier")

    target_rets, frontier_vols, _ = generate_efficient_frontier(
        mean_returns, cov_matrix, num_portfolios=40
    )

    fig_ef = go.Figure()

    # Plot Efficient Frontier Curve
    fig_ef.add_trace(
        go.Scatter(
            x=frontier_vols * 100,
            y=target_rets * 100,
            mode="lines",
            name="Efficient Frontier",
            line=dict(color="royalblue", width=3),
        )
    )

    # Individual Assets
    ind_vols = np.sqrt(np.diag(cov_matrix * 252)) * 100
    ind_rets = mean_returns * 252 * 100
    fig_ef.add_trace(
        go.Scatter(
            x=ind_vols,
            y=ind_rets,
            mode="markers+text",
            name="Individual Assets",
            text=tickers,
            textposition="top center",
            marker=dict(size=10, color="gray"),
        )
    )

    # Max Sharpe Point
    fig_ef.add_trace(
        go.Scatter(
            x=[vol_ms * 100],
            y=[ret_ms * 100],
            mode="markers",
            name="Max Sharpe Portfolio",
            marker=dict(size=14, color="gold", symbol="star"),
        )
    )

    # Min Variance Point
    fig_ef.add_trace(
        go.Scatter(
            x=[vol_mv * 100],
            y=[ret_mv * 100],
            mode="markers",
            name="Min Variance Portfolio",
            marker=dict(size=12, color="green", symbol="diamond"),
        )
    )

    fig_ef.update_layout(
        xaxis_title="Annualized Volatility (Risk) %",
        yaxis_title="Expected Annual Return %",
        height=500,
        legend=dict(x=0.01, y=0.99),
    )
    st.plotly_chart(fig_ef, use_container_width=True)

    # Pie Chart Allocation Comparison
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Max Sharpe Portfolio Allocation**")
        df_ms_w = pd.DataFrame(
            {"Asset": tickers, "Weight": max_sharpe_w * 100}
        ).query("Weight > 0.01")
        fig_pie_ms = px.pie(
            df_ms_w, values="Weight", names="Asset", hole=0.4, color_discrete_sequence=px.colors.sequential.RdBu
        )
        st.plotly_chart(fig_pie_ms, use_container_width=True)

    with col_b:
        st.markdown("**Min Variance Portfolio Allocation**")
        df_mv_w = pd.DataFrame(
            {"Asset": tickers, "Weight": min_var_w * 100}
        ).query("Weight > 0.01")
        fig_pie_mv = px.pie(
            df_mv_w, values="Weight", names="Asset", hole=0.4, color_discrete_sequence=px.colors.sequential.Greens
        )
        st.plotly_chart(fig_pie_mv, use_container_width=True)

# --- TAB 2: MONTE CARLO SIMULATION ---
with tab2:
    st.subheader(
        f"1-Year Monte Carlo Projections ({num_simulations:,} Trajectories)"
    )

    paths = run_monte_carlo_simulation(
        mean_returns,
        cov_matrix,
        max_sharpe_w,
        portfolio_val,
        252,
        num_simulations,
    )

    # Sample 100 paths for fast rendering
    sample_paths = paths[:, :100]

    fig_mc = go.Figure()
    for i in range(sample_paths.shape[1]):
        fig_mc.add_trace(
            go.Scatter(
                y=sample_paths[:, i],
                mode="lines",
                line=dict(width=1),
                showlegend=False,
                opacity=0.3,
            )
        )

    fig_mc.update_layout(
        xaxis_title="Trading Days",
        yaxis_title="Portfolio Value ($)",
        height=500,
    )
    st.plotly_chart(fig_mc, use_container_width=True)

    final_vals = paths[-1]
    mc_col1, mc_col2, mc_col3 = st.columns(3)
    mc_col1.metric("Median Projected Value", f"${np.median(final_vals):,.2f}")
    mc_col2.metric("5th Percentile (Downside Case)", f"${np.percentile(final_vals, 5):,.2f}")
    mc_col3.metric("95th Percentile (Upside Case)", f"${np.percentile(final_vals, 95):,.2f}")

# --- TAB 3: TAIL RISK (VaR / CVaR) ---
with tab3:
    st.subheader("Value at Risk (VaR) & Conditional VaR (Expected Shortfall)")

    portfolio_daily_returns = df_returns.dot(max_sharpe_w)
    var_95, cvar_95 = calculate_var_cvar(
        portfolio_daily_returns, confidence_level=0.95
    )
    var_99, cvar_99 = calculate_var_cvar(
        portfolio_daily_returns, confidence_level=0.99
    )

    risk_df = pd.DataFrame(
        {
            "Metric": ["95% Confidence Level", "99% Confidence Level"],
            "1-Day VaR ($)": [
                f"${var_95 * portfolio_val:,.2f}",
                f"${var_99 * portfolio_val:,.2f}",
            ],
            "1-Day VaR (%)": [f"{var_95 * 100:.2f}%", f"{var_99 * 100:.2f}%"],
            "1-Day CVaR ($)": [
                f"${cvar_95 * portfolio_val:,.2f}",
                f"${cvar_99 * portfolio_val:,.2f}",
            ],
            "1-Day CVaR (%)": [f"{cvar_95 * 100:.2f}%", f"{cvar_99 * 100:.2f}%"],
        }
    )

    st.table(risk_df)
    st.info(
        "💡 **Interpretation:** VaR indicates the maximum loss expected with a given confidence level. CVaR measures the expected loss in extreme tail scenarios beyond the VaR threshold."
    )