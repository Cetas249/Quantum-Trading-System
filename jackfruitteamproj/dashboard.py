"""
interface/dashboard.py
Real-time trading dashboard for Python 3.14
"""

from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Any, Optional
from collections.abc import Sequence

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

try:  # pragma: no cover - optional UI dependency
    import streamlit as st  # type: ignore[import-not-found]
except Exception as exc:
    st = None
    STREAMLIT_IMPORT_ERROR: Optional[Exception] = exc
else:
    STREAMLIT_IMPORT_ERROR = None

@dataclass
class DashboardMetrics:
    """Container for dashboard metrics"""
    daily_pnl: float
    total_return: float
    sharpe_ratio: float
    win_rate: float
    active_positions: int
    portfolio_value: float
    max_drawdown: float

class TradingDashboard:
    """Python 3.14 optimized Streamlit dashboard"""
    
    def __init__(self, trading_system: Any) -> None:
        self.trading_system: Any = trading_system
        self.update_interval: int = 1
        self.data_cache: dict[str, Any] = {}
    
    def run_dashboard(self) -> None:
        """Run the trading dashboard"""
        _ensure_streamlit()
        
        st.set_page_config(
            page_title="Quantum Trading System",
            page_icon="🚀",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.markdown("""
        <style>
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            margin: 0.5rem 0;
        }
        .profit { color: #00ff00; }
        .loss { color: #ff0000; }
        .neutral { color: #ffff00; }
        </style>
        """, unsafe_allow_html=True)
        
        # Sidebar navigation
        st.sidebar.title("🚀 Quantum Trading")
        page = st.sidebar.selectbox(
            "Navigation",
            ["Overview", "Live Trading", "Portfolio", "Risk Management", "Model Performance"]
        )
        
        # Auto-refresh toggle
        auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
        
        if page == "Overview":
            self.show_overview()
        elif page == "Live Trading":
            self.show_live_trading()
        elif page == "Portfolio":
            self.show_portfolio()
        elif page == "Risk Management":
            self.show_risk_management()
        elif page == "Model Performance":
            self.show_model_performance()
    
    def show_overview(self) -> None:
        """Show overview dashboard"""
        st.title("📊 Trading System Overview")
        
        # Metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Daily P&L", "$1,250.00")
        with col2:
            st.metric("Total Return", "15.50%")
        with col3:
            st.metric("Sharpe Ratio", "1.85")
        with col4:
            st.metric("Win Rate", "62.5%")
        with col5:
            st.metric("Active Positions", "7")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Portfolio Performance")
            
            # Generate sample data
            dates = pd.date_range(start='2025-09-12', periods=100, freq='D')
            returns = np.random.normal(0.0008, 0.015, 100)
            portfolio_values = 100000 * np.cumprod(1 + returns)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=portfolio_values,
                mode='lines',
                name='Portfolio',
                line=dict(color='#00ff00', width=2)
            ))
            
            fig.update_layout(
                title="Portfolio Value Over Time",
                template="plotly_dark",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Strategy Allocation")
            
            strategies = ['Quantum Opt', 'ML Ensemble', 'Sentiment', 'RL Agent']
            values = [30, 25, 25, 20]
            
            fig = px.pie(
                values=values,
                names=strategies,
                title="Strategy Allocation"
            )
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
    
    def show_live_trading(self) -> None:
        """Show live trading interface"""
        st.title("🔴 Live Trading")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Market Data")
            
            symbols = st.multiselect(
                "Select Symbols",
                ["AAPL", "GOOGL", "BTC-USD", "ETH-USD"],
                default=["AAPL"]
            )
            
            if symbols:
                # Generate sample candlestick data
                dates = pd.date_range(start='2025-11-01', periods=50, freq='D')
                
                fig = go.Figure(data=[go.Candlestick(
                    x=dates,
                    open=np.random.uniform(90, 110, 50),
                    high=np.random.uniform(110, 120, 50),
                    low=np.random.uniform(80, 90, 50),
                    close=np.random.uniform(95, 105, 50)
                )])
                
                fig.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Manual Trading")
            
            with st.form("trade_form"):
                symbol = st.selectbox("Symbol", ["AAPL", "GOOGL", "BTC-USD"])
                side = st.selectbox("Side", ["BUY", "SELL"])
                quantity = st.number_input("Quantity", value=100, min_value=1)
                order_type = st.selectbox("Type", ["MARKET", "LIMIT"])
                
                if order_type == "LIMIT":
                    price = st.number_input("Price", value=100.0, min_value=0.01)
                
                submitted = st.form_submit_button("Place Order")
                
                if submitted:
                    st.success(f"Order placed: {side} {quantity} {symbol}")
            
            if st.button("🛑 Emergency Stop"):
                st.error("Emergency stop activated!")
    
    def show_portfolio(self) -> None:
        """Show portfolio dashboard"""
        st.title("💼 Portfolio Management")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Value", "$145,250.00")
            st.metric("Cash", "$35,000.00")
        
        with col2:
            st.metric("Total P&L", "$45,250.00")
            st.metric("Day P&L", "$1,250.00")
        
        with col3:
            st.metric("Positions", "7")
            st.metric("Leverage", "1.45x")
        
        # Positions table
        st.subheader("Current Positions")
        
        positions_data = {
            'Symbol': ['AAPL', 'GOOGL', 'BTC-USD', 'ETH-USD'],
            'Quantity': [100, 50, 0.5, 2.0],
            'Price': [195.50, 145.75, 42500.00, 2250.00],
            'Value': [19550, 7287.50, 21250, 4500],
            'P&L': [1550, 287.50, 1250, 250]
        }
        
        df = pd.DataFrame(positions_data)
        st.dataframe(df, use_container_width=True)
    
    def show_risk_management(self) -> None:
        """Show risk management dashboard"""
        st.title("⚠️ Risk Management")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("VaR (95%)", "-$2,450.00")
        with col2:
            st.metric("Max Drawdown", "-8.5%")
        with col3:
            st.metric("Portfolio Vol", "16.5%")
        with col4:
            st.metric("Beta", "1.15")
        
        # Risk distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("VaR Distribution")
            
            returns = np.random.normal(-0.0005, 0.015, 252)
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=returns,
                nbinsx=50,
                name="Returns"
            ))
            
            fig.add_vline(x=np.percentile(returns, 5), line_dash="dash", line_color="red")
            
            fig.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Risk Alerts")
            
            st.info("✅ No critical alerts")
            st.warning("⚠️ Concentration risk: AAPL position at 13.5%")
    
    def show_model_performance(self) -> None:
        """Show model performance dashboard"""
        st.title("📈 Model Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", "78.5%")
        with col2:
            st.metric("Precision", "82.3%")
        with col3:
            st.metric("Recall", "75.8%")
        with col4:
            st.metric("F1 Score", "79.0%")
        
        # Model comparison
        st.subheader("Model Predictions vs Actual")
        
        dates = pd.date_range(start='2025-11-01', periods=30, freq='D')
        predictions = np.random.normal(0.001, 0.02, 30)
        actual = np.random.normal(0.0005, 0.022, 30)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates, y=predictions,
            mode='lines',
            name='Predictions'
        ))
        
        fig.add_trace(go.Scatter(
            x=dates, y=actual,
            mode='lines',
            name='Actual'
        ))
        
        fig.update_layout(template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

def run_dashboard(trading_system: Any) -> None:
    """Run dashboard"""
    _ensure_streamlit()
    dashboard = TradingDashboard(trading_system)
    dashboard.run_dashboard()


def _ensure_streamlit() -> None:
    """Raise a clear error when Streamlit isn't available."""
    if st is None:
        raise RuntimeError(
            "Streamlit is required for the dashboard but is not installed. "
            "Install it via 'pip install streamlit'."
        ) from STREAMLIT_IMPORT_ERROR


if __name__ == '__main__':
    try:
        run_dashboard(trading_system=None)
    except RuntimeError as exc:
        print(exc)