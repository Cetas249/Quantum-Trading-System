"""
dashboard_wx.py
wxPython-based trading dashboard - conversion from Streamlit
Python 3.14 compatible
"""

import wx
import wx.lib.scrolledpanel as scrolled
from datetime import datetime, timedelta
from typing import Any, Optional
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from wx_utils import Colors, set_dark_theme, create_titled_section, create_button
from chart_widgets import PlotlyWebViewPanel, MetricCard, DataGridPanel, MetricsRow
from ui_components import TradingForm, EmergencyStopButton, AutoRefreshTimer, NavigationButton


class OverviewPanel(scrolled.ScrolledPanel):
    """Overview dashboard page"""
    
    def __init__(self, parent: wx.Window):
        super().__init__(parent)
        self.SetBackgroundColour(Colors.BG_DARK)
        
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Title
        title = wx.StaticText(self, label="📊 Trading System Overview")
        title.SetForegroundColour(Colors.TEXT_MAIN)
        font = title.GetFont()
        font.SetPointSize(16)
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        title.SetFont(font)
        main_sizer.Add(title, 0, wx.ALL, 10)
        
        # Metrics row
        metrics_data = [
            {'label': 'Daily P&L', 'value': '$1,250.00', 'delta': '+$150'},
            {'label': 'Total Return', 'value': '15.50%', 'delta': '+1.2%'},
            {'label': 'Sharpe Ratio', 'value': '1.85'},
            {'label': 'Win Rate', 'value': '62.5%'},
            {'label': 'Active Positions', 'value': '7'}
        ]
        
        self.metrics_row = MetricsRow(self, metrics_data)
        main_sizer.Add(self.metrics_row, 0, wx.EXPAND | wx.ALL, 10)
        
        # Charts row
        charts_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # Portfolio performance chart
        portfolio_panel = wx.Panel(self)
        portfolio_panel.SetBackgroundColour(Colors.PANEL_BG)
        portfolio_sizer = wx.BoxSizer(wx.VERTICAL)
        
        portfolio_title = wx.StaticText(portfolio_panel, label="Portfolio Performance")
        portfolio_title.SetForegroundColour(Colors.ACCENT)
        font = portfolio_title.GetFont()
        font.SetPointSize(12)
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        portfolio_title.SetFont(font)
        portfolio_sizer.Add(portfolio_title, 0, wx.ALL, 10)
        
        # Create portfolio chart
        self.portfolio_chart = PlotlyWebViewPanel(portfolio_panel)
        self.update_portfolio_chart()
        portfolio_sizer.Add(self.portfolio_chart, 1, wx.EXPAND | wx.ALL, 5)
        
        portfolio_panel.SetSizer(portfolio_sizer)
        charts_sizer.Add(portfolio_panel, 1, wx.EXPAND | wx.ALL, 5)
        
        # Strategy allocation chart
        strategy_panel = wx.Panel(self)
        strategy_panel.SetBackgroundColour(Colors.PANEL_BG)
        strategy_sizer = wx.BoxSizer(wx.VERTICAL)
        
        strategy_title = wx.StaticText(strategy_panel, label="Strategy Allocation")
        strategy_title.SetForegroundColour(Colors.ACCENT)
        font = strategy_title.GetFont()
        font.SetPointSize(12)
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        strategy_title.SetFont(font)
        strategy_sizer.Add(strategy_title, 0, wx.ALL, 10)
        
        # Create strategy chart
        self.strategy_chart = PlotlyWebViewPanel(strategy_panel)
        self.update_strategy_chart()
        strategy_sizer.Add(self.strategy_chart, 1, wx.EXPAND | wx.ALL, 5)
        
        strategy_panel.SetSizer(strategy_sizer)
        charts_sizer.Add(strategy_panel, 1, wx.EXPAND | wx.ALL, 5)
        
        main_sizer.Add(charts_sizer, 1, wx.EXPAND | wx.ALL, 5)
        
        self.SetSizer(main_sizer)
        self.SetupScrolling()
    
    def update_portfolio_chart(self) -> None:
        """Update portfolio performance chart"""
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
            xaxis_title="Date",
            yaxis_title="Value ($)",
            height=400
        )
        
        self.portfolio_chart.update_figure(fig)
    
    def update_strategy_chart(self) -> None:
        """Update strategy allocation chart"""
        strategies = ['Quantum Opt', 'ML Ensemble', 'Sentiment', 'RL Agent']
        values = [30, 25, 25, 20]
        
        fig = px.pie(
            values=values,
            names=strategies,
            title="Strategy Allocation"
        )
        fig.update_layout(height=400)
        
        self.strategy_chart.update_figure(fig)


class LiveTradingPanel(scrolled.ScrolledPanel):
    """Live trading page"""
    
    def __init__(self, parent: wx.Window):
        super().__init__(parent)
        self.SetBackgroundColour(Colors.BG_DARK)
        
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Title
        title = wx.StaticText(self, label="🔴 Live Trading")
        title.SetForegroundColour(Colors.TEXT_MAIN)
        font = title.GetFont()
        font.SetPointSize(16)
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        title.SetFont(font)
        main_sizer.Add(title, 0, wx.ALL, 10)
        
        # Main content
        content_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # Left: Market data
        market_panel = wx.Panel(self)
        market_panel.SetBackgroundColour(Colors.PANEL_BG)
        market_sizer = wx.BoxSizer(wx.VERTICAL)
        
        market_title = wx.StaticText(market_panel, label="Market Data")
        market_title.SetForegroundColour(Colors.ACCENT)
        font = market_title.GetFont()
        font.SetPointSize(12)
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        market_title.SetFont(font)
        market_sizer.Add(market_title, 0, wx.ALL, 10)
        
        # Symbol selection
        symbol_label = wx.StaticText(market_panel, label="Select Symbols:")
        symbol_label.SetForegroundColour(Colors.TEXT_MAIN)
        market_sizer.Add(symbol_label, 0, wx.ALL, 5)
        
        self.symbol_list = wx.CheckListBox(
            market_panel,
            choices=["AAPL", "GOOGL", "BTC-USD", "ETH-USD"]
        )
        self.symbol_list.Check(0)  # Check AAPL by default
        self.symbol_list.Bind(wx.EVT_CHECKLISTBOX, self.on_symbol_changed)
        market_sizer.Add(self.symbol_list, 0, wx.EXPAND | wx.ALL, 5)
        
        # Candlestick chart
        self.candlestick_chart = PlotlyWebViewPanel(market_panel)
        self.update_candlestick_chart()
        market_sizer.Add(self.candlestick_chart, 1, wx.EXPAND | wx.ALL, 5)
        
        market_panel.SetSizer(market_sizer)
        content_sizer.Add(market_panel, 2, wx.EXPAND | wx.ALL, 5)
        
        # Right: Manual trading
        trading_panel = wx.Panel(self)
        trading_panel.SetBackgroundColour(Colors.PANEL_BG)
        trading_sizer = wx.BoxSizer(wx.VERTICAL)
        
        trading_title = wx.StaticText(trading_panel, label="Manual Trading")
        trading_title.SetForegroundColour(Colors.ACCENT)
        font = trading_title.GetFont()
        font.SetPointSize(12)
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        trading_title.SetFont(font)
        trading_sizer.Add(trading_title, 0, wx.ALL, 10)
        
        # Trading form
        self.trading_form = TradingForm(trading_panel, self.on_order_submit)
        trading_sizer.Add(self.trading_form, 0, wx.EXPAND | wx.ALL, 10)
        
        # Emergency stop button
        self.stop_button = EmergencyStopButton(trading_panel, self.on_emergency_stop)
        trading_sizer.Add(self.stop_button, 0, wx.EXPAND | wx.ALL, 10)
        
        trading_panel.SetSizer(trading_sizer)
        content_sizer.Add(trading_panel, 1, wx.EXPAND | wx.ALL, 5)
        
        main_sizer.Add(content_sizer, 1, wx.EXPAND | wx.ALL, 5)
        
        self.SetSizer(main_sizer)
        self.SetupScrolling()
    
    def update_candlestick_chart(self) -> None:
        """Update candlestick chart"""
        dates = pd.date_range(start='2025-11-01', periods=50, freq='D')
        
        fig = go.Figure(data=[go.Candlestick(
            x=dates,
            open=np.random.uniform(90, 110, 50),
            high=np.random.uniform(110, 120, 50),
            low=np.random.uniform(80, 90, 50),
            close=np.random.uniform(95, 105, 50)
        )])
        
        fig.update_layout(
            title="AAPL Price Chart",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=500
        )
        
        self.candlestick_chart.update_figure(fig)
    
    def on_symbol_changed(self, event: wx.Event) -> None:
        """Handle symbol selection change"""
        checked = [self.symbol_list.GetString(i) for i in range(self.symbol_list.GetCount())
                   if self.symbol_list.IsChecked(i)]
        if checked:
            self.update_candlestick_chart()
    
    def on_order_submit(self, data: dict) -> None:
        """Handle order form submission"""
        msg = f"Order placed: {data['side']} {data['quantity']} {data['symbol']}"
        if data['type'] == 'LIMIT':
            msg += f" at ${data['price']:.2f}"
        
        wx.MessageBox(msg, "Order Submitted", wx.OK | wx.ICON_INFORMATION)
    
    def on_emergency_stop(self) -> None:
        """Handle emergency stop activation"""
        print("Emergency stop activated!")


class PortfolioPanel(scrolled.ScrolledPanel):
    """Portfolio management page"""
    
    def __init__(self, parent: wx.Window):
        super().__init__(parent)
        self.SetBackgroundColour(Colors.BG_DARK)
        
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Title
        title = wx.StaticText(self, label="💼 Portfolio Management")
        title.SetForegroundColour(Colors.TEXT_MAIN)
        font = title.GetFont()
        font.SetPointSize(16)
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        title.SetFont(font)
        main_sizer.Add(title, 0, wx.ALL, 10)
        
        # Metrics
        metrics_grid = wx.GridSizer(2, 3, 10, 10)
        
        metrics = [
            {'label': 'Total Value', 'value': '$145,250.00'},
            {'label': 'Cash', 'value': '$35,000.00'},
            {'label': 'Total P&L', 'value': '$45,250.00', 'delta': '+$2,100'},
            {'label': 'Day P&L', 'value': '$1,250.00', 'delta': '+$150'},
            {'label': 'Positions', 'value': '7'},
            {'label': 'Leverage', 'value': '1.45x'}
        ]
        
        for metric in metrics:
            card = MetricCard(
                self,
                label=metric['label'],
                value=metric['value'],
                delta=metric.get('delta')
            )
            metrics_grid.Add(card, 0, wx.EXPAND | wx.ALL, 5)
        
        main_sizer.Add(metrics_grid, 0, wx.EXPAND | wx.ALL, 10)
        
        # Positions table
        positions_title = wx.StaticText(self, label="Current Positions")
        positions_title.SetForegroundColour(Colors.ACCENT)
        font = positions_title.GetFont()
        font.SetPointSize(12)
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        positions_title.SetFont(font)
        main_sizer.Add(positions_title, 0, wx.ALL, 10)
        
        positions_data = {
            'Symbol': ['AAPL', 'GOOGL', 'BTC-USD', 'ETH-USD'],
            'Quantity': [100, 50, 0.5, 2.0],
            'Price': [195.50, 145.75, 42500.00, 2250.00],
            'Value': [19550, 7287.50, 21250, 4500],
            'P&L': [1550, 287.50, 1250, 250]
        }
        
        df = pd.DataFrame(positions_data)
        self.positions_grid = DataGridPanel(self, df)
        main_sizer.Add(self.positions_grid, 1, wx.EXPAND | wx.ALL, 10)
        
        self.SetSizer(main_sizer)
        self.SetupScrolling()


class RiskManagementPanel(scrolled.ScrolledPanel):
    """Risk management page"""
    
    def __init__(self, parent: wx.Window):
        super().__init__(parent)
        self.SetBackgroundColour(Colors.BG_DARK)
        
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Title
        title = wx.StaticText(self, label="⚠️ Risk Management")
        title.SetForegroundColour(Colors.TEXT_MAIN)
        font = title.GetFont()
        font.SetPointSize(16)
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        title.SetFont(font)
        main_sizer.Add(title, 0, wx.ALL, 10)
        
        # Metrics row
        metrics_data = [
            {'label': 'VaR (95%)', 'value': '-$2,450.00'},
            {'label': 'Max Drawdown', 'value': '-8.5%'},
            {'label': 'Portfolio Vol', 'value': '16.5%'},
            {'label': 'Beta', 'value': '1.15'}
        ]
        
        metrics_row = MetricsRow(self, metrics_data)
        main_sizer.Add(metrics_row, 0, wx.EXPAND | wx.ALL, 10)
        
        # Charts and alerts
        content_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # VaR distribution
        var_panel = wx.Panel(self)
        var_panel.SetBackgroundColour(Colors.PANEL_BG)
        var_sizer = wx.BoxSizer(wx.VERTICAL)
        
        var_title = wx.StaticText(var_panel, label="VaR Distribution")
        var_title.SetForegroundColour(Colors.ACCENT)
        font = var_title.GetFont()
        font.SetPointSize(12)
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        var_title.SetFont(font)
        var_sizer.Add(var_title, 0, wx.ALL, 10)
        
        self.var_chart = PlotlyWebViewPanel(var_panel)
        self.update_var_chart()
        var_sizer.Add(self.var_chart, 1, wx.EXPAND | wx.ALL, 5)
        
        var_panel.SetSizer(var_sizer)
        content_sizer.Add(var_panel, 1, wx.EXPAND | wx.ALL, 5)
        
        # Risk alerts
        alerts_panel = wx.Panel(self)
        alerts_panel.SetBackgroundColour(Colors.PANEL_BG)
        alerts_sizer = wx.BoxSizer(wx.VERTICAL)
        
        alerts_title = wx.StaticText(alerts_panel, label="Risk Alerts")
        alerts_title.SetForegroundColour(Colors.ACCENT)
        font = alerts_title.GetFont()
        font.SetPointSize(12)
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        alerts_title.SetFont(font)
        alerts_sizer.Add(alerts_title, 0, wx.ALL, 10)
        
        alert1 = wx.StaticText(alerts_panel, label="✅ No critical alerts")
        alert1.SetForegroundColour(Colors.SUCCESS)
        alerts_sizer.Add(alert1, 0, wx.ALL, 10)
        
        alert2 = wx.StaticText(alerts_panel, label="⚠️ Concentration risk: AAPL position at 13.5%")
        alert2.SetForegroundColour(Colors.WARNING)
        alerts_sizer.Add(alert2, 0, wx.ALL, 10)
        
        alerts_panel.SetSizer(alerts_sizer)
        content_sizer.Add(alerts_panel, 1, wx.EXPAND | wx.ALL, 5)
        
        main_sizer.Add(content_sizer, 1, wx.EXPAND | wx.ALL, 5)
        
        self.SetSizer(main_sizer)
        self.SetupScrolling()
    
    def update_var_chart(self) -> None:
        """Update VaR distribution chart"""
        returns = np.random.normal(-0.0005, 0.015, 252)
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=returns,
            nbinsx=50,
            name="Returns"
        ))
        
        fig.add_vline(x=np.percentile(returns, 5), line_dash="dash", line_color="red")
        
        fig.update_layout(
            title="Return Distribution",
            xaxis_title="Returns",
            yaxis_title="Frequency",
            height=400
        )
        
        self.var_chart.update_figure(fig)


class ModelPerformancePanel(scrolled.ScrolledPanel):
    """Model performance page"""
    
    def __init__(self, parent: wx.Window):
        super().__init__(parent)
        self.SetBackgroundColour(Colors.BG_DARK)
        
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Title
        title = wx.StaticText(self, label="📈 Model Performance")
        title.SetForegroundColour(Colors.TEXT_MAIN)
        font = title.GetFont()
        font.SetPointSize(16)
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        title.SetFont(font)
        main_sizer.Add(title, 0, wx.ALL, 10)
        
        # Metrics row
        metrics_data = [
            {'label': 'Accuracy', 'value': '78.5%'},
            {'label': 'Precision', 'value': '82.3%'},
            {'label': 'Recall', 'value': '75.8%'},
            {'label': 'F1 Score', 'value': '79.0%'}
        ]
        
        metrics_row = MetricsRow(self, metrics_data)
        main_sizer.Add(metrics_row, 0, wx.EXPAND | wx.ALL, 10)
        
        # Predictions vs Actual chart
        chart_title = wx.StaticText(self, label="Model Predictions vs Actual")
        chart_title.SetForegroundColour(Colors.ACCENT)
        font = chart_title.GetFont()
        font.SetPointSize(12)
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        chart_title.SetFont(font)
        main_sizer.Add(chart_title, 0, wx.ALL, 10)
        
        self.predictions_chart = PlotlyWebViewPanel(self)
        self.update_predictions_chart()
        main_sizer.Add(self.predictions_chart, 1, wx.EXPAND | wx.ALL, 10)
        
        self.SetSizer(main_sizer)
        self.SetupScrolling()
    
    def update_predictions_chart(self) -> None:
        """Update predictions vs actual chart"""
        dates = pd.date_range(start='2025-11-01', periods=30, freq='D')
        predictions = np.random.normal(0.001, 0.02, 30)
        actual = np.random.normal(0.0005, 0.022, 30)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates, y=predictions,
            mode='lines',
            name='Predictions',
            line=dict(color=Colors.ACCENT.GetAsString(wx.C2S_HTML_SYNTAX))
        ))
        
        fig.add_trace(go.Scatter(
            x=dates, y=actual,
            mode='lines',
            name='Actual',
            line=dict(color=Colors.SUCCESS.GetAsString(wx.C2S_HTML_SYNTAX))
        ))
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Returns",
            height=500
        )
        
        self.predictions_chart.update_figure(fig)


class NavigationPanel(wx.Panel):
    """Left sidebar navigation panel"""
    
    def __init__(self, parent: wx.Window, on_page_change: callable):
        super().__init__(parent)
        self.SetBackgroundColour(Colors.PANEL_BG)
        self.on_page_change = on_page_change
        
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Title
        title = wx.StaticText(self, label="🚀 Quantum Trading")
        title.SetForegroundColour(Colors.TEXT_MAIN)
        font = title.GetFont()
        font.SetPointSize(14)
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        title.SetFont(font)
        sizer.Add(title, 0, wx.ALL | wx.CENTER, 15)
        
        # Navigation buttons
        pages = [
            ("Overview", "📊"),
            ("Live Trading", "🔴"),
            ("Portfolio", "💼"),
            ("Risk Management", "⚠️"),
            ("Model Performance", "📈")
        ]
        
        self.nav_buttons = []
        
        for i, (page_name, icon) in enumerate(pages):
            btn = NavigationButton(self, page_name, icon)
            btn.Bind(wx.EVT_BUTTON, lambda evt, idx=i: self.on_nav_click(idx))
            self.nav_buttons.append(btn)
            sizer.Add(btn, 0, wx.EXPAND | wx.ALL, 5)
        
        # Set first button as active
        self.nav_buttons[0].set_active(True)
        self.active_index = 0
        
        # Spacer
        sizer.AddStretchSpacer()
        
        # Auto-refresh toggle
        refresh_label = wx.StaticText(self, label="Auto Refresh:")
        refresh_label.SetForegroundColour(Colors.TEXT_DIM)
        sizer.Add(refresh_label, 0, wx.ALL, 10)
        
        self.refresh_checkbox = wx.CheckBox(self, label="Enabled")
        self.refresh_checkbox.SetForegroundColour(Colors.TEXT_MAIN)
        self.refresh_checkbox.SetValue(True)
        sizer.Add(self.refresh_checkbox, 0, wx.ALL, 10)
        
        self.SetSizer(sizer)
    
    def on_nav_click(self, index: int) -> None:
        """Handle navigation button click"""
        # Update button states
        for i, btn in enumerate(self.nav_buttons):
            btn.set_active(i == index)
        
        self.active_index = index
        self.on_page_change(index)


class MainFrame(wx.Frame):
    """Main application window"""
    
    def __init__(self, trading_system: Any = None):
        super().__init__(None, title="Quantum Trading System", size=(1400, 900))
        self.trading_system = trading_system
        
        # Set icon and background
        self.SetBackgroundColour(Colors.BG_DARK)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Main sizer
        main_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # Navigation panel (sidebar)
        self.nav_panel = NavigationPanel(self, self.on_page_change)
        main_sizer.Add(self.nav_panel, 0, wx.EXPAND | wx.ALL, 0)
        
        # Content area
        self.content_panel = wx.Panel(self)
        self.content_panel.SetBackgroundColour(Colors.BG_DARK)
        self.content_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Create all page panels
        self.pages = [
            OverviewPanel(self.content_panel),
            LiveTradingPanel(self.content_panel),
            PortfolioPanel(self.content_panel),
            RiskManagementPanel(self.content_panel),
            ModelPerformancePanel(self.content_panel)
        ]
        
        # Add all pages to sizer (initially hidden except first)
        for i, page in enumerate(self.pages):
            self.content_sizer.Add(page, 1, wx.EXPAND | wx.ALL, 0)
            if i > 0:
                page.Hide()
        
        self.content_panel.SetSizer(self.content_sizer)
        main_sizer.Add(self.content_panel, 1, wx.EXPAND | wx.ALL, 0)
        
        self.SetSizer(main_sizer)
        
        # Current page index
        self.current_page = 0
        
        # Auto-refresh timer
        self.refresh_timer = AutoRefreshTimer(self, self.on_auto_refresh, 1000)
        self.refresh_timer.start()
        
        # Center window
        self.Centre()
    
    def create_menu_bar(self) -> None:
        """Create menu bar"""
        menubar = wx.MenuBar()
        
        # File menu
        file_menu = wx.Menu()
        file_menu.Append(wx.ID_EXIT, "E&xit")
        self.Bind(wx.EVT_MENU, self.on_exit, id=wx.ID_EXIT)
        
        menubar.Append(file_menu, "&File")
        
        # View menu
        view_menu = wx.Menu()
        self.refresh_item = view_menu.AppendCheckItem(wx.ID_ANY, "Auto &Refresh")
        self.refresh_item.Check(True)
        self.Bind(wx.EVT_MENU, self.on_toggle_refresh, self.refresh_item)
        
        menubar.Append(view_menu, "&View")
        
        # Help menu
        help_menu = wx.Menu()
        help_menu.Append(wx.ID_ABOUT, "&About")
        self.Bind(wx.EVT_MENU, self.on_about, id=wx.ID_ABOUT)
        
        menubar.Append(help_menu, "&Help")
        
        self.SetMenuBar(menubar)
    
    def on_page_change(self, index: int) -> None:
        """Handle page change"""
        # Hide current page
        self.pages[self.current_page].Hide()
        
        # Show new page
        self.pages[index].Show()
        self.current_page = index
        
        # Refresh layout
        self.content_panel.Layout()
    
    def on_toggle_refresh(self, event: wx.Event) -> None:
        """Toggle auto-refresh"""
        self.refresh_timer.toggle()
    
    def on_auto_refresh(self) -> None:
        """Handle auto-refresh timer"""
        # Update current page data
        # This would fetch real data from the trading system
        pass
    
    def on_about(self, event: wx.Event) -> None:
        """Show about dialog"""
        info = wx.adv.AboutDialogInfo()
        info.SetName("Quantum Trading System")
        info.SetVersion("1.0.0")
        info.SetDescription("Advanced trading dashboard powered by Python 3.14")
        info.SetWebSite("https://github.com/yourusername/quantum-trading")
        
        wx.adv.AboutBox(info)
    
    def on_exit(self, event: wx.Event) -> None:
        """Handle exit"""
        self.Close(True)


class TradingDashboardApp(wx.App):
    """Main wxPython application"""
    
    def __init__(self, trading_system: Any = None):
        self.trading_system = trading_system
        super().__init__()
    
    def OnInit(self) -> bool:
        """Initialize application"""
        self.frame = MainFrame(self.trading_system)
        self.frame.Show()
        return True


def run_dashboard_wx(trading_system: Any = None) -> None:
    """
    Run the wx dashbaord
    
    Args:
        trading_system: Trading system instance (optional)
    """
    app = TradingDashboardApp(trading_system)
    app.MainLoop()


if __name__ == '__main__':
    run_dashboard_wx()
