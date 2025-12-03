"""
chart_widgets.py
Custom chart and display widgets for wxPython dashboard
"""

import wx
import wx.html2
import wx.grid
import plotly.graph_objects as go
from typing import Any
import pandas as pd

from wx_utils import Colors, plotly_to_html, format_currency, format_percentage, create_metric_colors


class PlotlyWebViewPanel(wx.Panel):
    """Panel that embeds Plotly charts using WebView"""
    
    def __init__(self, parent: wx.Window, fig: go.Figure | None = None):
        super().__init__(parent)
        self.SetBackgroundColour(Colors.PANEL_BG)
        
        # Create WebView
        self.webview = wx.html2.WebView.New(self)
        
        # Layout
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.webview, 1, wx.EXPAND | wx.ALL, 5)
        self.SetSizer(sizer)
        
        # Load initial figure if provided
        if fig is not None:
            self.update_figure(fig)
    
    def update_figure(self, fig: go.Figure) -> None:
        """
        Update the displayed figure
        
        Args:
            fig: Plotly figure to display
        """
        html = plotly_to_html(fig)
        self.webview.SetPage(html, "")
    
    def clear(self) -> None:
        """Clear the chart"""
        self.webview.SetPage("<html><body style='background-color: rgb(15, 23, 42);'></body></html>", "")


class MetricCard(wx.Panel):
    """Card widget for displaying a metric with label and value"""
    
    def __init__(self, parent: wx.Window, label: str, value: str, delta: str | None = None):
        super().__init__(parent)
        self.SetBackgroundColour(Colors.PANEL_BG)
        
        # Create UI elements
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Label
        label_text = wx.StaticText(self, label=label)
        label_text.SetForegroundColour(Colors.TEXT_DIM)
        font = label_text.GetFont()
        font.SetPointSize(9)
        label_text.SetFont(font)
        
        # Value
        self.value_text = wx.StaticText(self, label=value)
        self.value_text.SetForegroundColour(Colors.TEXT_MAIN)
        value_font = self.value_text.GetFont()
        value_font.SetPointSize(18)
        value_font.SetWeight(wx.FONTWEIGHT_BOLD)
        self.value_text.SetFont(value_font)
        
        # Delta (optional)
        self.delta_text = None
        if delta is not None:
            self.delta_text = wx.StaticText(self, label=delta)
            delta_font = self.delta_text.GetFont()
            delta_font.SetPointSize(10)
            self.delta_text.SetFont(delta_font)
            
            # Color based on positive/negative
            if delta.startswith('+'):
                self.delta_text.SetForegroundColour(Colors.SUCCESS)
            elif delta.startswith('-'):
                self.delta_text.SetForegroundColour(Colors.DANGER)
            else:
                self.delta_text.SetForegroundColour(Colors.WARNING)
        
        # Layout
        sizer.Add(label_text, 0, wx.ALL | wx.ALIGN_CENTER, 5)
        sizer.Add(self.value_text, 0, wx.ALL | wx.ALIGN_CENTER, 5)
        if self.delta_text is not None:
            sizer.Add(self.delta_text, 0, wx.ALL | wx.ALIGN_CENTER, 2)
        
        self.SetSizer(sizer)
    
    def update_value(self, value: str, delta: str | None = None) -> None:
        """
        Update the metric value
        
        Args:
            value: New value
            delta: New delta (optional)
        """
        self.value_text.SetLabel(value)
        
        if delta is not None and self.delta_text is not None:
            self.delta_text.SetLabel(delta)
            
            # Update color
            if delta.startswith('+'):
                self.delta_text.SetForegroundColour(Colors.SUCCESS)
            elif delta.startswith('-'):
                self.delta_text.SetForegroundColour(Colors.DANGER)
            else:
                self.delta_text.SetForegroundColour(Colors.WARNING)
        
        self.Layout()


class DataGridPanel(wx.Panel):
    """Panel with a grid for displaying tabular data"""
    
    def __init__(self, parent: wx.Window, df: pd.DataFrame | None = None):
        super().__init__(parent)
        self.SetBackgroundColour(Colors.PANEL_BG)
        
        # Create grid
        self.grid = wx.grid.Grid(self)
        self.grid.SetDefaultCellBackgroundColour(Colors.PANEL_BG)
        self.grid.SetDefaultCellTextColour(Colors.TEXT_MAIN)
        self.grid.SetLabelBackgroundColour(Colors.BG_DARK)
        self.grid.SetLabelTextColour(Colors.TEXT_MAIN)
        self.grid.SetGridLineColour(Colors.BORDER)
        
        # Layout
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.grid, 1, wx.EXPAND | wx.ALL, 5)
        self.SetSizer(sizer)
        
        # Load initial data if provided
        if df is not None:
            self.update_data(df)
    
    def update_data(self, df: pd.DataFrame) -> None:
        """
        Update grid with new data
        
        Args:
            df: Pandas DataFrame to display
        """
        # Clear existing data
        if self.grid.GetNumberRows() > 0:
            self.grid.DeleteRows(0, self.grid.GetNumberRows())
        if self.grid.GetNumberCols() > 0:
            self.grid.DeleteCols(0, self.grid.GetNumberCols())
        
        # Set dimensions
        rows, cols = df.shape
        self.grid.CreateGrid(rows, cols)
        
        # Set column headers
        for i, col in enumerate(df.columns):
            self.grid.SetColLabelValue(i, str(col))
        
        # Fill data
        for i in range(rows):
            for j in range(cols):
                value = df.iloc[i, j]
                
                # Format value
                if isinstance(value, float):
                    # Check if it's currency-like (has a column name with $ or price/value)
                    col_name = df.columns[j].lower()
                    if '$' in str(df.columns[j]) or 'price' in col_name or 'value' in col_name or 'p&l' in col_name:
                        formatted = format_currency(value)
                    else:
                        formatted = f"{value:.2f}"
                else:
                    formatted = str(value)
                
                self.grid.SetCellValue(i, j, formatted)
                
                # Color code P&L columns
                if 'p&l' in df.columns[j].lower():
                    if value > 0:
                        self.grid.SetCellTextColour(i, j, Colors.SUCCESS)
                    elif value < 0:
                        self.grid.SetCellTextColour(i, j, Colors.DANGER)
        
        # Auto-size columns
        self.grid.AutoSizeColumns()
        
        # Make read-only
        self.grid.EnableEditing(False)
        
        self.Layout()


class MetricsRow(wx.Panel):
    """Panel containing a row of metric cards"""
    
    def __init__(self, parent: wx.Window, metrics: list[dict[str, str]]):
        """
        Args:
            metrics: List of metric dicts with keys: label, value, delta (optional)
        """
        super().__init__(parent)
        self.SetBackgroundColour(Colors.BG_DARK)
        
        self.metric_cards: list[MetricCard] = []
        
        # Create horizontal sizer
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        for metric in metrics:
            card = MetricCard(
                self,
                label=metric['label'],
                value=metric['value'],
                delta=metric.get('delta')
            )
            self.metric_cards.append(card)
            sizer.Add(card, 1, wx.EXPAND | wx.ALL, 5)
        
        self.SetSizer(sizer)
    
    def update_metric(self, index: int, value: str, delta: str | None = None) -> None:
        """
        Update a specific metric
        
        Args:
            index: Index of metric to update
            value: New value
            delta: New delta (optional)
        """
        if 0 <= index < len(self.metric_cards):
            self.metric_cards[index].update_value(value, delta)
