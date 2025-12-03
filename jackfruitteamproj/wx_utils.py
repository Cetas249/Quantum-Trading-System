"""
wx_utils.py
Utility functions and constants for wxPython dashboard
"""

import wx
from typing import Any
import plotly.graph_objects as go


# Color scheme matching Streamlit dark theme exactly
class Colors:
    """Color constants for the dashboard - matches Streamlit dark theme"""
    # Background colors - Streamlit dark theme
    BG_DARK = wx.Colour(14, 17, 23)  # Streamlit's main background #0e1117
    PANEL_BG = wx.Colour(38, 39, 48)  # Streamlit's secondary background #262730
    
    # Accent colors - Streamlit palette
    ACCENT = wx.Colour(255, 75, 75)  # Streamlit's primary red #FF4B4B
    SUCCESS = wx.Colour(0, 255, 0)  # Bright green for profit
    DANGER = wx.Colour(255, 0, 0)  # Bright red for loss
    WARNING = wx.Colour(255, 255, 0)  # Yellow for neutral
    PURPLE = wx.Colour(168, 85, 247)  # Purple accent
    CYAN = wx.Colour(33, 195, 228)  # Streamlit's cyan #21C3E4
    
    # Text colors - Streamlit theme
    TEXT_MAIN = wx.Colour(250, 250, 250)  # Streamlit's main text #FAFAFA
    TEXT_DIM = wx.Colour(163, 168, 184)  # Streamlit's secondary text #A3A8B8
    
    # Border
    BORDER = wx.Colour(77, 77, 77, 80)  # Streamlit's border color with alpha
    
    # Gradient colors (for metrics cards) - Streamlit gradient
    GRADIENT_START = wx.Colour(102, 126, 234)  # #667eea
    GRADIENT_END = wx.Colour(118, 75, 162)  # #764ba2


def create_metric_colors(value: float, delta: float | None = None) -> wx.Colour:
    """
    Get color for metric based on value delta
    
    Args:
        value: Metric value
        delta: Change in value (positive = green, negative = red)
        
    Returns:
        Color for the metric
    """
    if delta is None:
        return Colors.TEXT_MAIN
    
    if delta > 0:
        return Colors.SUCCESS
    elif delta < 0:
        return Colors.DANGER
    else:
        return Colors.WARNING


def format_currency(value: float, decimals: int = 2) -> str:
    """Format value as currency"""
    return f"${value:,.{decimals}f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format value as percentage"""
    return f"{value:.{decimals}f}%"


def format_number(value: float, decimals: int = 2) -> str:
    """Format number with commas"""
    return f"{value:,.{decimals}f}"


def plotly_to_html(fig: go.Figure) -> str:
    """
    Convert Plotly figure to HTML string for embedding in WebView
    Matches Streamlit's plotly_dark theme exactly
    
    Args:
        fig: Plotly figure object
        
    Returns:
        HTML string with embedded Plotly chart
    """
    # Update layout for dark theme matching Streamlit
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='#0e1117',  # Streamlit's main background
        plot_bgcolor='#0e1117',   # Streamlit's plot background
        font=dict(
            color='#fafafa',      # Streamlit's text color
            family='sans-serif'
        ),
        margin=dict(l=40, r=40, t=40, b=40),
        colorway=['#FF4B4B', '#21C3E4', '#00ff00', '#A855F7', '#FFFF00']  # Streamlit colors
    )
    
    # Convert to HTML
    html = fig.to_html(
        include_plotlyjs='cdn',
        config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d']
        }
    )
    
    return html


def create_bordered_panel(parent: wx.Window, border_color: wx.Colour = None) -> wx.Panel:
    """
    Create a panel with border styling
    
    Args:
        parent: Parent window
        border_color: Border color (default: Colors.BORDER)
        
    Returns:
        Styled panel
    """
    if border_color is None:
        border_color = Colors.BORDER
    
    panel = wx.Panel(parent)
    panel.SetBackgroundColour(Colors.PANEL_BG)
    
    return panel


def add_padding(sizer: wx.Sizer, padding: int = 10) -> None:
    """
    Add padding to all sides of a sizer
    
    Args:
        sizer: Sizer to add padding to
        padding: Padding amount in pixels
    """
    # This is handled by border flags in Add() calls
    pass


def create_titled_section(parent: wx.Window, title: str) -> tuple[wx.StaticBox, wx.StaticBoxSizer]:
    """
    Create a titled section box
    
    Args:
        parent: Parent window
        title: Section title
        
    Returns:
        Tuple of (StaticBox, StaticBoxSizer)
    """
    box = wx.StaticBox(parent, label=title)
    box.SetForegroundColour(Colors.ACCENT)
    box.SetBackgroundColour(Colors.PANEL_BG)
    
    font = box.GetFont()
    font.SetPointSize(10)
    font.SetWeight(wx.FONTWEIGHT_BOLD)
    box.SetFont(font)
    
    sizer = wx.StaticBoxSizer(box, wx.VERTICAL)
    
    return box, sizer


def create_button(parent: wx.Window, label: str, color: wx.Colour = None) -> wx.Button:
    """
    Create a styled button
    
    Args:
        parent: Parent window
        label: Button label
        color: Button color (default: Colors.ACCENT)
        
    Returns:
        Styled button
    """
    if color is None:
        color = Colors.ACCENT
    
    button = wx.Button(parent, label=label)
    button.SetBackgroundColour(color)
    button.SetForegroundColour(Colors.TEXT_MAIN)
    
    font = button.GetFont()
    font.SetWeight(wx.FONTWEIGHT_BOLD)
    button.SetFont(font)
    
    return button


def set_dark_theme(window: wx.Window) -> None:
    """
    Apply dark theme to window and all children
    
    Args:
        window: Window to apply theme to
    """
    window.SetBackgroundColour(Colors.BG_DARK)
    window.SetForegroundColour(Colors.TEXT_MAIN)
    
    # Apply to all children recursively
    for child in window.GetChildren():
        if isinstance(child, wx.Panel):
            child.SetBackgroundColour(Colors.PANEL_BG)
            child.SetForegroundColour(Colors.TEXT_MAIN)
        elif isinstance(child, wx.StaticText):
            child.SetForegroundColour(Colors.TEXT_MAIN)
