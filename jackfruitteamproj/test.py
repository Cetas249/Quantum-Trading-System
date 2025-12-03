import pytest
import wx
from wx_utils import Colors, format_currency
from chart_widgets import MetricCard
from ui_components import TradingForm
def test_colors():
    """Test color values"""
    assert Colors.BG_DARK == wx.Colour(14, 17, 23)
    assert Colors.ACCENT == wx.Colour(255, 75, 75)
def test_format_currency():
    """Test currency formatting"""
    assert format_currency(1234.56) == "$1,234.56"
    assert format_currency(1000000) == "$1,000,000.00"
def test_metric_card():
    """Test MetricCard creation"""
    app = wx.App()
    frame = wx.Frame(None)
    card = MetricCard(frame, "Test Metric", "$100", "+$10")
    assert card is not None
    app.Destroy()
# Run with: pytest tests/ -v