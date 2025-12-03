"""
ui_components.py
Reusable UI components for wxPython dashboard
"""

import wx
from typing import Callable, Any
from wx_utils import Colors, create_button


class TradingForm(wx.Panel):
    """Form panel for manual trading"""
    
    def __init__(self, parent: wx.Window, on_submit: Callable[[dict], None]):
        """
        Args:
            on_submit: Callback function when form is submitted, receives dict with form data
        """
        super().__init__(parent)
        self.SetBackgroundColour(Colors.PANEL_BG)
        self.on_submit = on_submit
        
        # Create form elements
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Symbol selection
        symbol_label = wx.StaticText(self, label="Symbol:")
        symbol_label.SetForegroundColour(Colors.TEXT_MAIN)
        self.symbol_choice = wx.Choice(self, choices=["AAPL", "GOOGL", "BTC-USD", "ETH-USD"])
        self.symbol_choice.SetSelection(0)
        
        sizer.Add(symbol_label, 0, wx.ALL, 5)
        sizer.Add(self.symbol_choice, 0, wx.EXPAND | wx.ALL, 5)
        
        # Side selection (BUY/SELL)
        side_label = wx.StaticText(self, label="Side:")
        side_label.SetForegroundColour(Colors.TEXT_MAIN)
        self.side_choice = wx.Choice(self, choices=["BUY", "SELL"])
        self.side_choice.SetSelection(0)
        
        sizer.Add(side_label, 0, wx.ALL, 5)
        sizer.Add(self.side_choice, 0, wx.EXPAND | wx.ALL, 5)
        
        # Quantity
        qty_label = wx.StaticText(self, label="Quantity:")
        qty_label.SetForegroundColour(Colors.TEXT_MAIN)
        self.qty_spin = wx.SpinCtrl(self, value="100", min=1, max=100000)
        
        sizer.Add(qty_label, 0, wx.ALL, 5)
        sizer.Add(self.qty_spin, 0, wx.EXPAND | wx.ALL, 5)
        
        # Order type
        type_label = wx.StaticText(self, label="Order Type:")
        type_label.SetForegroundColour(Colors.TEXT_MAIN)
        self.type_choice = wx.Choice(self, choices=["MARKET", "LIMIT"])
        self.type_choice.SetSelection(0)
        self.type_choice.Bind(wx.EVT_CHOICE, self.on_type_changed)
        
        sizer.Add(type_label, 0, wx.ALL, 5)
        sizer.Add(self.type_choice, 0, wx.EXPAND | wx.ALL, 5)
        
        # Limit price (hidden by default)
        self.price_label = wx.StaticText(self, label="Limit Price:")
        self.price_label.SetForegroundColour(Colors.TEXT_MAIN)
        self.price_spin = wx.SpinCtrlDouble(self, value="100.00", min=0.01, max=1000000.00, inc=0.01)
        
        self.price_label.Hide()
        self.price_spin.Hide()
        
        sizer.Add(self.price_label, 0, wx.ALL, 5)
        sizer.Add(self.price_spin, 0, wx.EXPAND | wx.ALL, 5)
        
        # Submit button
        submit_btn = create_button(self, "Place Order", Colors.ACCENT)
        submit_btn.Bind(wx.EVT_BUTTON, self.on_submit_clicked)
        
        sizer.Add(submit_btn, 0, wx.EXPAND | wx.ALL, 10)
        
        self.SetSizer(sizer)
    
    def on_type_changed(self, event: wx.Event) -> None:
        """Handle order type change"""
        if self.type_choice.GetStringSelection() == "LIMIT":
            self.price_label.Show()
            self.price_spin.Show()
        else:
            self.price_label.Hide()
            self.price_spin.Hide()
        
        self.Layout()
    
    def on_submit_clicked(self, event: wx.Event) -> None:
        """Handle form submission"""
        data = {
            'symbol': self.symbol_choice.GetStringSelection(),
            'side': self.side_choice.GetStringSelection(),
            'quantity': self.qty_spin.GetValue(),
            'type': self.type_choice.GetStringSelection()
        }
        
        if data['type'] == "LIMIT":
            data['price'] = self.price_spin.GetValue()
        
        self.on_submit(data)


class EmergencyStopButton(wx.Panel):
    """Emergency stop button with confirmation"""
    
    def __init__(self, parent: wx.Window, on_stop: Callable[[], None]):
        """
        Args:
            on_stop: Callback function when emergency stop is confirmed
        """
        super().__init__(parent)
        self.SetBackgroundColour(Colors.PANEL_BG)
        self.on_stop = on_stop
        
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Emergency stop button
        stop_btn = wx.Button(self, label="🛑 Emergency Stop")
        stop_btn.SetBackgroundColour(Colors.DANGER)
        stop_btn.SetForegroundColour(Colors.TEXT_MAIN)
        
        font = stop_btn.GetFont()
        font.SetPointSize(12)
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        stop_btn.SetFont(font)
        
        stop_btn.Bind(wx.EVT_BUTTON, self.on_stop_clicked)
        
        sizer.Add(stop_btn, 0, wx.EXPAND | wx.ALL, 10)
        self.SetSizer(sizer)
    
    def on_stop_clicked(self, event: wx.Event) -> None:
        """Handle emergency stop click with confirmation"""
        dlg = wx.MessageDialog(
            self,
            "Are you sure you want to activate emergency stop?\n\nThis will halt all trading operations.",
            "Confirm Emergency Stop",
            wx.YES_NO | wx.NO_DEFAULT | wx.ICON_WARNING
        )
        
        if dlg.ShowModal() == wx.ID_YES:
            self.on_stop()
            wx.MessageBox(
                "Emergency stop activated!",
                "Emergency Stop",
                wx.OK | wx.ICON_ERROR
            )
        
        dlg.Destroy()


class AutoRefreshTimer:
    """Timer for auto-refreshing dashboard data"""
    
    def __init__(self, window: wx.Window, callback: Callable[[], None], interval_ms: int = 1000):
        """
        Args:
            window: Parent window
            callback: Function to call on each interval
            interval_ms: Refresh interval in milliseconds
        """
        self.window = window
        self.callback = callback
        self.interval_ms = interval_ms
        self.timer = wx.Timer(window)
        self.enabled = False
        
        window.Bind(wx.EVT_TIMER, self.on_timer, self.timer)
    
    def start(self) -> None:
        """Start auto-refresh"""
        if not self.enabled:
            self.timer.Start(self.interval_ms)
            self.enabled = True
    
    def stop(self) -> None:
        """Stop auto-refresh"""
        if self.enabled:
            self.timer.Stop()
            self.enabled = False
    
    def toggle(self) -> bool:
        """
        Toggle auto-refresh on/off
        
        Returns:
            New enabled state
        """
        if self.enabled:
            self.stop()
        else:
            self.start()
        return self.enabled
    
    def on_timer(self, event: wx.Event) -> None:
        """Handle timer event"""
        self.callback()


class NavigationButton(wx.Button):
    """Styled navigation button for sidebar"""
    
    def __init__(self, parent: wx.Window, label: str, icon: str = ""):
        """
        Args:
            parent: Parent window
            label: Button label
            icon: Optional emoji icon
        """
        display_label = f"{icon} {label}" if icon else label
        super().__init__(parent, label=display_label)
        
        self.SetBackgroundColour(Colors.PANEL_BG)
        self.SetForegroundColour(Colors.TEXT_MAIN)
        
        font = self.GetFont()
        font.SetPointSize(10)
        font.SetWeight(wx.FONTWEIGHT_NORMAL)
        self.SetFont(font)
        
        self.is_active = False
    
    def set_active(self, active: bool) -> None:
        """Set button active state"""
        self.is_active = active
        
        if active:
            self.SetBackgroundColour(Colors.ACCENT)
            self.SetForegroundColour(Colors.TEXT_MAIN)
            font = self.GetFont()
            font.SetWeight(wx.FONTWEIGHT_BOLD)
            self.SetFont(font)
        else:
            self.SetBackgroundColour(Colors.PANEL_BG)
            self.SetForegroundColour(Colors.TEXT_MAIN)
            font = self.GetFont()
            font.SetWeight(wx.FONTWEIGHT_NORMAL)
            self.SetFont(font)
        
        self.Refresh()
