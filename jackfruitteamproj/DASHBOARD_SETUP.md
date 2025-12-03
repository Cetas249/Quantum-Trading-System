# Dashboard Setup

## wxPython Dashboard (Recommended)

The trading dashboard is built with wxPython for a native desktop experience.

### Installation

```powershell
cd C:\Users\cetas\Documents\python\jackfruitteamproj
pip install wxPython>=4.2.0
```

**Note**: If wxPython fails to install on Python 3.14, you may need to:
- Use a nightly build: `pip install -U -f https://wxpython.org/Phoenix/snapshot-builds/ wxPython`
- Or use Python 3.12 which has official wxPython support

### Running the Dashboard

```powershell
python dashboard_wx.py
```

Or integrate with the trading system:
```powershell
# Modify main.py to launch wxPython dashboard
python main.py
```

### Features
- 📊 **Overview**: Portfolio performance and strategy allocation charts
- 🔴 **Live Trading**: Real-time candlestick charts and manual order entry
- 💼 **Portfolio**: Current positions table and metrics
- ⚠️ **Risk Management**: VaR distribution and risk alerts  
- 📈 **Model Performance**: Predictions vs actual comparison

---

## Streamlit Dashboard (Legacy - Optional)

If you prefer the web-based Streamlit interface:

```powershell
streamlit run dashboard.py
```

**Note**: The Streamlit version is kept for backward compatibility but is not actively maintained. The wxPython version is recommended for production use.

---

## Quick Comparison

| Feature | wxPython (Recommended) | Streamlit (Legacy) |
|---------|------------------------|---------------------|
| **Type** | Native Desktop App | Web Browser App |
| **Python 3.14** | ✅ Full Support | ⚠️ Limited |
| **Installation** | Simple (`pip install`) | Simple (`pip install`) |
| **Performance** | ✅ Excellent | Good |
| **Browser Required** | ❌ No | ✅ Yes |
| **Auto-refresh** | ✅ Built-in | ✅ Built-in |
| **All Features** | ✅ Complete | ✅ Complete |

---

## Notes

- ✅ Virtual environments are **not required** - both dashboards work with system Python
- ✅ All colors and styling match between wxPython and Streamlit versions
- ✅ wxPython provides better performance and native OS integration
- ℹ️ Both dashboards have identical functionality
