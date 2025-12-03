"""
models/risk_manager.py
Advanced risk management for Python 3.14
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import numpy as np

class AlertLevel(Enum):
    """Risk alert levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

@dataclass
class RiskMetrics:
    """Container for risk metrics"""
    volatility: float
    downside_volatility: float
    var_historical: float
    var_parametric: float
    var_monte_carlo: float
    expected_shortfall: float
    max_drawdown: float
    calmar_ratio: float
    sharpe_ratio: float
    sortino_ratio: float
    omega_ratio: float
    skewness: float
    kurtosis: float

@dataclass
class RiskAlert:
    """Risk alert container"""
    alert_type: str
    level: AlertLevel
    message: str
    symbol: Optional[str] = None
    value: Optional[float] = None

class AdvancedRiskManager:
    """Python 3.14 risk management system"""
    
    def __init__(self, confidence_level: float = 0.05) -> None:
        self.confidence_level: float = confidence_level
        self.risk_metrics: dict[str, float] = {}
        self.alerts: list[RiskAlert] = []
    
    def calculate_comprehensive_risk_metrics(
        self,
        portfolio_returns: np.ndarray,
        benchmark_returns: Optional[np.ndarray] = None
    ) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        
        from scipy import stats
        
        volatility = float(np.std(portfolio_returns) * np.sqrt(252))
        downside_volatility = float(self._calculate_downside_volatility(portfolio_returns))
        
        var_historical = float(self._calculate_historical_var(portfolio_returns))
        var_parametric = float(self._calculate_parametric_var(portfolio_returns))
        var_monte_carlo = float(self._calculate_monte_carlo_var(portfolio_returns))
        
        expected_shortfall = float(self._calculate_expected_shortfall(portfolio_returns))
        
        max_drawdown = float(self._calculate_max_drawdown(portfolio_returns))
        calmar_ratio = float(np.mean(portfolio_returns) * 252 / abs(max_drawdown)) if max_drawdown != 0 else 0.0
        
        risk_free_rate = 0.02
        excess_returns = portfolio_returns - risk_free_rate/252
        sharpe_ratio = float(np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)) if np.std(excess_returns) != 0 else 0.0
        sortino_ratio = float(np.mean(excess_returns) / downside_volatility * np.sqrt(252)) if downside_volatility != 0 else 0.0
        
        omega_ratio = float(self._calculate_omega_ratio(portfolio_returns))
        
        skewness = float(stats.skew(portfolio_returns))
        kurtosis = float(stats.kurtosis(portfolio_returns))
        
        return RiskMetrics(
            volatility=volatility,
            downside_volatility=downside_volatility,
            var_historical=var_historical,
            var_parametric=var_parametric,
            var_monte_carlo=var_monte_carlo,
            expected_shortfall=expected_shortfall,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            omega_ratio=omega_ratio,
            skewness=skewness,
            kurtosis=kurtosis
        )
    
    def _calculate_downside_volatility(
        self,
        returns: np.ndarray,
        mar: float = 0.0
    ) -> float:
        """Calculate downside volatility"""
        downside_returns = returns[returns < mar]
        if len(downside_returns) == 0:
            return 0.0
        return float(np.std(downside_returns) * np.sqrt(252))
    
    def _calculate_historical_var(self, returns: np.ndarray) -> float:
        """Calculate Historical VaR"""
        return float(np.percentile(returns, self.confidence_level * 100))
    
    def _calculate_parametric_var(self, returns: np.ndarray) -> float:
        """Calculate Parametric VaR"""
        from scipy import stats
        mean_return = float(np.mean(returns))
        std_return = float(np.std(returns))
        z_score = float(stats.norm.ppf(self.confidence_level))
        return float(mean_return + z_score * std_return)
    
    def _calculate_monte_carlo_var(
        self,
        returns: np.ndarray,
        num_simulations: int = 10000
    ) -> float:
        """Calculate Monte Carlo VaR"""
        mean_return = float(np.mean(returns))
        std_return = float(np.std(returns))
        
        random_returns = np.random.normal(mean_return, std_return, num_simulations)
        return float(np.percentile(random_returns, self.confidence_level * 100))
    
    def _calculate_expected_shortfall(self, returns: np.ndarray) -> float:
        """Calculate Expected Shortfall"""
        var = self._calculate_historical_var(returns)
        return float(np.mean(returns[returns <= var]))
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate Maximum Drawdown"""
        cumulative_returns = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        return float(drawdown.min())
    
    def _calculate_omega_ratio(self, returns: np.ndarray, threshold: float = 0.0) -> float:
        """Calculate Omega Ratio"""
        returns_above = returns[returns > threshold] - threshold
        returns_below = threshold - returns[returns < threshold]
        
        if len(returns_below) == 0:
            return float('inf')
        
        return float(np.sum(returns_above) / np.sum(returns_below))
    
    def generate_risk_alerts(
        self,
        current_positions: dict[str, dict[str, float]],
        market_data: dict[str, dict[str, Any]],
        portfolio_value: float
    ) -> list[RiskAlert]:
        """Generate risk alerts"""
        
        self.alerts = []
        
        # Position concentration risk
        for symbol, position in current_positions.items():
            position_weight = (position.get('quantity', 0) * position.get('current_price', 0)) / portfolio_value
            
            if position_weight > 0.25:
                self.alerts.append(RiskAlert(
                    alert_type='CONCENTRATION_RISK',
                    level=AlertLevel.HIGH,
                    message=f'Position {symbol} exceeds 25% concentration limit',
                    symbol=symbol,
                    value=position_weight
                ))
        
        return self.alerts