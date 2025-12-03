"""
config/settings.py
Python 3.14 compatible configuration module
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any
from collections.abc import Mapping, Sequence

# Python 3.14: Check for free-threaded mode
FREE_THREADED = hasattr(sys, '_is_gil_disabled') and sys._is_gil_disabled()

@dataclass
class QuantumConfig:
    """Quantum computing configuration"""
    num_qubits: int = 6
    layers: int = 3
    optimization_method: str = 'COBYLA'
    max_iterations: int = 200
    device_backend: str = 'default.qubit'
    enable_vqa: bool = True
    vqa_layers: int = 2

@dataclass
class TradingConfig:
    """Trading system configuration"""
    initial_capital: float = 100000.0
    transaction_cost: float = 0.001
    max_position_size: float = 0.1
    max_daily_loss: float = 0.05
    risk_free_rate: float = 0.02
    prediction_horizon: int = 5
    paper_trading: bool = True

@dataclass
class DataConfig:
    """Data collection configuration"""
    market_data_update_interval: int = 1  # seconds
    sentiment_update_interval: int = 300  # seconds
    max_lookback_days: int = 365
    enable_cache: bool = True
    cache_expiry: int = 3600
    parallel_data_streams: int = 20 if FREE_THREADED else 4

@dataclass
class ExecutionConfig:
    """Execution engine configuration"""
    max_workers: int = 20 if FREE_THREADED else 4
    order_queue_size: int = 1000
    execution_timeout: int = 300
    max_retries: int = 3
    latency_monitoring: bool = True
    use_multiple_interpreters: bool = FREE_THREADED

def get_config() -> dict[str, Any]:
    """Get complete system configuration"""
    return {
        'quantum': QuantumConfig(),
        'trading': TradingConfig(),
        'data': DataConfig(),
        'execution': ExecutionConfig(),
        'free_threaded': FREE_THREADED,
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}"
    }
