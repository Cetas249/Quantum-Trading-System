"""
main.py
Main application entry point for Python 3.14
"""

import asyncio
from datetime import datetime, timedelta

from settings import FREE_THREADED, get_config
from optimization_algorithms import QuantumOptimizer
from market_data_collector import MarketDataCollector, SentimentAnalyzer
from ensemble_predictor import EnsembleTradingPredictor
from risk_manager import AdvancedRiskManager
from execution_system import ExecutionEngine

async def main() -> None:
    """Main async application"""
    
    print("=" * 80)
    print("Quantum-Inspired Financial Trading System - Python 3.14")
    print(f"Free-Threaded Mode: {FREE_THREADED}")
    print("=" * 80)
    
    # Get configuration
    config = get_config()
    
    # Initialize quantum optimizer
    quantum_optimizer = QuantumOptimizer(num_qubits=config['quantum'].num_qubits)
    
    # Initialize market data collector
    market_collector = MarketDataCollector(use_threading=FREE_THREADED)
    
    # Initialize sentiment analyzer
    sentiment_analyzer = SentimentAnalyzer()
    
    # Initialize ensemble predictor
    ensemble = EnsembleTradingPredictor(prediction_horizon=5)
    
    # Initialize risk manager
    risk_manager = AdvancedRiskManager()
    
    # Initialize execution engine
    execution_engine = ExecutionEngine(use_threading=FREE_THREADED)
    execution_engine.start()
    
    # Fetch market data
    symbols = [
        {'symbol': 'AAPL', 'type': 'stock'},
        {'symbol': 'BTC-USD', 'type': 'crypto'}
    ]
    
    lookback_days = 180
    start_date = datetime.now() - timedelta(days=lookback_days)
    end_date = datetime.now()
    
    print("\\nFetching market data...")
    market_data = await market_collector.fetch_multi_asset_data(symbols, start_date, end_date)
    
    print("Analyzing sentiment...")
    sentiment_data = await sentiment_analyzer.analyze_multi_source_sentiment(
        [s['symbol'] for s in symbols]
    )
    
    print("Building ML models...")
    if 'AAPL' in market_data:
        features = ensemble.prepare_features(
            market_data['AAPL'],
            sentiment_data=sentiment_data
        )
        if features.empty:
            print("Insufficient historical data to train ensemble models.")
        else:
            try:
                model_scores = ensemble.train_ensemble(features)
                print(f"Model scores: {model_scores}")
            except Exception as exc:
                print(f"Error training ensemble models: {exc}")
    
    # Run trading logic
    print("\\nStarting trading...")
    await execution_engine.execute_trade('AAPL', 'BUY', 10, 'market')
    
    # Cleanup
    execution_engine.stop()
    print("\\nTrading system terminated.")

if __name__ == '__main__':
    asyncio.run(main())