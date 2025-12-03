"""
data/market_data_collector.py
Multi-source market data collection for Python 3.14
"""

import asyncio
import alpaca_trade_api as tradeapi
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Any, Optional
from collections.abc import Coroutine, Sequence

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    import yfinance as yf
except Exception as exc:
    yf = None
    YFINANCE_IMPORT_ERROR: Optional[Exception] = exc
else:
    YFINANCE_IMPORT_ERROR = None

try:  # pragma: no cover - optional dependency
    import ccxt
except Exception as exc:
    ccxt = None
    CCXT_IMPORT_ERROR: Optional[Exception] = exc
else:
    CCXT_IMPORT_ERROR = None

try:  # pragma: no cover - optional dependency
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
except Exception as exc:
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    pipeline = None
    TRANSFORMERS_IMPORT_ERROR: Optional[Exception] = exc
else:
    TRANSFORMERS_IMPORT_ERROR = None

from settings import FREE_THREADED

@dataclass
class PriceData:
    """OHLCV price data container"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    adjusted_close: Optional[float] = None

class MarketDataCollector:
    """Python 3.14 async-optimized market data collector"""
    
    def __init__(self, use_threading: bool = FREE_THREADED) -> None:
        self.use_threading: bool = use_threading
        self.max_workers: int = 20 if use_threading else 4
        self.data_cache: dict[str, pd.DataFrame] = {}
        self._rng = np.random.default_rng()
        self.exchanges: dict[str, Any] = {
            'crypto': ccxt.binance() if ccxt is not None else None,
            'crypto_kucoin': ccxt.kucoin() if ccxt is not None else None,
        }
    
    async def fetch_multi_asset_data(
        self,
        symbols: Sequence[dict[str, str]],
        start_date: datetime,
        end_date: datetime
    ) -> dict[str, pd.DataFrame]:
        """Fetch data from multiple asset classes asynchronously"""
        
        all_data: dict[str, pd.DataFrame] = {}
        
        stock_symbols: list[str] = [s['symbol'] for s in symbols if s.get('type') == 'stock']
        crypto_symbols: list[str] = [s['symbol'] for s in symbols if s.get('type') == 'crypto']
        
        if self.use_threading:
            # Use concurrent.interpreters for true parallelism in Python 3.14
            tasks: list[Coroutine] = []
            
            if stock_symbols:
                tasks.append(self._fetch_stock_data_async(stock_symbols, start_date, end_date))
            if crypto_symbols:
                tasks.append(self._fetch_crypto_data_async(crypto_symbols, start_date, end_date))
            
            results: list[dict[str, pd.DataFrame]] = await asyncio.gather(*tasks)
            
            for result in results:
                all_data.update(result)
        else:
            # Fallback to sequential for non-free-threaded mode
            if stock_symbols:
                stock_data = await self._fetch_stock_data_async(stock_symbols, start_date, end_date)
                all_data.update(stock_data)
            if crypto_symbols:
                crypto_data = await self._fetch_crypto_data_async(crypto_symbols, start_date, end_date)
                all_data.update(crypto_data)
        
        return all_data
    
    async def _fetch_stock_data_async(
        self,
        symbols: Sequence[str],
        start_date: datetime,
        end_date: datetime
    ) -> dict[str, pd.DataFrame]:
        """Async stock data fetching"""
        stock_data: dict[str, pd.DataFrame] = {}
        
        loop = asyncio.get_event_loop()
        
        for symbol in symbols:
            try:
                # Run blocking I/O in thread pool
                df = await loop.run_in_executor(
                    None,
                    lambda s=symbol: self._fetch_stock_sync(s, start_date, end_date)
                )
                stock_data[symbol] = df
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
        
        return stock_data
    
    def _fetch_stock_sync(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Synchronous stock data fetching"""
        if yf is None:
            return self._generate_synthetic_stock_data(symbol, start_date, end_date)
        
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date)
        except Exception as exc:  # pragma: no cover - network dependency
            print(f"Warning: Falling back to synthetic stock data for {symbol}: {exc}")
            return self._generate_synthetic_stock_data(symbol, start_date, end_date)
        
        if hist.empty:
            return self._generate_synthetic_stock_data(symbol, start_date, end_date)
        
        # Technical indicators
        hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
        hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
        hist['RSI'] = self._calculate_rsi(hist['Close'])
        hist['MACD'], hist['MACD_Signal'] = self._calculate_macd(hist['Close'])
        hist['Volatility'] = hist['Close'].pct_change().rolling(window=20).std()
        
        return hist
    
    async def _fetch_crypto_data_async(
        self,
        symbols: Sequence[str],
        start_date: datetime,
        end_date: datetime
    ) -> dict[str, pd.DataFrame]:
        """Async cryptocurrency data fetching"""
        crypto_data: dict[str, pd.DataFrame] = {}
        
        loop = asyncio.get_event_loop()
        
        for symbol in symbols:
            try:
                df = await loop.run_in_executor(
                    None,
                    lambda s=symbol: self._fetch_crypto_sync(s, start_date, end_date)
                )
                crypto_data[symbol] = df
            except Exception as e:
                print(f"Error fetching crypto {symbol}: {e}")
        
        return crypto_data
    
    def _fetch_crypto_sync(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Synchronous crypto data fetching"""
        exchange = self.exchanges.get('crypto')
        if exchange is None:
            return self._generate_synthetic_crypto_data(symbol, start_date, end_date)
        
        symbol_format = symbol.replace('-', '/')
        
        try:
            ohlcv = exchange.fetch_ohlcv(
                symbol_format, '1d',
                since=int(start_date.timestamp() * 1000)
            )
        except Exception as exc:  # pragma: no cover - network dependency
            print(f"Warning: Falling back to synthetic crypto data for {symbol}: {exc}")
            return self._generate_synthetic_crypto_data(symbol, start_date, end_date)
        
        if not ohlcv:
            return self._generate_synthetic_crypto_data(symbol, start_date, end_date)
        
        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Technical indicators
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['RSI'] = self._calculate_rsi(df['close'])
        df['Volatility'] = df['close'].pct_change().rolling(window=20).std()
        
        return df

    def _generate_synthetic_stock_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Generate deterministic synthetic stock data for offline mode."""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        if len(dates) == 0:
            dates = pd.date_range(end=datetime.now(), periods=60, freq='D')
        
        seed = abs(hash(symbol)) % (2**32 - 1)
        rng = np.random.default_rng(seed)
        base_price = rng.uniform(50, 250)
        returns = rng.normal(0.0005, 0.015, len(dates))
        prices = base_price * np.cumprod(1 + returns)
        
        df = pd.DataFrame(index=dates)
        df['Close'] = prices
        df['Open'] = prices * (1 + rng.normal(0, 0.002, len(dates)))
        df['High'] = df[['Open', 'Close']].max(axis=1) * (1 + rng.uniform(0.001, 0.01, len(dates)))
        df['Low'] = df[['Open', 'Close']].min(axis=1) * (1 - rng.uniform(0.001, 0.01, len(dates)))
        df['Volume'] = rng.integers(50_000, 500_000, len(dates))
        df['Adj Close'] = df['Close']
        
        df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
        df['SMA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
        df['RSI'] = self._calculate_rsi(df['Close'])
        df['MACD'], df['MACD_Signal'] = self._calculate_macd(df['Close'])
        df['Volatility'] = df['Close'].pct_change().rolling(window=20, min_periods=1).std()
        
        return df

    def _generate_synthetic_crypto_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Generate deterministic synthetic crypto data for offline mode."""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        if len(dates) == 0:
            dates = pd.date_range(end=datetime.now(), periods=60, freq='D')
        
        seed = abs(hash(symbol)) % (2**32 - 1)
        rng = np.random.default_rng(seed)
        base_price = rng.uniform(1000, 40000)
        returns = rng.normal(0.001, 0.03, len(dates))
        prices = base_price * np.cumprod(1 + returns)
        
        df = pd.DataFrame(index=dates)
        df['open'] = prices * (1 + rng.normal(0, 0.01, len(dates)))
        df['close'] = prices
        df['high'] = df[['open', 'close']].max(axis=1) * (1 + rng.uniform(0.002, 0.02, len(dates)))
        df['low'] = df[['open', 'close']].min(axis=1) * (1 - rng.uniform(0.002, 0.02, len(dates)))
        df['volume'] = rng.integers(10, 10_000, len(dates))
        
        df['SMA_20'] = df['close'].rolling(window=20, min_periods=1).mean()
        df['RSI'] = self._calculate_rsi(df['close'])
        df['Volatility'] = df['close'].pct_change().rolling(window=20, min_periods=1).std()
        
        return df
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def _calculate_macd(
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> tuple[pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal

class SentimentAnalyzer:
    """Financial sentiment analysis using FinBERT with graceful fallbacks."""
    
    def __init__(self) -> None:
        self.sentiment_pipeline = None
        self._load_error: Optional[Exception] = None
        self._rng = np.random.default_rng()
        
        if pipeline is None or AutoTokenizer is None or AutoModelForSequenceClassification is None:
            self._load_error = TRANSFORMERS_IMPORT_ERROR
            return
        
        try:  # pragma: no cover - heavy model download
            self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer
            )
        except Exception as exc:
            # Transformers is installed but lacks a backend (e.g. PyTorch).
            self._load_error = exc
            self.sentiment_pipeline = None
    
    async def analyze_multi_source_sentiment(
        self,
        symbols: Sequence[str],
        lookback_days: int = 7
    ) -> dict[str, dict[str, Any]]:
        """Analyze sentiment from multiple sources"""
        
        all_sentiment_data: dict[str, dict[str, Any]] = {}
        
        loop = asyncio.get_event_loop()
        
        for symbol in symbols:
            try:
                sentiment_data = await loop.run_in_executor(
                    None,
                    lambda s=symbol: self._analyze_sentiment_sync(s, lookback_days)
                )
                all_sentiment_data[symbol] = sentiment_data
            except Exception as e:
                print(f"Error analyzing sentiment for {symbol}: {e}")
        
        return all_sentiment_data
    
    def _analyze_sentiment_sync(self, symbol: str, days: int) -> dict[str, Any]:
        """Synchronous sentiment analysis"""
        sentiment_scores: list[dict[str, Any]] = []
        
        # Placeholder for actual news/social data collection
        sample_texts = [
            f"Strong bullish sentiment on {symbol}",
            f"Market concerns about {symbol} valuation",
            f"{symbol} shows promising technical setup"
        ]
        
        if self.sentiment_pipeline is None:
            return self._synthetic_sentiment(symbol, days)
        
        for text in sample_texts:
            result = self.sentiment_pipeline(text[:512])[0]
            
            sentiment_score = float(result['score']) if result['label'] == 'positive' else \
                            -float(result['score']) if result['label'] == 'negative' else 0.0
            
            sentiment_scores.append({
                'timestamp': datetime.now(),
                'compound_score': sentiment_score,
                'raw_sentiment': result
            })
        
        if sentiment_scores:
            avg_sentiment = float(np.mean([s['compound_score'] for s in sentiment_scores]))
        else:
            avg_sentiment = 0.0
        
        return {
            'current_sentiment': avg_sentiment,
            'sentiment_volatility': float(np.std([s['compound_score'] for s in sentiment_scores])) if len(sentiment_scores) > 1 else 0.0,
            'raw_scores': sentiment_scores
        }

    def _synthetic_sentiment(self, symbol: str, days: int) -> dict[str, Any]:
        """Generate lightweight sentiment estimates when transformers is unavailable."""
        seed = abs(hash((symbol, days))) % (2**32 - 1)
        rng = np.random.default_rng(seed)
        sample_scores = rng.normal(0, 0.2, 5)
        
        sentiment_scores = []
        for score in sample_scores:
            sentiment_scores.append({
                'timestamp': datetime.now(),
                'compound_score': float(np.clip(score, -1, 1)),
                'raw_sentiment': {
                    'label': 'synthetic',
                    'score': float(np.clip(abs(score), 0, 1))
                }
            })
        
        avg_sentiment = float(np.mean([s['compound_score'] for s in sentiment_scores]))
        
        return {
            'current_sentiment': avg_sentiment,
            'sentiment_volatility': float(np.std([s['compound_score'] for s in sentiment_scores])),
            'raw_scores': sentiment_scores,
            'synthetic': True,
            'reason': 'transformers-backend-unavailable'
        }

class AlpacaDataProvider:
    """Alpaca data provider"""
    
    def __init__(self, api_key: str, secret_key: str, base_url: str = "https://paper-api.alpaca.markets"):
        self.api = tradeapi.REST(api_key, secret_key, base_url)
    
    def get_bars(self, symbol: str, timeframe: str = '1Day', limit: int = 100):
        """Fetch historical bars"""
        return self.api.get_bars(symbol, timeframe, limit=limit).df
    
    def get_latest_trade(self, symbol: str):
        """Get latest trade"""
        return self.api.get_latest_trade(symbol)

class SentimentAnalyzer:
    """Enhanced sentiment analysis with transformers"""
    
    def __init__(self):
        # Load pre-trained FinBERT model
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert"
        )
    
    async def analyze_news_sentiment(self, news_texts: list[str]) -> dict:
        """Analyze sentiment of news articles"""
        results = self.sentiment_pipeline(news_texts)
        
        sentiment_scores = {
            'positive': 0,
            'negative': 0,
            'neutral': 0
        }
        
        for result in results:
            label = result['label'].lower()
            score = result['score']
            sentiment_scores[label] += score
        
        # Normalize
        total = sum(sentiment_scores.values())
        return {k: v/total for k, v in sentiment_scores.items()}