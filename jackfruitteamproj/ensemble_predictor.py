"""
models/ensemble_predictor.py
Ensemble ML models for Python 3.14
"""
from dataclasses import dataclass
from typing import Any, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
import torch
import torch.nn as nn
import torch.optim as optim
@dataclass
class EnsembleModel:
    """Container for ensemble model results"""
    predictions: dict[str, float]
    ensemble_prediction: float
    model_weights: dict[str, float]
    confidence: float
class EnsembleTradingPredictor:
    """Python 3.14 optimized ensemble predictor"""
    
    def __init__(self, prediction_horizon: int = 5) -> None:
        self.prediction_horizon: int = prediction_horizon
        self.models: dict[str, Any] = {}
        self.scalers: dict[str, RobustScaler] = {}
        self.feature_selectors: dict[str, SelectKBest] = {}
        self.model_weights: dict[str, float] = {}
    
    def prepare_features(
        self,
        price_data: pd.DataFrame,
        sentiment_data: Optional[dict[str, dict]] = None,
        alternative_data: Optional[dict[str, dict]] = None
    ) -> pd.DataFrame:
        """Prepare features using Python 3.14 native types"""
        
        features = pd.DataFrame(index=price_data.index)
        
        # Technical indicators
        features['returns'] = price_data['Close'].pct_change()
        features['returns_lag1'] = features['returns'].shift(1)
        features['returns_lag2'] = features['returns'].shift(2)
        features['returns_lag3'] = features['returns'].shift(3)
        
        # Moving averages
        features['sma_5'] = price_data['Close'].rolling(5).mean()
        features['sma_10'] = price_data['Close'].rolling(10).mean()
        features['sma_20'] = price_data['Close'].rolling(20).mean()
        features['sma_50'] = price_data['Close'].rolling(50).mean()
        
        # Moving average crossovers
        features['sma_5_10_cross'] = (features['sma_5'] > features['sma_10']).astype(int)
        features['sma_10_20_cross'] = (features['sma_10'] > features['sma_20']).astype(int)
        
        # Volatility
        features['volatility_5'] = features['returns'].rolling(5).std()
        features['volatility_20'] = features['returns'].rolling(20).std()
        
        # RSI
        features['rsi'] = self._calculate_rsi(price_data['Close'])
        
        # MACD
        features['macd'], features['macd_signal'] = self._calculate_macd(price_data['Close'])
        
        # Bollinger Bands
        features['bb_upper'], features['bb_lower'] = self._calculate_bollinger_bands(price_data['Close'])
        
        # Volume
        if 'Volume' in price_data.columns:
            features['volume_sma'] = price_data['Volume'].rolling(20).mean()
            features['volume_ratio'] = price_data['Volume'] / features['volume_sma']
        
        # Sentiment features
        if sentiment_data:
            for symbol, sent_data in sentiment_data.items():
                features[f'{symbol}_sentiment'] = sent_data.get('current_sentiment', 0)
        
        # Target variable
        features['target'] = features['returns'].shift(-self.prediction_horizon)
        
        return features.dropna()
    
    def build_ensemble_models(self) -> None:
        """Build ensemble of models"""
        
        self.models['random_forest'] = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42
        )
        
        self.models['gradient_boosting'] = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        self.models['svm'] = SVR(kernel='rbf', C=1.0, gamma='scale')
        
        self.models['neural_network'] = MLPRegressor(
            hidden_layer_sizes=(100, 50),
            learning_rate_init=0.001,
            max_iter=500,
            random_state=42
        )
        
        self.models['ridge'] = Ridge(alpha=1.0)
        self.models['lasso'] = Lasso(alpha=0.1)
    
    def build_lstm_model(self, input_shape: tuple[int, ...]) -> Any:
        """Build LSTM model for time series"""
        model = models.Sequential([
            layers.LSTM(100, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.2),
            layers.BatchNormalization(),
            layers.LSTM(50, return_sequences=False),
            layers.Dropout(0.2),
            layers.BatchNormalization(),
            layers.Dense(25, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(1)
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_ensemble(
        self,
        features_df: pd.DataFrame,
        validation_split: float = 0.2
    ) -> dict[str, float]:
        """Train ensemble models"""
        
        X = features_df.drop(['target'], axis=1)
        y = features_df['target']
        
        # Feature selection
        selector = SelectKBest(score_func=f_regression, k=min(50, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        self.feature_selectors['main'] = selector
        
        # Scaling
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_selected)
        self.scalers['main'] = scaler
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=5)
        
        self.build_ensemble_models()
        
        model_scores: dict[str, float] = {}
        
        for name, model in self.models.items():
            try:
                scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='neg_mean_squared_error')
                model_scores[name] = float(-np.mean(scores))
                model.fit(X_scaled, y)
            except Exception as e:
                print(f"Error training {name}: {e}")
                model_scores[name] = float('inf')
        
        self._calculate_model_weights(model_scores)
        
        return model_scores
    
    def _calculate_model_weights(self, model_scores: dict[str, float]) -> None:
        """Calculate ensemble weights"""
        
        valid_scores = {k: v for k, v in model_scores.items() if v != float('inf')}
        
        if not valid_scores:
            self.model_weights = {k: 1/len(self.models) for k in self.models.keys()}
            return
        
        inverse_scores = {k: 1/v for k, v in valid_scores.items()}
        total_inverse = sum(inverse_scores.values())
        
        self.model_weights = {k: v/total_inverse for k, v in inverse_scores.items()}
    
    def predict_ensemble(self, features_df: pd.DataFrame) -> EnsembleModel:
        """Make ensemble predictions"""
        
        X = features_df.drop(['target'], axis=1, errors='ignore')
        
        X_selected = self.feature_selectors['main'].transform(X)
        X_scaled = self.scalers['main'].transform(X_selected)
        
        predictions: dict[str, float] = {}
        
        for name, model in self.models.items():
            try:
                pred = model.predict(X_scaled[-1:])
                predictions[name] = float(pred[0])
            except Exception as e:
                print(f"Error predicting with {name}: {e}")
                predictions[name] = 0.0
        
        ensemble_pred = sum(
            predictions[name] * self.model_weights.get(name, 0)
            for name in predictions.keys()
        )
        
        confidence = float(np.mean([abs(predictions[name]) for name in predictions.keys()]))
        
        return EnsembleModel(
            predictions=predictions,
            ensemble_prediction=ensemble_pred,
            model_weights=self.model_weights,
            confidence=confidence
        )
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
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
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
    
    @staticmethod
    def _calculate_bollinger_bands(
        prices: pd.Series,
        window: int = 20,
        num_std: int = 2
    ) -> tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band, lower_band
class DeepLearningPredictor:
    """LSTM/Transformer for price prediction"""
    
    def __init__(self, sequence_length: int = 60):
        self.sequence_length = sequence_length
        self.model = self._build_model()
    
    def _build_model(self):
        """Build LSTM model"""
        model = keras.Sequential([
            layers.LSTM(128, return_sequences=True, input_shape=(self.sequence_length, 5)),
            layers.Dropout(0.2),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, X_train, y_train, epochs=50):
        """Train the model"""
        return self.model.fit(
            X_train, y_train,
            epochs=epochs,
            validation_split=0.2,
            batch_size=32,
            verbose=1
        )
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
class TradingRLAgent(nn.Module):
    """Reinforcement learning trading agent"""
    
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
    
    def forward(self, state):
        """Forward pass"""
        return self.network(state)
    
    def train_step(self, state, action, reward, next_state):
        """Training step for Q-learning"""
        # Q-learning update
        current_q = self.forward(state)
        next_q = self.forward(next_state)
        
        target_q = current_q.clone()
        target_q[action] = reward + 0.99 * torch.max(next_q)
        
        loss = nn.MSELoss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()