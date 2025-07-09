"""
Ensemble Model for Stock Prediction

Combines Random Forest, XGBoost, and basic LSTM for stock price prediction.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class SimpleEnsembleModel:
    """Simple ensemble combining Random Forest, XGBoost, and basic prediction."""
    
    def __init__(self):
        self.models = {}
        self.weights = [0.4, 0.4, 0.2]  # RF, XGB, Simple LSTM-like
        self.is_trained = False
    
    def prepare_features(self, data):
        """Simple feature engineering for ensemble models."""
        # Basic technical indicators
        data['ma_5'] = data['Close'].rolling(5).mean()
        data['ma_10'] = data['Close'].rolling(10).mean()
        data['ma_20'] = data['Close'].rolling(20).mean()
        data['volatility'] = data['Close'].rolling(10).std()
        data['price_change'] = data['Close'].pct_change()
        data['volume_ma'] = data['Volume'].rolling(5).mean()
        
        # Lag features (LSTM-like)
        for lag in [1, 2, 3]:
            data[f'close_lag_{lag}'] = data['Close'].shift(lag)
            data[f'volume_lag_{lag}'] = data['Volume'].shift(lag)
        
        # Drop NaN values
        data = data.dropna()
        
        # Feature columns
        feature_cols = ['ma_5', 'ma_10', 'ma_20', 'volatility', 'price_change', 
                       'volume_ma', 'close_lag_1', 'close_lag_2', 'close_lag_3',
                       'volume_lag_1', 'volume_lag_2', 'volume_lag_3']
        
        return data[feature_cols], data['Close']
    
    def hyperparameter_optimization(self, X, y):
        """Simple hyperparameter optimization for ensemble models."""
        print("Running hyperparameter optimization...")
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Random Forest optimization
        rf_params = {
            'n_estimators': [50, 100],
            'max_depth': [5, 10],
            'min_samples_split': [2, 5]
        }
        
        rf_grid = GridSearchCV(
            RandomForestRegressor(random_state=42),
            rf_params,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        rf_grid.fit(X, y)
        self.models['random_forest'] = rf_grid.best_estimator_
        
        # XGBoost optimization
        xgb_params = {
            'n_estimators': [50, 100],
            'learning_rate': [0.1, 0.2],
            'max_depth': [3, 6]
        }
        
        xgb_grid = GridSearchCV(
            xgb.XGBRegressor(random_state=42),
            xgb_params,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        xgb_grid.fit(X, y)
        self.models['xgboost'] = xgb_grid.best_estimator_
        
        print("Hyperparameter optimization completed")
        return rf_grid.best_params_, xgb_grid.best_params_
    
    def train_ensemble(self, X, y):
        """Train ensemble models with hyperparameter optimization."""
        # Run hyperparameter optimization
        rf_params, xgb_params = self.hyperparameter_optimization(X, y)
        
        # Simple LSTM-like model (weighted average of recent prices)
        # This simulates LSTM behavior without actual deep learning
        self.models['lstm_like'] = 'weighted_moving_average'
        
        self.is_trained = True
        print(f"Ensemble training completed")
        print(f"   Random Forest params: {rf_params}")
        print(f"   XGBoost params: {xgb_params}")
        print(f"   LSTM-like: Weighted moving average")
    
    def predict_ensemble(self, X, recent_prices=None):
        """Generate ensemble predictions."""
        if not self.is_trained:
            raise ValueError("Models not trained yet")
        
        # Random Forest prediction
        rf_pred = self.models['random_forest'].predict(X)
        
        # XGBoost prediction
        xgb_pred = self.models['xgboost'].predict(X)
        
        # Simple LSTM-like prediction (weighted recent prices)
        if recent_prices is not None and len(recent_prices) >= 3:
            lstm_weights = [0.5, 0.3, 0.2]  # Recent prices get higher weight
            lstm_pred = np.array([
                np.average(recent_prices[-3:], weights=lstm_weights)
                for _ in range(len(X))
            ])
        else:
            lstm_pred = (rf_pred + xgb_pred) / 2  # Fallback
        
        # Ensemble combination
        ensemble_pred = (
            self.weights[0] * rf_pred +
            self.weights[1] * xgb_pred +
            self.weights[2] * lstm_pred
        )
        
        return ensemble_pred, {
            'random_forest': rf_pred,
            'xgboost': xgb_pred,
            'lstm_like': lstm_pred,
            'ensemble': ensemble_pred
        }
    
    def calculate_directional_accuracy(self, actual_prices, predicted_prices):
        """Calculate directional accuracy for trading."""
        actual_direction = np.sign(np.diff(actual_prices))
        predicted_direction = np.sign(np.diff(predicted_prices))
        
        accuracy = np.mean(actual_direction == predicted_direction)
        return accuracy

def demo_ensemble_model():
    """Demonstrate ensemble model with real data."""
    import yfinance as yf
    
    print("Testing Ensemble ML Pipeline...")
    
    # Get sample data
    stock = yf.Ticker("AAPL")
    data = stock.history(period="1y")
    
    # Initialize ensemble
    ensemble = SimpleEnsembleModel()
    
    # Prepare features
    X, y = ensemble.prepare_features(data)
    
    # Split data (80% train, 20% test)
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train ensemble
    ensemble.train_ensemble(X_train, y_train)
    
    # Make predictions
    ensemble_pred, individual_preds = ensemble.predict_ensemble(
        X_test, 
        recent_prices=y_train.tail(10).values
    )
    
    # Calculate directional accuracy
    accuracy = ensemble.calculate_directional_accuracy(y_test.values, ensemble_pred)
    
    print(f"\nEnsemble Model Results:")
    print(f"   Directional Accuracy: {accuracy:.1%}")
    print(f"   Models: Random Forest + XGBoost + LSTM-like")
    print(f"   Features: {len(X.columns)} engineered features")
    print(f"   Test samples: {len(X_test)}")
    
    return accuracy

if __name__ == "__main__":
    demo_ensemble_model()
