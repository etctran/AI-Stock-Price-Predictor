"""
Quick Accuracy Test for Resume Verification

Let's see what accuracy the actual code achieves to verify resume claims.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

def quick_accuracy_test():
    """Test actual accuracy of the prediction system."""
    
    print("🧪 Testing Actual Prediction Accuracy...")
    print("=" * 50)
    
    # Test multiple stocks for more reliable results
    test_symbols = ['AAPL', 'GOOGL', 'MSFT']
    all_accuracies = []
    
    for symbol in test_symbols:
        try:
            print(f"\n📊 Testing {symbol}...")
            
            # Get data
            stock = yf.Ticker(symbol)
            data = stock.history(period="1y")
            
            if len(data) < 100:
                print(f"❌ Insufficient data for {symbol}")
                continue
            
            # Simple feature engineering (like your code)
            data['ma_5'] = data['Close'].rolling(5).mean()
            data['ma_20'] = data['Close'].rolling(20).mean()
            data['volatility'] = data['Close'].rolling(10).std()
            data['price_change'] = data['Close'].pct_change()
            data['close_lag_1'] = data['Close'].shift(1)
            data['close_lag_2'] = data['Close'].shift(2)
            
            # Drop NaN
            data = data.dropna()
            
            if len(data) < 50:
                print(f"❌ Not enough clean data for {symbol}")
                continue
            
            # Prepare features and target
            feature_cols = ['ma_5', 'ma_20', 'volatility', 'price_change', 'close_lag_1', 'close_lag_2']
            X = data[feature_cols]
            y = data['Close']
            
            # Split data (80% train, 20% test)
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Simple ensemble (Random Forest + XGBoost)
            rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
            xgb_model = xgb.XGBRegressor(n_estimators=50, random_state=42, eval_metric='rmse')
            
            # Train models
            rf_model.fit(X_train, y_train)
            xgb_model.fit(X_train, y_train)
            
            # Make predictions
            rf_pred = rf_model.predict(X_test)
            xgb_pred = xgb_model.predict(X_test)
            
            # Simple ensemble (average)
            ensemble_pred = (rf_pred + xgb_pred) / 2
            
            # Calculate directional accuracy
            actual_direction = np.sign(np.diff(y_test.values))
            predicted_direction = np.sign(np.diff(ensemble_pred))
            
            accuracy = np.mean(actual_direction == predicted_direction)
            all_accuracies.append(accuracy)
            
            print(f"   Directional Accuracy: {accuracy:.1%}")
            print(f"   Test samples: {len(X_test)}")
            
        except Exception as e:
            print(f"❌ Error testing {symbol}: {str(e)}")
            continue
    
    # Overall results
    if all_accuracies:
        avg_accuracy = np.mean(all_accuracies)
        min_accuracy = np.min(all_accuracies)
        max_accuracy = np.max(all_accuracies)
        
        print(f"\n🎯 OVERALL RESULTS:")
        print(f"   Average Accuracy: {avg_accuracy:.1%}")
        print(f"   Range: {min_accuracy:.1%} - {max_accuracy:.1%}")
        print(f"   Stocks tested: {len(all_accuracies)}")
        
        # Resume claim verification
        print(f"\n✅ RESUME VERIFICATION:")
        if avg_accuracy >= 0.75:
            print(f"   ✅ 75% claim is SUPPORTED ({avg_accuracy:.1%})")
        elif avg_accuracy >= 0.70:
            print(f"   ⚠️  Close to 75% ({avg_accuracy:.1%}) - might need to say 70%")
        else:
            print(f"   ❌ 75% claim NOT supported ({avg_accuracy:.1%}) - use {avg_accuracy:.0%}")
        
        return avg_accuracy
    else:
        print("❌ No successful tests completed")
        return None

def test_simple_baseline():
    """Test simple baseline for comparison."""
    
    print(f"\n📊 Testing Simple Baseline (Moving Average Strategy)...")
    
    try:
        # Test with AAPL
        stock = yf.Ticker("AAPL")
        data = stock.history(period="6mo")
        
        # Simple strategy: buy when price > 20-day MA
        data['ma_20'] = data['Close'].rolling(20).mean()
        data = data.dropna()
        
        # Generate signals
        data['signal'] = np.where(data['Close'] > data['ma_20'], 1, -1)
        
        # Calculate directional accuracy
        actual_direction = np.sign(data['Close'].diff())
        predicted_direction = data['signal']
        
        # Remove NaN
        mask = ~np.isnan(actual_direction)
        accuracy = np.mean(actual_direction[mask] == predicted_direction[mask])
        
        print(f"   Simple MA Strategy Accuracy: {accuracy:.1%}")
        print(f"   (This is your baseline to beat)")
        
        return accuracy
        
    except Exception as e:
        print(f"❌ Baseline test failed: {str(e)}")
        return None

if __name__ == "__main__":
    # Test baseline first
    baseline = test_simple_baseline()
    
    # Test your actual model
    actual_accuracy = quick_accuracy_test()
    
    if actual_accuracy and baseline:
        improvement = actual_accuracy - baseline
        print(f"\n🚀 IMPROVEMENT OVER BASELINE: {improvement:+.1%}")
