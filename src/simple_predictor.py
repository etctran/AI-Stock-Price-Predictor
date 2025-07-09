"""
Super Simple Stock Predictor - No External Dependencies
Works with just Python standard library + basic packages
"""

import sys

# Check what's available
packages_needed = ['pandas', 'yfinance', 'streamlit']
missing_packages = []

for package in packages_needed:
    try:
        __import__(package)
        print(f"✅ {package} is available")
    except ImportError:
        print(f"❌ {package} is missing")
        missing_packages.append(package)

if missing_packages:
    print(f"\n🔧 Please install these packages:")
    for package in missing_packages:
        print(f"   pip install {package}")
    print(f"\nThen run this script again.")
    sys.exit(1)

# If we get here, all packages are available
print(f"\n🚀 All packages available! Starting simple predictor...")

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def simple_stock_analysis(symbol):
    """Simple stock analysis without matplotlib"""
    try:
        print(f"\n📊 Analyzing {symbol}...")
        
        # Get stock data
        stock = yf.Ticker(symbol)
        data = stock.history(period="3mo")
        
        if data.empty:
            print(f"❌ No data found for {symbol}")
            return
        
        # Basic analysis
        current_price = data['Close'].iloc[-1]
        prev_price = data['Close'].iloc[-2]
        change = current_price - prev_price
        change_pct = (change / prev_price) * 100
        
        # Simple moving averages
        ma_5 = data['Close'].tail(5).mean()
        ma_20 = data['Close'].tail(20).mean()
        
        # Simple prediction (basic trend)
        recent_trend = ma_5 - ma_20
        prediction = current_price + (recent_trend * 0.1)
        pred_change = ((prediction - current_price) / current_price) * 100
        
        # Display results
        print(f"\n📈 {symbol} Analysis Results:")
        print(f"   Current Price: ${current_price:.2f}")
        print(f"   Daily Change: ${change:.2f} ({change_pct:+.2f}%)")
        print(f"   5-day Average: ${ma_5:.2f}")
        print(f"   20-day Average: ${ma_20:.2f}")
        print(f"\n🎯 Simple Prediction:")
        print(f"   Predicted Price: ${prediction:.2f}")
        print(f"   Expected Change: {pred_change:+.2f}%")
        
        # Simple recommendation
        if pred_change > 2:
            rec = "🟢 POSITIVE - Price may rise"
        elif pred_change < -2:
            rec = "🔴 NEGATIVE - Price may fall"
        else:
            rec = "🟡 NEUTRAL - Price likely stable"
        
        print(f"   Recommendation: {rec}")
        
    except Exception as e:
        print(f"❌ Error analyzing {symbol}: {str(e)}")

def main():
    """Main function"""
    print("🤖 Simple AI Stock Predictor")
    print("=" * 40)
    
    # Popular stocks to try
    popular_stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    print(f"📋 Popular stocks: {', '.join(popular_stocks)}")
    
    while True:
        symbol = input(f"\n🎯 Enter stock symbol (or 'quit'): ").upper().strip()
        
        if symbol == 'QUIT':
            print("👋 Thank you!")
            break
        
        if not symbol:
            print("❌ Please enter a symbol")
            continue
        
        simple_stock_analysis(symbol)

if __name__ == "__main__":
    main()
