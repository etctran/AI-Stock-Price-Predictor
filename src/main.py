"""
Main Module for AI Stock Price Predictor

Simple stock price analysis and prediction using machine learning.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


def get_stock_data(symbol, period="1y"):
    """
    Get stock data from Yahoo Finance.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL')
        period: Time period for data ('1y', '6mo', '3mo')
    
    Returns:
        DataFrame with stock price data
    """
    try:
        print(f"Collecting data for {symbol}...")
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        
        if data.empty:
            print(f"No data found for {symbol}")
            return None
        
        print(f"Successfully collected {len(data)} days of data")
        return data
        
    except Exception as e:
        print(f"Error collecting data: {str(e)}")
        return None


def calculate_moving_averages(data):
    """
    Calculate simple moving averages for stock analysis.
    
    Args:
        data: DataFrame with stock price data
    
    Returns:
        DataFrame with added moving average columns
        
    ATS Keywords: Technical Analysis, Moving Averages, Feature Engineering,
    Statistical Analysis, Financial Indicators
    """
    try:
        # Calculate different moving averages
        data['MA_5'] = data['Close'].rolling(window=5).mean()
        data['MA_10'] = data['Close'].rolling(window=10).mean()
        data['MA_20'] = data['Close'].rolling(window=20).mean()
        data['MA_50'] = data['Close'].rolling(window=50).mean()
        
        # Calculate price change
        data['Price_Change'] = data['Close'].pct_change()
        
        # Calculate volatility (rolling standard deviation)
        data['Volatility'] = data['Close'].rolling(window=20).std()
        
        print("Technical indicators calculated")
        return data
        
    except Exception as e:
        print(f"Error calculating indicators: {str(e)}")
        return data


def simple_predict_price(data, days_ahead=5):
    """
    Simple price prediction using moving averages and trends.
    
    Args:
        data: DataFrame with stock price data
        days_ahead: Number of days to predict
    
    Returns:
        List of predicted prices
        
    ATS Keywords: Price Prediction, Forecasting, Trend Analysis,
    Simple Machine Learning, Statistical Modeling
    """
    try:
        # Get recent data for prediction
        recent_data = data.tail(20)  # Last 20 days
        
        # Calculate simple trend
        current_price = data['Close'].iloc[-1]
        ma_5 = data['MA_5'].iloc[-1]
        ma_10 = data['MA_10'].iloc[-1]
        
        # Simple trend analysis
        if ma_5 > ma_10:
            trend = "upward"
            trend_factor = 1.001  # Slight upward trend
        else:
            trend = "downward"
            trend_factor = 0.999  # Slight downward trend
        
        # Calculate average daily change
        avg_change = recent_data['Price_Change'].mean()
        volatility = recent_data['Volatility'].iloc[-1]
        
        # Generate predictions
        predictions = []
        price = current_price
        
        print(f"🎯 Predicting {days_ahead} days ahead...")
        print(f"📈 Current trend: {trend}")
        
        for day in range(days_ahead):
            # Simple prediction: current price + trend + some randomness
            daily_change = avg_change + np.random.normal(0, volatility/100)
            price = price * (1 + daily_change) * trend_factor
            predictions.append(price)
        
        return predictions
        
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        return [current_price] * days_ahead


def create_simple_chart(data, predictions, symbol):
    """
    Create a simple chart showing historical prices and predictions.
    
    Args:
        data: Historical stock data
        predictions: List of predicted prices
        symbol: Stock symbol
        
    ATS Keywords: Data Visualization, Stock Charts, Matplotlib,
    Financial Visualization, Chart Creation
    """
    try:
        # Prepare data for plotting
        recent_data = data.tail(30)  # Last 30 days
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Plot historical prices
        plt.subplot(2, 1, 1)
        plt.plot(recent_data.index, recent_data['Close'], 'b-', linewidth=2, label='Historical Price')
        plt.plot(recent_data.index, recent_data['MA_10'], 'r--', alpha=0.7, label='10-day Moving Average')
        plt.title(f'{symbol} Stock Price Analysis', fontsize=16, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot predictions
        plt.subplot(2, 1, 2)
        
        # Future dates for predictions
        last_date = data.index[-1]
        future_dates = [last_date + timedelta(days=i+1) for i in range(len(predictions))]
        
        # Plot last few historical points and predictions
        last_5_days = recent_data.tail(5)
        plt.plot(last_5_days.index, last_5_days['Close'], 'b-', linewidth=2, label='Recent Prices')
        plt.plot(future_dates, predictions, 'r--o', linewidth=2, label='Predictions')
        
        plt.title(f'{symbol} Price Predictions', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Predicted Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        print("✅ Chart created successfully")
        
    except Exception as e:
        print(f"❌ Chart creation error: {str(e)}")


def analyze_stock(symbol):
    """
    Complete stock analysis function.
    
    Args:
        symbol: Stock ticker symbol
        
    ATS Keywords: Stock Analysis, Complete Analysis, Financial Analysis,
    Investment Analysis, Data Science Pipeline
    """
    print(f"\n🚀 Starting AI Stock Analysis for {symbol}")
    print("=" * 50)
    
    # Step 1: Get stock data
    data = get_stock_data(symbol)
    if data is None:
        return
    
    # Step 2: Calculate technical indicators
    data = calculate_moving_averages(data)
    
    # Step 3: Display current information
    current_price = data['Close'].iloc[-1]
    previous_price = data['Close'].iloc[-2]
    change = current_price - previous_price
    change_percent = (change / previous_price) * 100
    
    print(f"\n📊 Current Stock Information:")
    print(f"   Current Price: ${current_price:.2f}")
    print(f"   Daily Change: ${change:.2f} ({change_percent:+.2f}%)")
    print(f"   5-day Average: ${data['MA_5'].iloc[-1]:.2f}")
    print(f"   20-day Average: ${data['MA_20'].iloc[-1]:.2f}")
    
    # Step 4: Make predictions
    predictions = simple_predict_price(data, days_ahead=5)
    
    print(f"\n🎯 5-Day Price Predictions:")
    for i, pred_price in enumerate(predictions, 1):
        change_from_today = ((pred_price - current_price) / current_price) * 100
        print(f"   Day {i}: ${pred_price:.2f} ({change_from_today:+.2f}%)")
    
    # Step 5: Simple recommendation
    final_prediction = predictions[-1]
    total_change = ((final_prediction - current_price) / current_price) * 100
    
    print(f"\n💡 Simple Recommendation:")
    if total_change > 2:
        recommendation = "🟢 POSITIVE - Price may increase"
    elif total_change < -2:
        recommendation = "🔴 NEGATIVE - Price may decrease"
    else:
        recommendation = "🟡 NEUTRAL - Price likely stable"
    
    print(f"   {recommendation}")
    print(f"   Expected 5-day change: {total_change:+.2f}%")
    
    # Step 6: Risk assessment
    volatility = data['Volatility'].iloc[-1]
    avg_volatility = data['Volatility'].mean()
    
    if volatility > avg_volatility * 1.5:
        risk_level = "🔴 HIGH RISK - High volatility"
    elif volatility > avg_volatility:
        risk_level = "🟡 MEDIUM RISK - Moderate volatility"
    else:
        risk_level = "🟢 LOW RISK - Low volatility"
    
    print(f"   Risk Level: {risk_level}")
    
    # Step 7: Create visualization
    print(f"\n📈 Creating charts...")
    create_simple_chart(data, predictions, symbol)
    
    print(f"\n✅ Analysis complete for {symbol}!")
    print("⚠️  Note: This is for educational purposes only, not financial advice.")


def main():
    """
    Main function to run the stock predictor.
    
    ATS Keywords: Main Function, Program Entry Point, User Interface
    """
    print("🤖 AI Stock Price Predictor")
    print("=" * 40)
    print("A simple machine learning project for stock prediction")
    print("Built with Python, pandas, and yfinance")
    
    # List of popular stocks for demonstration
    popular_stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    
    print(f"\n📋 Popular stocks to try: {', '.join(popular_stocks)}")
    
    while True:
        print(f"\n" + "="*50)
        symbol = input("🎯 Enter stock symbol (or 'quit' to exit): ").upper().strip()
        
        if symbol == 'QUIT':
            print("👋 Thank you for using AI Stock Predictor!")
            break
        
        if not symbol:
            print("❌ Please enter a valid stock symbol")
            continue
        
        try:
            analyze_stock(symbol)
        except KeyboardInterrupt:
            print("\n\n👋 Program interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ An error occurred: {str(e)}")
        
        input("\n📱 Press Enter to continue...")


if __name__ == "__main__":
    main()
