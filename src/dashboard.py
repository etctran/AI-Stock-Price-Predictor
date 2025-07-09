"""
Streamlit Dashboard for AI Stock Price Predictor

Interactive web interface for stock price prediction and analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import time

# Import our new risk analysis modules
try:
    from risk_analysis import analyze_trading_performance
    from var_calculator import PortfolioRiskManager
    RISK_MODULES_AVAILABLE = True
except ImportError:
    RISK_MODULES_AVAILABLE = False

# Set page configuration
st.set_page_config(
    page_title="AI Stock Predictor",
    page_icon="📈",
    layout="wide"
)

# Main title
st.title("📈 AI Stock Price Predictor")
st.write("A simple machine learning dashboard for predicting stock prices")

# Sidebar for inputs
st.sidebar.header("Control Panel")

# Stock symbol input
symbol = st.sidebar.text_input("Enter Stock Symbol:", value="AAPL")
symbol = symbol.upper()

# Number of days to predict
prediction_days = st.sidebar.slider("Days to Predict:", 1, 30, 5)

# Prediction button
if st.sidebar.button("Make Prediction"):
    st.session_state.make_prediction = True

# Main dashboard area
col1, col2 = st.columns([3, 2])

# Left column - Charts and data
with col1:
    st.subheader(f"Stock Analysis for {symbol}")
    
    try:
        # Get stock data using yfinance
        print("Loading stock data...")
        stock = yf.Ticker(symbol)
        hist_data = stock.history(period="6mo")
        
        if not hist_data.empty:
            # Current price info
            current_price = hist_data['Close'].iloc[-1]
            previous_price = hist_data['Close'].iloc[-2]
            price_change = current_price - previous_price
            change_percent = (price_change / previous_price) * 100
            
            # Display current price
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Current Price", f"${current_price:.2f}")
            with col_b:
                st.metric("Daily Change", f"${price_change:.2f}", f"{change_percent:.2f}%")
            with col_c:
                st.metric("Volume", f"{hist_data['Volume'].iloc[-1]:,.0f}")
            
            # Price chart
            st.write("**Price History (6 Months)**")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(hist_data.index, hist_data['Close'], color='#1f77b4', linewidth=2)
            ax.set_title(f"{symbol} Stock Price History")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price ($)")
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
            # Volume chart
            st.write("**Trading Volume**")
            fig2, ax2 = plt.subplots(figsize=(12, 4))
            ax2.bar(hist_data.index, hist_data['Volume'], color='orange', alpha=0.7)
            ax2.set_title(f"{symbol} Trading Volume")
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Volume")
            plt.xticks(rotation=45)
            st.pyplot(fig2)
            
        else:
            st.error(f"Could not find data for {symbol}")
            
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")

# Right column - Predictions and info
with col2:
    st.subheader("AI Predictions")
    
    # Check if prediction button was clicked
    if hasattr(st.session_state, 'make_prediction') and st.session_state.make_prediction:
        
        with st.spinner("AI is analyzing..."):
            time.sleep(2)  # Simulate processing
            
        try:
            # Simple prediction logic (for demonstration)
            stock = yf.Ticker(symbol)
            data = stock.history(period="1mo")
            current_price = data['Close'].iloc[-1] if not data.empty else 100.0
            
            # Simple prediction using random walk with trend
            np.random.seed(hash(symbol) % 1000)  # Consistent randomness per symbol
            
            # Calculate simple moving average trend
            returns = data['Close'].pct_change().dropna()
            avg_return = returns.mean()
            volatility = returns.std()
            
            # Generate predictions
            predictions = []
            price = current_price
            
            for day in range(prediction_days):
                # Simple random walk with slight trend
                change = np.random.normal(avg_return, volatility)
                price = price * (1 + change)
                predictions.append(price)
            
            # Create prediction dates
            future_dates = []
            for i in range(1, prediction_days + 1):
                future_date = datetime.now() + timedelta(days=i)
                future_dates.append(future_date.strftime("%Y-%m-%d"))
            
            # Display predictions
            st.success("Prediction Complete!")
            
            # Show predicted price
            final_predicted_price = predictions[-1]
            total_change = final_predicted_price - current_price
            total_change_percent = (total_change / current_price) * 100
            
            st.metric(
                f"Predicted Price ({prediction_days} days)",
                f"${final_predicted_price:.2f}",
                f"{total_change_percent:+.2f}%"
            )
            
            # Risk assessment (simple)
            if abs(total_change_percent) < 2:
                risk_level = "Low Risk"
                recommendation = "HOLD"
            elif abs(total_change_percent) < 5:
                risk_level = "Medium Risk"
                recommendation = "MONITOR"
            else:
                risk_level = "High Risk"
                recommendation = "CAUTION"
            
            st.write(f"**Risk Level:** {risk_level}")
            st.write(f"**Recommendation:** {recommendation}")
            
            # Add risk-adjusted performance analysis if available
            if RISK_MODULES_AVAILABLE:
                st.write("**📊 Risk-Adjusted Performance**")
                perf_results = analyze_trading_performance(symbol)
                if perf_results:
                    col_perf1, col_perf2 = st.columns(2)
                    with col_perf1:
                        st.metric("Sharpe Ratio", f"{perf_results['strategy_sharpe']:.3f}")
                        st.metric("Alpha", f"{perf_results['alpha']:+.2f}%")
                    with col_perf2:
                        st.metric("Beta", f"{perf_results['beta']:.3f}")
                        st.metric("vs Benchmark", f"{perf_results['outperformance']:+.1f}%")
            
            # Prediction chart
            st.write("**Price Prediction**")
            
            # Combine historical and predicted data
            last_5_days = data['Close'].tail(5)
            all_dates = list(last_5_days.index.strftime("%Y-%m-%d")) + future_dates
            all_prices = list(last_5_days.values) + predictions
            
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            
            # Historical prices
            ax3.plot(range(len(last_5_days)), last_5_days.values, 
                    'o-', color='blue', label='Historical', linewidth=2)
            
            # Predicted prices
            ax3.plot(range(len(last_5_days)-1, len(all_prices)), 
                    all_prices[len(last_5_days)-1:], 
                    'o--', color='red', label='Predicted', linewidth=2)
            
            ax3.set_title(f"{symbol} Price Prediction")
            ax3.set_xlabel("Days")
            ax3.set_ylabel("Price ($)")
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Set x-axis labels
            ax3.set_xticks(range(len(all_dates)))
            ax3.set_xticklabels([d.split('-')[1] + '-' + d.split('-')[2] for d in all_dates], 
                               rotation=45)
            
            st.pyplot(fig3)
            
            # Prediction table
            st.write("**Daily Predictions**")
            prediction_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted Price': [f"${p:.2f}" for p in predictions],
                'Change from Today': [f"{((p-current_price)/current_price*100):+.1f}%" for p in predictions]
            })
            st.dataframe(prediction_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
        
        # Reset the prediction trigger
        st.session_state.make_prediction = False

# Footer
st.markdown("---")
st.write("**⚠️ Disclaimer:** This is for educational purposes only. Not financial advice.")

# Instructions for running
st.sidebar.markdown("---")
st.sidebar.subheader("Instructions")
st.sidebar.write("""
1. Enter a stock symbol (e.g., AAPL, GOOGL)
2. Choose prediction timeframe
3. Click 'Make Prediction'
4. View results and charts
""")
