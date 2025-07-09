# AI Stock Price Predictor

A simple machine learning project to predict stock prices using Python and basic technical analysis.

## What it does

This project fetches real stock data and uses simple algorithms to predict future stock prices. It's great for learning about:

- Data analysis with pandas
- API integration with Yahoo Finance
- Basic machine learning concepts
- Web dashboards with Streamlit
- Financial data visualization

## How to run it

### Quick start
```bash
python run.py
```

### Manual setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run the web dashboard
streamlit run src/dashboard.py

# Or run the command line version
python src/main.py
```

## Features

- **Stock data collection** - Gets real-time data from Yahoo Finance
- **Technical indicators** - Calculates moving averages and trends
- **Price prediction** - Simple forecasting based on historical patterns
- **Interactive dashboard** - Clean web interface built with Streamlit
- **Risk analysis** - Basic volatility and risk assessment
- **Data visualization** - Charts and graphs showing price history and predictions

## File structure

```
src/
├── main.py           # Command line interface
├── dashboard.py      # Web dashboard (Streamlit)
├── simple_predictor.py  # Core prediction logic
├── risk_analysis.py  # Risk calculation functions
└── database.py       # Data storage utilities
```

## Technologies used

- **Python 3.8+**
- **pandas** - Data manipulation
- **yfinance** - Stock data API
- **scikit-learn** - Machine learning
- **streamlit** - Web dashboard
- **matplotlib** - Data visualization
- **numpy** - Numerical computing

## Example usage

```python
# Get stock data
import yfinance as yf
data = yf.download('AAPL', period='1y')

# Make prediction
from src.simple_predictor import predict_price
prediction = predict_price('AAPL', days=5)
```

## Disclaimer

This is for educational purposes only. Not financial advice.
