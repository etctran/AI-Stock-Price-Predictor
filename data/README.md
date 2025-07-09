# Data Directory

This directory contains stock market data used by the predictor.

## Structure:
- `raw/` - Raw stock data from Yahoo Finance
- `processed/` - Cleaned data with technical indicators

## Data Sources:
- Yahoo Finance API via yfinance library
- Real-time stock prices and trading volumes

## Usage:
Data is automatically downloaded when you run the prediction scripts. No manual data management needed.

**Note:** Large data files are excluded from version control.
