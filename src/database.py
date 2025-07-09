"""
Database Integration for AI Stock Price Predictor

SQLite database storage for predictions and analysis results.
"""

import sqlite3
import pandas as pd
from datetime import datetime

class SimpleDatabase:
    """Simple database class for storing predictions."""
    
    def __init__(self, db_path="stock_predictions.db"):
        """Initialize database connection."""
        self.db_path = db_path
        self.create_tables()
    
    def create_tables(self):
        """Create database tables for storing predictions."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                current_price REAL,
                predicted_price REAL,
                prediction_date DATE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        print("Database tables created successfully")
    
    def save_prediction(self, symbol, current_price, predicted_price):
        """Save prediction to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions (symbol, current_price, predicted_price, prediction_date)
            VALUES (?, ?, ?, ?)
        ''', (symbol, current_price, predicted_price, datetime.now().date()))
        
        conn.commit()
        conn.close()
        print(f"Prediction saved to database for {symbol}")
    
    def get_predictions(self, symbol=None):
        """Retrieve predictions from database."""
        conn = sqlite3.connect(self.db_path)
        
        if symbol:
            query = "SELECT * FROM predictions WHERE symbol = ? ORDER BY created_at DESC"
            df = pd.read_sql_query(query, conn, params=(symbol,))
        else:
            query = "SELECT * FROM predictions ORDER BY created_at DESC LIMIT 10"
            df = pd.read_sql_query(query, conn)
        
        conn.close()
        return df

# Example usage
if __name__ == "__main__":
    db = SimpleDatabase()
    print("Database integration ready!")
