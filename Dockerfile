# Simple Dockerfile for AI Stock Price Predictor
# Enables containerization and cloud deployment with minimal setup

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY *.md ./

# Expose port for web application
EXPOSE 8501

# Set environment variables
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_PORT=8501

# Default command to run Streamlit dashboard
CMD ["streamlit", "run", "src/dashboard.py", "--server.address", "0.0.0.0"]
