FROM python:3.9

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install additional backtesting libraries
RUN pip install --no-cache-dir backtrader==1.9.78.123

# Set matplotlib to use Agg backend (non-GUI)
ENV MPLBACKEND=Agg

# Copy project
COPY . .

# Create directories for data persistence
RUN mkdir -p /app/data /app/backtest_results /app/data/db

# Expose ports for FastAPI and Gradio
EXPOSE 8000 7860

# Default command
CMD ["uvicorn", "ui.app:app", "--host", "0.0.0.0", "--port", "8000"]
