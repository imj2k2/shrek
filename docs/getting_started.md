# Getting Started

This guide will help you set up and run the Shrek Trading Platform on your local machine.

## Prerequisites

- Docker and Docker Compose
- Git
- Polygon.io API key (and optionally S3 credentials)
- Alpaca API key and secret (optional for live trading)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/shrek.git
   cd shrek
   ```

2. Create your environment file:
   ```bash
   cp .env.template .env
   ```

3. Edit the `.env` file with your API keys and configuration.

## Running the Platform

The platform uses Docker for easy deployment. Use the provided `run_docker.sh` script to manage the containers:

```bash
# Start the platform
./run_docker.sh

# Rebuild and restart (after code changes)
./run_docker.sh --rebuild

# View logs
./run_docker.sh --logs

# Stop the platform
./run_docker.sh --stop
```

## Accessing the UI

Once the platform is running, you can access the Gradio UI at:
- http://localhost:7860

The backend API is available at:
- http://localhost:8080

## Initial Setup

1. **Refresh the Database**: 
   - Go to the "Database Management" tab
   - Select your preferred data source
   - Click "Refresh Database"

2. **Run a Backtest**:
   - Go to the "Backtest" tab
   - Enter symbols, date range, and strategy
   - Click "Run Backtest"

3. **Deploy a Strategy** (optional):
   - Go to the "Deploy to Alpaca" tab
   - Enter your Alpaca credentials
   - Configure your strategy
   - Click "Deploy to Alpaca"

## Project Structure

- `/data`: Database and data synchronization code
- `/ui`: User interface code (Gradio)
- `/agents`: Trading agents implementation
- `/backtest`: Backtesting engine
- `/trading`: Trading execution code
- `/docs`: Documentation

## Troubleshooting

If you encounter issues:

1. Check the logs: `./run_docker.sh --logs`
2. Verify your API keys in the `.env` file
3. Ensure Docker is running properly
4. Try rebuilding the containers: `./run_docker.sh --rebuild`

## Next Steps

- Explore the [Database Management](database_management.md) documentation
- Learn about [Data Sources](data_sources.md)
- Understand the [Environment Configuration](environment_configuration.md)
- Try [Backtesting](backtesting.md) your strategies
