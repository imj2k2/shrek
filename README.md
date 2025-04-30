# AI-driven Multi-Agent Algorithmic Trading Platform

# AI-Driven Multi-Agent Trading Platform

## Overview
This platform leverages multi-agent AI for automated trading across stocks, options, and crypto, with real-time data, risk management, and interactive UI/Discord integration. The system is designed to be robust, with multiple data source fallbacks, comprehensive error handling, and a user-friendly interface.

---

## Features
- **Modular Agent Architecture**: Specialized agents for stocks, options, crypto, risk management, and coordination
- **Robust Data Pipeline**: Primary Polygon.io API with fallbacks to Yahoo Finance and realistic mock data generation
- **Interactive UI**: Gradio-based interface for portfolio management, backtesting, and performance analysis
- **Advanced Technical Analysis**: Comprehensive technical indicators (RSI, MACD, Bollinger Bands, ATR, etc.)
- **Risk Management**: Real-time risk assessment and position sizing
- **Discord Integration**: Real-time alerts and notifications
- **Containerized Deployment**: Complete Docker Compose setup for easy deployment
- **Persistent Storage**: Redis for state management and data persistence

---

## Detailed Setup Guide

### Prerequisites
- Docker and Docker Compose installed
- API keys for data providers (Polygon.io recommended)
- Broker API keys (if using live trading)

### Step 1: Clone the Repository
```bash
git clone <repo-url>
cd shrek
```

### Step 2: Configure API Keys
Edit the `run_docker.sh` script to set your API keys:

```bash
# Set your API keys here
POLYGON_API_KEY="your_polygon_api_key_here"
export POLYGON_API_KEY=${POLYGON_API_KEY:-"demo"}
export ALPACA_API_KEY=${ALPACA_API_KEY:-"demo"}
export ALPACA_API_SECRET=${ALPACA_API_SECRET:-"demo"}
export DISCORD_TOKEN=${DISCORD_TOKEN:-"demo"}
export BROKER_TYPE=${BROKER_TYPE:-"mock"}
```

**Important API Key Notes:**
- **Polygon.io**: Required for real market data. Without a valid key, the system will fall back to Yahoo Finance or mock data.
- **Alpaca**: Required for live trading. Use "mock" for BROKER_TYPE to run in simulation mode.
- **Discord**: Optional for notifications. System works without it.

### Step 3: Run the Platform
Use the provided script to start all services:

```bash
# Make the script executable if needed
chmod +x run_docker.sh

# Start the platform
./run_docker.sh
```

Additional script options:
```bash
# Rebuild containers
./run_docker.sh --rebuild

# View logs
./run_docker.sh --logs

# Stop all services
./run_docker.sh --down
```

### Step 4: Access the Platform
Once running, access the platform through:
- **Gradio UI**: http://localhost:7860 - For portfolio management, backtesting, and visualization
- **FastAPI Docs**: http://localhost:8000/docs - For API documentation and direct API access
- **Discord Bot**: If configured, the bot will be active in your Discord server

---

## Architecture
- `agents/`: Modular agent implementations for different asset classes and strategies
- `indicators/`: Technical indicators for market analysis
- `mcp/`: Model Context Protocol for agent communication and coordination
- `trading/`: Broker integration (Lumibot, Alpaca, Robinhood) for execution
- `risk/`: Real-time risk management and position sizing
- `data/`: Market data fetching and persistent storage
- `ui/`: FastAPI backend and Gradio UI components
- `discord_bot/`: Discord integration for alerts and monitoring
- `backtest/`: Backtesting engine for strategy evaluation
- `docker-compose.yml`: Container orchestration
- `run_docker.sh`: Helper script for managing the platform

---

## Detailed Usage Guide

### Backtesting Strategies

1. **Access the Backtesting Interface**:
   - Navigate to the Gradio UI at http://localhost:7860
   - Select the "Backtest" tab

2. **Configure Your Backtest**:
   - **Agent**: Select the trading agent/strategy (value_agent, trend_agent, etc.)
   - **Symbols**: Enter comma-separated ticker symbols (e.g., "AAPL,MSFT,GOOG")
   - **Date Range**: Set the start and end dates for your backtest
   - **Initial Capital**: Set the starting capital amount

3. **Run and Analyze Results**:
   - Click "Run Backtest" to execute
   - Review the performance metrics at the top of the results
   - Examine the equity curve visualization
   - Check which data source was used for each symbol (Polygon, Yahoo, or Mock)
   - Review the detailed JSON results for more information

### Portfolio Management

1. **View Your Portfolio**:
   - Navigate to the "Portfolio" tab in the Gradio UI
   - See current positions, equity, and performance metrics
   - Use the "Refresh" button to update with latest data

2. **Execute Trades** (when using a live broker):
   - Configure your broker settings in the run_docker.sh script
   - Use the trading interface to place orders
   - Monitor execution in the portfolio view

### Data Source Management

The platform uses a tiered approach to data sources:

1. **Polygon.io** (Primary): Used when a valid API key is provided
2. **Yahoo Finance** (Secondary): Used if Polygon fails or no key is available
3. **Mock Data** (Fallback): Generated with realistic patterns if both real sources fail

The data source used for each symbol is clearly indicated in the backtest results.

### Discord Integration

If you've configured a Discord bot token:

1. Invite the bot to your Discord server
2. Configure the channels for different alert types
3. Receive real-time notifications about:
   - Trading signals
   - Position changes
   - Risk alerts
   - Performance updates

---

## Extending the Platform

### Adding New Strategies

1. Create a new agent class in the `agents/` directory
2. Implement the required methods for analysis and signal generation
3. Register the agent in the coordinator
4. Add the new agent type to the UI options

### Customizing Risk Management

Modify the `risk/advanced_risk_manager.py` file to adjust:
- Maximum drawdown limits
- Position sizing rules
- Stop-loss and take-profit levels
- Portfolio concentration limits

---

## Troubleshooting

### Common Issues

1. **Data Source Errors**:
   - Check your Polygon API key if you're seeing "401" errors
   - Ensure internet connectivity for external data sources
   - The system will fall back to mock data automatically

2. **Docker Issues**:
   - Run `docker-compose down` followed by `docker-compose up --build` to rebuild
   - Check logs with `./run_docker.sh --logs`
   - Ensure ports 8000, 7860, and 6379 are available

3. **Performance Issues**:
   - Adjust the number of symbols or date range for faster backtests
   - Consider using a machine with more resources for large backtests

---

## Security
- All secrets/configs are managed via environment variables
- No credentials are hardcoded in the codebase
- Redis data is persisted in a Docker volume
- API access is restricted to localhost by default

---

## License
MIT License (or specify your license)
