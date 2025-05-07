# Shrek Trading Platform - Development Context

## Project Overview
Shrek is an algorithmic trading platform with backtesting capabilities, stock screening, and portfolio management. The platform is containerized using Docker and consists of a FastAPI backend and a Gradio UI frontend.

## Key Components

### Backend
- **FastAPI** server providing API endpoints for backtesting, screening, and database operations
- **StocksAgent** class that implements trading strategies (momentum, mean_reversion, breakout)
- **Backtester** engine for simulating trades on historical data
- **MarketDatabase** for storing and retrieving market data
- **AdvancedRiskManager** for managing trading risk

### Frontend
- **Gradio** UI for interactive use of the platform
- Multiple UI files (simple_ui.py, clean_ui.py, etc.) with simple_ui.py being the primary one

### Data Sources
- **Polygon.io** API for market data
- Local caching in SQLite database
- S3 bucket for historical data

## Recent Fixes and Improvements

### Backtesting Engine
1. **Symbol Handling**: Fixed critical issue where the backtesting engine was using "UNKNOWN" as the symbol instead of the actual stock symbol.
2. **Price Data Handling**: Enhanced to handle different column names for price data ('close', 'Close', 'c') and to use a fixed price for testing when needed.
3. **Trade Execution**: Improved validation of trade parameters and fixed quantity handling.
4. **Strategy Selection**: Modified to properly use the requested strategy (momentum, mean_reversion, breakout) instead of always using debug_forced strategy.
5. **Data Type Handling**: Added support for different data types (numpy arrays, pandas Series, lists) in the StocksAgent.

### Risk Management
1. **Position Format Handling**: Modified the risk manager to handle the dictionary format of positions coming from the backtest engine.
2. **Trailing Stops**: Fixed to properly work with the backtest engine's position format.

### Database
1. **Connection Handling**: Improved thread-local connection management and error handling.
2. **Error Recovery**: Added ability to recover from connection failures.
3. **Data Type Conversion**: Better handling of various data formats and types.

### API Endpoints
1. **Screener API**: Added proper FastAPI endpoint for stock screening.
2. **Backtest Parameters**: Updated to properly handle strategy_name parameter.
3. **Postman Collection**: Created comprehensive test cases for different strategies and screener configurations.

## Testing
- Use the `shrek_api_postman_collection_fixed.json` file for API testing
- Run with Newman: `newman run shrek_api_postman_collection_fixed.json -r htmlextra`

## Docker Setup
- Run with `docker-compose up -d --build` to rebuild and start containers
- Backend runs on port 8080
- Gradio UI runs on port 7860

## Known Issues
- Database connection may fail in some edge cases
- Some price data formats may not be handled correctly
- Forced trade generation is now disabled by default but can be enabled for testing

## Next Steps
1. Further improve error handling in the backtest engine
2. Enhance the screener functionality with more criteria
3. Add more comprehensive testing for different market conditions
4. Improve the UI for better user experience
