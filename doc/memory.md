Backtesting authenticity: The backtesting feature needs to maintain authenticity with real trading strategies against real market data. Trades should be triggered by genuine market conditions, not artificially forced.
Shrek Trading Platform architecture: A Docker-based architecture with FastAPI backend, Gradio UI, and Redis.
The platform includes screener-backtesting integration allowing for both long and short positions.
When code is updated, we should run ./run_docker.sh --rebuild to rebuild containers.
Proper handling of stock screener to backtest tab integration is important.

