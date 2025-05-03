# Shrek Project Documentation

## Project Overview
Shrek is a modular trading and financial application designed for multi-asset algorithmic trading, backtesting, and research. It uses a Docker-based architecture with the following services:
- **Backend API (FastAPI)**: Core trading logic and endpoints (port 8000)
- **Gradio UI**: User interface for interaction and visualization (port 7860)
- **Redis**: For caching and inter-process communication

The project is implemented in Python 3.9 and is orchestrated via Docker Compose. All development and deployment workflows use `./run_docker.sh --rebuild` for rebuilding and restarting containers.

## Directory Structure
```
shrek/
├── agents/           # Trading agents and strategy coordination
├── backtest/         # Backtesting engine and utilities
├── config/           # Configuration and settings
├── core/             # (To be documented)
├── data/             # Data fetching and storage
├── discord_bot/      # Discord bot integration
├── indicators/       # Technical indicators (ATR, RSI, MACD, etc.)
├── mcp/              # Protocol logic
├── risk/             # Risk management modules
├── trading/          # Trading logic, wrappers
├── ui/               # Gradio UI, FastAPI endpoints, schemas
├── Dockerfile        # Docker build file
├── docker-compose.yml# Docker Compose orchestration
├── run_docker.sh     # Build/run helper script
├── requirements.txt  # Python dependencies
└── ...
```

## Architecture & Design
- **Modular agent-based design**: Each asset class (stocks, crypto, options) has its own agent, coordinated by a central `CoordinatorAgent`.
- **Strategy separation**: Strategies (momentum, mean reversion, breakout, value, etc.) are encapsulated in agent methods.
- **Indicator abstraction**: Technical indicators are implemented in the `indicators/` module and used by agents.
- **Backtesting**: The `backtest/` module provides a flexible engine for evaluating strategies.
- **Risk management**: Dedicated risk modules assess portfolio and trade risk.
- **UI/API**: User interaction via Gradio and REST API endpoints.

## Module Documentation

### agents/
- **Purpose**: Contains trading agents for different asset classes and the coordinator logic.
- **Key Files**:
  - `coordinator.py`: Defines `CoordinatorAgent`, which manages multiple asset agents and coordinates their signals.
  - `crypto_agent.py`: Implements `CryptoAgent` for crypto-specific signal generation.
  - `stocks_agent.py`: Implements `StocksAgent` for stock trading strategies (momentum, mean reversion, breakout, value investing, etc.).

#### coordinator.py
- **CoordinatorAgent**
  - `__init__(agents: Dict[str, Any])`: Initializes with a dictionary of agents.
  - `coordinate(market_data)`: Collects signals from all agents and coordinates allocations.
- **StocksAgent/OptionsAgent/CryptoAgent/RiskAgent/TradingExecutor**: Lightweight wrappers or templates for asset-specific logic, risk checks, and trade execution.

#### crypto_agent.py
- **CryptoAgent**
  - `__init__()`: Initializes the agent.
  - `generate_signals(data)`: Computes signals using MACD, VWAP, RSI, Bollinger Bands, ATR, and EMA-20 from the `indicators` module. Returns a dict of computed indicator values for trading decisions.

#### stocks_agent.py
- **StocksAgent**
  - `__init__(mcp=None)`: Sets up logging, strategy parameters, position sizing, and tracking variables. Optionally connects to a Model Context Protocol (MCP).
  - `generate_signals(data: Dict[str, Any])`: Main entry for generating trading signals. Integrates multiple strategies and can force trades in debug mode.
  - `_momentum_strategy(close)`: Uses RSI and MACD for momentum signals.
  - `_mean_reversion_strategy(close, high=None, low=None)`: Uses Bollinger Bands and VWAP for mean reversion.
  - `_breakout_strategy(close, high=None, low=None, volume=None)`: Uses ATR and moving averages for breakout signals.
  - `_combine_signals(strategy_signals)`: Combines signals from different strategies with weights.
  - `_apply_position_sizing(signals, data)`: Applies position sizing rules.
  - `update_performance(trade_result)`: Updates agent performance metrics.
  - `_adjust_strategies_based_on_performance()`: Dynamically adjusts strategy weights.
  - `_value_strategy(data)`: Implements value investing logic.

### [Other modules will be documented in similar detail.]

---

## Development & Usage
- **Rebuilding/Restarting**: Run `./run_docker.sh --rebuild` after code changes.
- **Adding Documentation**: Extend this file and add module-specific docs (e.g., `doc/backtest.md`, `doc/indicators.md`, etc.).

## [To Do]
- Fill in detailed documentation for each remaining module and function.
- Add diagrams or sequence flows for agent coordination and backtesting.
- Link to API reference and UI usage guides.

---

*This document is auto-generated and should be extended as the project evolves.*
