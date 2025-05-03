# Trading Module Documentation

## Overview
The `trading/` directory provides broker integration and portfolio management using the Lumibot framework, supporting Alpaca, Robinhood, and paper trading. It abstracts order placement, portfolio state, and transaction history for use by trading agents and the backend API.

## Files & Purpose
- **lumibot_wrapper.py**: Main wrapper for broker connections, portfolio, and transaction management.

---

## lumibot_wrapper.py

### Class: LumibotBroker
- **Purpose**: Unified broker interface for Alpaca, Robinhood, and paper trading via Lumibot. Handles order execution, portfolio state, and transaction history.
- **Key Methods**:
    - `__init__()`: Initializes broker connection, loads credentials from environment, sets up mock portfolio for paper trading.
    - `_initialize_connection()`: Connects to the selected broker (stubbed for paper/dev mode).
    - `_load_or_create_portfolio()`: Loads or initializes portfolio data from storage.
    - `_load_or_create_transactions()`: Loads or initializes transaction history from storage.
    - `place_order(signal)`: Places buy/sell orders, updates positions, and records transactions.
    - `_update_portfolio_value()`: Recalculates portfolio value, drawdown, and history.
    - `get_portfolio()`: Returns current portfolio state.
    - `get_positions()`: Returns current open positions.
    - `get_transactions(symbol=None, limit=20)`: Returns transaction history, optionally filtered by symbol.
    - `get_cash()`: Returns available cash.
    - `get_buying_power()`: Returns available buying power (cash in this implementation).

### Design Notes
- **Environment Integration**: Loads API keys and credentials from environment variables.
- **Paper Trading**: Fully functional mock trading for development/testing; no real broker connection required.
- **Persistence**: Uses `data/storage.py` for persistent storage of portfolio and transactions.
- **Extensible**: Can be extended for additional brokers or features as needed.

---

*Extend this document as new broker features or trading logic are added.*
