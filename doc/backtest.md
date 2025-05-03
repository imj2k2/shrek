# Backtest Module Documentation

## Overview
The `backtest/` directory provides the core infrastructure for evaluating trading strategies, both for single assets and portfolios. It supports flexible agent integration, risk management, metrics, and reporting. The module is designed for extensibility and reproducibility, with CLI and test support.

## Files & Purpose
- **backtest_engine.py**: Main backtesting engine, result container, and example runner.
- **portfolio_backtest.py**: Portfolio-level backtesting for multiple assets/strategies with allocation and caching.
- **run_backtest.py**: CLI entry point for running backtests with argument parsing and reporting.
- **test_backtest.py**: Unit tests for backtesting logic, including mocks and result validation.

---

## backtest_engine.py

### Classes
- **BacktestResult**: Container for backtest results, equity curves, trades, and metrics. Provides plotting and saving utilities.
    - `__init__`: Initializes with strategy name, date range, capital, trades, equity curve, metrics, and optional benchmark.
    - `plot_equity_curve(save_path=None)`: Plots equity curve, drawdowns, and benchmark comparison.
    - `plot_monthly_returns(save_path=None)`: Plots monthly returns heatmap.
    - `plot_trade_distribution(save_path=None)`: Plots profit/loss distribution for trades.
    - `save_results(directory)`: Saves results, metrics, and plots to a directory.

- **Backtester**: Main backtesting engine for trading strategies.
    - `__init__(initial_capital=100000.0, commission=0.001, slippage=0.001, data_directory='backtest_data', data_fetcher=None)`: Sets up backtest environment.
    - `run_backtest(agent, symbols, start_date, end_date, ...)`: Runs backtest for an agent and symbol list over a period, applying risk management and generating metrics.
    - `_fetch_historical_data(symbol, start_date, end_date, timeframe='day')`: Fetches and caches historical data.
    - `_prepare_data_for_agent(hist_data, date, agent)`: Formats data for agent signal generation.
    - `_update_portfolio_prices(portfolio, data, date)`: Updates portfolio prices.
    - `_execute_trade(portfolio, signal, price_data, date)`: Executes trades based on agent signals.
    - `_calculate_equity(portfolio, data, date)`: Computes total equity.
    - `_calculate_benchmark_comparison(equity_curve, benchmark_data, dates)`: Benchmarks vs. reference asset.
    - `_calculate_metrics(equity_curve, trades)`: Computes returns, volatility, Sharpe, drawdown, win rate, etc.

- **run_example_backtest()**: Demonstrates running a backtest with sample configuration.

---

## portfolio_backtest.py

### Classes
- **PortfolioBacktester**: Backtester for portfolios of multiple assets and strategies with customizable weights.
    - `__init__(data_fetcher=None, broker=None)`: Initializes with optional data fetcher and broker, sets up cache.
    - `run_portfolio_backtest(portfolio_config, start_date, end_date, generate_insights=False)`: Runs backtest for a portfolio config, normalizes allocations, and aggregates results.
    - `_combine_results(symbol_results, allocations, start_date, end_date)`: Merges results for portfolio-level performance.
    - `_calculate_portfolio_metrics(equity_curve)`: Computes total/annualized return, Sharpe, drawdown, volatility, win rate.
    - `_get_cache_key(portfolio_config, start_date, end_date)`: Generates cache key for result reuse.
    - `_check_cache(cache_key)`: Checks memory and disk cache.
    - `_save_to_cache(cache_key, result)`: Saves results to cache.
    - `clear_cache()`: Clears all cached results.
    - `get_available_strategies()`: Lists available strategies for portfolio backtesting.

---

## run_backtest.py

- **Purpose**: CLI tool for running backtests via command-line arguments.
- **Key Functions**:
    - `parse_args()`: Parses command-line arguments (agent type, symbols, date range, capital, risk params, output, etc.).
    - `main()`: Initializes agents, risk manager, runs backtest, saves results, and prints/report metrics.

---

## test_backtest.py

- **Purpose**: Unit tests for the backtesting engine and result handling.
- **Key Classes**:
    - `MockDataFetcher`: Supplies synthetic stock/crypto data for tests.
    - `MockStocksAgent`: Generates mock signals for different test scenarios.
    - `TestBacktester`: Test cases for buy-and-hold, trend following, risk manager, multi-symbol, and result methods.

---

## Design Notes
- **Agent-agnostic**: Backtest engine supports any agent with a `generate_signals` method.
- **Risk management**: Integrates with advanced risk manager modules.
- **Metrics**: Computes industry-standard performance metrics.
- **Caching**: Portfolio backtester caches results for reproducibility and speed.
- **Extensibility**: New agents/strategies can be added with minimal changes.

---

*Extend this document as new features or modules are added to the backtesting system.*
