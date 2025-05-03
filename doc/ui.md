# UI Module Documentation

## Overview
The `ui/` directory implements the user interface and API endpoints for interacting with the Shrek trading platform. It provides both a FastAPI backend (for REST API and agent orchestration) and a Gradio-based web frontend for portfolio management, backtesting, and strategy deployment.

## Files & Purpose
- **app.py**: FastAPI application setup, CORS configuration, and router inclusion.
- **gradio_ui.py**: Gradio-based frontend for portfolio, backtesting, and strategy management.
- **portfolio_api.py**: FastAPI endpoints for portfolio backtesting.
- **schemas.py**: API schemas, request/response models, and endpoints for agent, backtest, and strategy management.

---

## File-Level Documentation

### app.py
- **Purpose**: Initializes FastAPI app, configures CORS, and registers routers for agent and portfolio APIs.
- **Key Endpoints**:
    - `/ping`: Health check endpoint.

### gradio_ui.py
- **Purpose**: Implements the Gradio web UI for running backtests, viewing portfolios, deploying strategies, and visualizing results.
- **Key Functions**:
    - `create_ui()`: Constructs the Gradio interface with tabs for portfolio, backtesting, and strategy management.
    - `update_portfolio()`: Fetches and displays current portfolio from backend API.
    - `run_backtest(...)`: Runs backtests via API and visualizes results.
    - `plot_backtest_results(results)`: Generates Plotly charts for backtest results.
    - `run_portfolio_backtest(...)`: Runs portfolio-level backtests and displays results.
    - `plot_portfolio_results(results)`: Plots portfolio backtest results.
    - `launch_gradio()`: Launches the UI (used in main block or container entrypoint).
- **Design**: Modular, extensible, and supports both direct API integration and standalone launch.

### portfolio_api.py
- **Purpose**: Defines API endpoints for running portfolio backtests using FastAPI.
- **Key Classes & Endpoints**:
    - `PortfolioBacktestRequest`: Pydantic model for portfolio backtest requests.
    - `/portfolio/backtest`: POST endpoint to run portfolio backtests.
- **Design**: Integrates with backtest and data modules for comprehensive portfolio analysis.

### schemas.py
- **Purpose**: Centralizes API schemas, request/response models, and agent orchestration logic.
- **Key Classes & Endpoints**:
    - `BacktestRequest`, `StrategyDeployRequest`, `StrategyListResponse`, etc.: Pydantic models for API requests/responses.
    - `/backtest/run`: POST endpoint to run backtests.
    - `/strategy/deploy`: POST endpoint to deploy strategies for live trading.
    - `/portfolio`: GET endpoint for portfolio state.
    - `/risk/metrics`: GET endpoint for risk metrics.
    - `/strategy/list`, `/strategy/{id}`: Endpoints for strategy repository management.
- **Design**: Connects agents, broker, and backtester; provides a unified API for frontend and external integrations.

---

## Design Notes
- **Separation of Concerns**: API and UI logic are separated for maintainability.
- **Extensible**: New endpoints and UI features can be added with minimal changes.
- **Containerized**: Gradio UI and FastAPI backend run in separate containers for scalability.

---

*Extend this document as new UI features or API endpoints are added.*
