# Config & Data Modules Documentation

## config/

### settings.py
- **Purpose**: Centralizes all environment variable-based configuration for API keys, credentials, and service URLs.
- **Settings class**: Uses `pydantic.BaseSettings` for type safety and environment integration.
    - `ALPACA_API_KEY`, `ALPACA_API_SECRET`: For Alpaca brokerage API.
    - `ROBINHOOD_USERNAME`, `ROBINHOOD_PASSWORD`: Robinhood credentials.
    - `DISCORD_TOKEN`: Discord bot authentication.
    - `REDIS_URL`: Redis connection string.
    - `POLYGON_API_KEY`: Polygon.io market data API.
    - `LUMIBOT_CONFIG`: Lumibot trading framework config.
- **Usage**: Import and use `settings = Settings()` throughout the codebase for configuration access.

---

## data/

### data_fetcher.py
- **Purpose**: Provides unified access to stock, crypto, and options market data from APIs (Polygon.io, Yahoo Finance) or generates mock data for backtesting.
- **DataFetcher class**:
    - `__init__()`: Loads API keys, sets up logger.
    - `fetch_stock_data(symbol, start_date, end_date, source=None)`: Gets stock data from Polygon.io or Yahoo Finance, with fallback to mock data.
    - `fetch_crypto_data(symbol, start_date, end_date, interval="1d")`: Gets crypto data or generates mock data.
    - `fetch_options_data(symbol)`: Gets options data from Polygon.io.
    - `_generate_mock_data(symbol, start_date_str, end_date_str, is_crypto=False)`: Creates synthetic data for testing and backtests.
- **Design**: Robust to API failure, logs all actions, and ensures data availability for all modules.

### storage.py
- **Purpose**: Abstracts persistent storage for caching and state, supporting both Redis (if available) and local JSON file fallback.
- **Storage class**:
    - `__init__(redis_url=None, fallback_file='storage.json')`: Connects to Redis or sets up file-based fallback.
    - `set(key, value)`: Stores a value (dict) by key.
    - `get(key)`: Retrieves a value by key.
    - `_load_file()`: Loads data from the fallback JSON file.
- **Design**: Allows seamless switch between in-memory (Redis) and disk-based storage, supporting stateless and stateful deployments.

---

*Extend this document as new configuration or data-handling features are added.*
