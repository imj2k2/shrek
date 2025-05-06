# Environment Configuration

The Shrek Trading Platform uses environment variables for configuration to ensure security and flexibility. This document explains how to configure the environment for development, testing, and production.

## Environment Files

The platform uses `.env` files to store environment variables. These files are excluded from version control to prevent exposing sensitive information like API keys.

### .env.template

A template file (`.env.template`) is provided as a reference for creating your own `.env` file. This template includes all required environment variables with placeholder values.

```bash
# Polygon.io API Key
POLYGON_API_KEY=your_polygon_api_key_here

# Polygon.io S3 Credentials
POLYGON_S3_ACCESS_KEY=your_polygon_s3_access_key_here
POLYGON_S3_SECRET_KEY=your_polygon_s3_secret_key_here

# Polygon.io S3 Fallback (set to true to fallback to API if S3 fails)
POLYGON_S3_FALLBACK=true

# Alpaca API Credentials
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_API_SECRET=your_alpaca_api_secret_here

# Discord Bot Token
DISCORD_TOKEN=your_discord_token_here

# Broker Configuration
BROKER_TYPE=mock  # Options: mock, paper, alpaca, robinhood

# Debug Mode
DEBUG_TRADING=true  # Set to false in production
```

### Creating Your .env File

To create your own `.env` file:

1. Copy the template: `cp .env.template .env`
2. Edit the `.env` file and replace placeholder values with your actual API keys and configuration

## Docker Integration

The Docker setup automatically loads environment variables from the `.env` file if it exists. If the file doesn't exist, default values are used.

### run_docker.sh Script

The `run_docker.sh` script handles loading environment variables:

```bash
# Load environment variables from .env file if it exists
if [ -f ".env" ]; then
  echo "Loading environment variables from .env file..."
  set -a  # automatically export all variables
  source .env
  set +a  # stop automatically exporting
else
  echo "Warning: .env file not found. Using default values."
  # You can create a .env file from the template with: cp .env.template .env
fi

# Set default values for environment variables if not set in .env
export POLYGON_API_KEY=${POLYGON_API_KEY:-"demo"}
export ALPACA_API_KEY=${ALPACA_API_KEY:-"demo"}
export ALPACA_API_SECRET=${ALPACA_API_SECRET:-"demo"}
export DISCORD_TOKEN=${DISCORD_TOKEN:-"demo"}
export BROKER_TYPE=${BROKER_TYPE:-"mock"}
export DEBUG_TRADING=${DEBUG_TRADING:-"true"}
```

### Docker Compose Configuration

The `docker-compose.yml` file passes environment variables to the containers:

```yaml
services:
  backend:
    # ...
    environment:
      - REDIS_URL=redis://redis:6379/0
      - POLYGON_API_KEY=${POLYGON_API_KEY:-demo}
      - POLYGON_S3_ACCESS_KEY=${POLYGON_S3_ACCESS_KEY:-${POLYGON_API_KEY:-demo}}
      - POLYGON_S3_SECRET_KEY=${POLYGON_S3_SECRET_KEY:-${POLYGON_API_KEY:-demo}}
      - POLYGON_S3_FALLBACK=${POLYGON_S3_FALLBACK:-true}
      - ALPACA_API_KEY=${ALPACA_API_KEY:-demo}
      - ALPACA_API_SECRET=${ALPACA_API_SECRET:-demo}
      - BROKER_TYPE=${BROKER_TYPE:-mock}
      - DEBUG_TRADING=${DEBUG_TRADING:-true}
      - DB_PATH=/app/data/db/market_data.db
    # ...
```

## Environment Variables Reference

| Variable | Description | Default Value |
|----------|-------------|---------------|
| `POLYGON_API_KEY` | API key for Polygon.io | "demo" |
| `POLYGON_S3_ACCESS_KEY` | S3 access key for Polygon.io | Same as `POLYGON_API_KEY` |
| `POLYGON_S3_SECRET_KEY` | S3 secret key for Polygon.io | Same as `POLYGON_API_KEY` |
| `POLYGON_S3_FALLBACK` | Whether to fallback to API if S3 fails | "true" |
| `ALPACA_API_KEY` | API key for Alpaca | "demo" |
| `ALPACA_API_SECRET` | API secret for Alpaca | "demo" |
| `DISCORD_TOKEN` | Token for Discord bot | "demo" |
| `BROKER_TYPE` | Type of broker to use | "mock" |
| `DEBUG_TRADING` | Enable debug mode for trading | "true" |

## Security Considerations

- Never commit `.env` files to version control
- Use different environment configurations for development, testing, and production
- Regularly rotate API keys and update your `.env` files
- Use the least privileged API keys possible for your use case

## Troubleshooting

If you encounter issues with environment variables:

1. Verify that your `.env` file exists and contains the correct values
2. Check the Docker logs for any environment-related errors
3. Try rebuilding the Docker containers with `./run_docker.sh --rebuild`
4. Ensure that the `.env` file is in the correct location (project root)
