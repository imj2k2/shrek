# Data Sources

The Shrek Trading Platform integrates with multiple data sources to provide comprehensive market data for trading and backtesting. This document explains the available data sources and how they are used.

## Polygon.io

Polygon.io is the primary data source for historical and real-time market data. The platform supports two methods of accessing Polygon.io data:

### Polygon.io S3

The platform can fetch historical market data directly from Polygon.io's S3 bucket, which is significantly faster and more efficient than using the API, especially for large datasets.

**Configuration**:
- `POLYGON_S3_ACCESS_KEY`: Your Polygon.io S3 access key
- `POLYGON_S3_SECRET_KEY`: Your Polygon.io S3 secret key

**Implementation**:
```python
# Initialize Boto3 session with Polygon.io S3 credentials
session = boto3.Session(
    aws_access_key_id=POLYGON_S3_ACCESS_KEY,
    aws_secret_access_key=POLYGON_S3_SECRET_KEY
)

# Create S3 client with Polygon.io endpoint
from botocore.config import Config
s3_client = session.client(
    's3',
    endpoint_url=POLYGON_S3_ENDPOINT,
    config=Config(signature_version='s3v4')
)
```

### Polygon.io API

If S3 access fails or is not configured, the platform can fetch data from the Polygon.io REST API.

**Configuration**:
- `POLYGON_API_KEY`: Your Polygon.io API key

**Implementation**:
```python
# Make API request to Polygon.io
response = requests.get(
    f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}",
    params={"apiKey": POLYGON_API_KEY}
)
```

### Fallback Mechanism

The platform includes a fallback mechanism to handle data source failures:

```python
# Set fallback behavior
POLYGON_S3_FALLBACK = os.environ.get("POLYGON_S3_FALLBACK", "true").lower() in ("true", "yes", "1")

# Try S3 first
try:
    s3_result = synchronizer.sync_stock_prices_from_polygon_s3(date=end_date)
    
    # If S3 failed or had no results, use API
    if s3_result["symbols_processed"] == 0 and POLYGON_S3_FALLBACK:
        api_result = synchronizer.sync_stock_prices_from_polygon_api(
            start_date=start_date,
            end_date=end_date
        )
except Exception as e:
    if POLYGON_S3_FALLBACK:
        # Fallback to API
        api_result = synchronizer.sync_stock_prices_from_polygon_api(
            start_date=start_date,
            end_date=end_date
        )
```

## Yahoo Finance

The platform uses Yahoo Finance for fundamental data, including:

- Company financials
- Key statistics
- Analyst recommendations
- Earnings data

**Implementation**:
```python
# Fetch fundamental data from Yahoo Finance
def sync_stock_fundamentals_from_yahoo(self, symbols=None):
    # Implementation details...
    pass
```

## Local Database

All fetched data is stored in a local SQLite database for efficient access during trading and backtesting.

**Key Tables**:
- `stock_prices`: Historical price data for stocks
- `stock_fundamentals`: Fundamental data for stocks
- `data_sync_log`: Log of data synchronization activities

## Data Refresh

The platform provides a UI for refreshing data from different sources:

1. Navigate to the "Database Management" tab
2. Select your preferred data source
3. Enter specific symbols or leave empty for default universe
4. Set the number of days of historical data
5. Click "Refresh Database"

## Best Practices

- Use Polygon.io S3 for large data refreshes when possible
- Enable fallback to ensure data availability
- Regularly refresh fundamental data from Yahoo Finance
- Monitor data synchronization logs for any issues
- Use specific symbols when possible to reduce data volume

## Troubleshooting

If you encounter data source issues:

1. Verify your API keys and S3 credentials
2. Check network connectivity to the data sources
3. Review the logs for any error messages
4. Try a different data source if one is unavailable
5. For S3 issues, ensure your credentials have the correct permissions
