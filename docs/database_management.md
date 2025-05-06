# Database Management

The Shrek Trading Platform includes a comprehensive database management system for storing and retrieving market data. This document explains how to use the database management features.

## Database Management UI

The platform includes a dedicated "Database Management" tab in the UI for refreshing market data and managing the database.

### Features

- **Multiple Data Sources**: Choose between Polygon.io S3, Polygon.io API, or Yahoo Finance
- **Symbol Selection**: Refresh specific symbols or use the default universe
- **Historical Data Range**: Control how many days of historical data to fetch
- **Database Reset**: Option to reset the database before refreshing
- **Status Monitoring**: View detailed status and logs of database operations

### Using the Database Management UI

1. Navigate to the "Database Management" tab in the Gradio UI
2. Select your preferred data source:
   - `polygon_s3`: Polygon.io S3 (fastest, requires S3 credentials)
   - `polygon_api`: Polygon.io API (requires API key)
   - `yahoo`: Yahoo Finance (for fundamental data)
3. Enter specific symbols (comma-separated) or leave empty for default universe
4. Set the number of days of historical data to fetch
5. Choose whether to reset the database (use with caution)
6. Click "Refresh Database"
7. Monitor the refresh status and logs
8. Click "Get Database Status" to view the current database state

## Data Sources

### Polygon.io S3

The platform can fetch historical market data from Polygon.io's S3 bucket, which is significantly faster than using the API.

**Configuration**:
- `POLYGON_S3_ACCESS_KEY`: Your Polygon.io S3 access key
- `POLYGON_S3_SECRET_KEY`: Your Polygon.io S3 secret key
- `POLYGON_S3_FALLBACK`: Set to "true" to fallback to API if S3 fails

### Polygon.io API

If S3 access fails or is not configured, the platform can fetch data from the Polygon.io API.

**Configuration**:
- `POLYGON_API_KEY`: Your Polygon.io API key

### Yahoo Finance

The platform uses Yahoo Finance for fundamental data.

## Database Structure

The market database uses SQLite with thread-safe connections. Key tables include:

- `stock_prices`: Historical price data for stocks
- `stock_fundamentals`: Fundamental data for stocks
- `data_sync_log`: Log of data synchronization activities

## Thread Safety

The database implementation uses thread-local storage for SQLite connections to ensure thread safety:

```python
def _get_connection(self):
    """Get a thread-local database connection"""
    if not hasattr(self._thread_local, 'conn') or self._thread_local.conn is None:
        self._thread_local.conn = sqlite3.connect(self.db_path)
        # Enable foreign keys
        self._thread_local.conn.execute("PRAGMA foreign_keys = ON")
    return self._thread_local.conn
```

## Database Reset

The database can be reset through the UI or programmatically:

```python
# Reset the database
db = get_market_db()
db.reset_tables()
```

This will delete all data from all tables except system tables.

## Fallback Mechanism

The platform includes a fallback mechanism to handle data source failures:

1. Try to fetch data from Polygon.io S3
2. If S3 fails or returns no data, fallback to Polygon.io API
3. Log all failures and fallbacks for troubleshooting

## API Endpoints

The platform provides API endpoints for database management:

- `POST /database/refresh`: Refresh the database from a specified source
- `GET /database/status`: Get the current database status

## Troubleshooting

If you encounter database issues:

1. Check the database status using the "Get Database Status" button
2. Review the logs for any error messages
3. Try refreshing with a different data source
4. If necessary, reset the database and refresh with a smaller date range
5. Verify that your API keys and S3 credentials are correct

## Best Practices

- Regularly refresh the database to ensure you have the latest market data
- Use Polygon.io S3 for large data refreshes when possible
- Reset the database only when necessary
- Monitor the database size and performance
- Use specific symbols when possible to reduce data volume
