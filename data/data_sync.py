"""
Data synchronization module for fetching and storing market data.
Provides scheduled tasks to fetch data from Polygon.io and other sources.
"""
import os
import logging
import time
import threading
import schedule
import pandas as pd
import requests
import boto3
from botocore.exceptions import ClientError
import io
import zipfile
import gzip
import tempfile
from datetime import datetime, timedelta
import json
import yfinance as yf

from data.database import get_market_db
from data.data_fetcher import DataFetcher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API keys and configuration
POLYGON_API_KEY = os.environ.get("POLYGON_API_KEY", "")

# Polygon.io S3 credentials
POLYGON_S3_ACCESS_KEY = os.environ.get("POLYGON_S3_ACCESS_KEY", POLYGON_API_KEY)
POLYGON_S3_SECRET_KEY = os.environ.get("POLYGON_S3_SECRET_KEY", POLYGON_API_KEY)
POLYGON_S3_FALLBACK = os.environ.get("POLYGON_S3_FALLBACK", "true").lower() in ("true", "yes", "1")

# Polygon.io S3 endpoint and bucket configuration
POLYGON_S3_ENDPOINT = "https://files.polygon.io"
POLYGON_S3_BUCKET = "flatfiles"

# Polygon.io data prefixes for different data types
POLYGON_PREFIX_STOCKS = "us_stocks_sip"
POLYGON_PREFIX_OPTIONS = "us_options_opra"
POLYGON_PREFIX_INDICES = "us_indices"
POLYGON_PREFIX_FOREX = "global_forex"
POLYGON_PREFIX_CRYPTO = "global_crypto"

# Default symbols (S&P 500 subset)
DEFAULT_SYMBOLS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "GOOG", "TSLA", "BRK.B", "UNH",
    "JPM", "XOM", "JNJ", "V", "PG", "MA", "HD", "CVX", "AVGO", "MRK",
    "LLY", "COST", "ABBV", "PEP", "KO", "ADBE", "WMT", "MCD", "CRM", "BAC",
    "TMO", "CSCO", "ACN", "ABT", "CMCSA", "AMD", "DHR", "NFLX", "VZ", "NEE",
    "INTC", "PFE", "ORCL", "PM", "TXN", "COP", "INTU", "IBM", "CAT", "QCOM"
]

class DataSynchronizer:
    """
    Handles data synchronization from various sources to the local database.
    Provides methods for both on-demand and scheduled data fetching.
    """
    
    def __init__(self):
        """Initialize the data synchronizer"""
        self.db = get_market_db()
        self.data_fetcher = DataFetcher()
        self.scheduler_thread = None
        self.is_running = False
        logger.info("Data synchronizer initialized")
    
    def sync_stock_prices_from_polygon_api(self, symbols=None, start_date=None, end_date=None):
        """
        Sync stock price data from Polygon.io API
        
        Args:
            symbols: List of stock symbols (defaults to DEFAULT_SYMBOLS)
            start_date: Start date (defaults to 7 days ago)
            end_date: End date (defaults to yesterday)
            
        Returns:
            Dictionary with sync results
        """
        if not POLYGON_API_KEY:
            logger.error("Polygon API key not found")
            return {"status": "error", "message": "Polygon API key not found"}
        
        if symbols is None:
            symbols = DEFAULT_SYMBOLS
        
        if isinstance(symbols, str):
            symbols = [s.strip() for s in symbols.split(',')]
        
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        
        if end_date is None:
            end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        results = {
            "status": "success",
            "symbols_processed": 0,
            "records_stored": 0,
            "failures": []
        }
        
        for symbol in symbols:
            try:
                logger.info(f"Fetching price data for {symbol} from Polygon API")
                df = self.data_fetcher.fetch_stock_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    source="polygon"
                )
                
                if df is not None and not df.empty:
                    records = self.db.store_stock_prices(df, symbol, source='polygon')
                    results["records_stored"] += records
                    results["symbols_processed"] += 1
                else:
                    logger.warning(f"No data returned from Polygon API for {symbol}")
                    results["failures"].append({
                        "symbol": symbol,
                        "reason": "No data returned from API"
                    })
            except Exception as e:
                logger.error(f"Error syncing {symbol} from Polygon API: {str(e)}")
                results["failures"].append({
                    "symbol": symbol,
                    "reason": str(e)
                })
        
        # Log the sync activity
        self.db.log_data_sync(
            source='polygon_api',
            data_type='prices',
            start_date=start_date,
            end_date=end_date,
            symbols=','.join(symbols),
            status='success' if not results["failures"] else 'partial',
            records_processed=results["records_stored"],
            error_message=json.dumps(results["failures"]) if results["failures"] else None
        )
        
        logger.info(f"Polygon API sync completed: {results['symbols_processed']} symbols, {results['records_stored']} records")
        return results
    
    def sync_stock_prices_from_polygon_s3(self, symbols=None, date=None):
        """
        Sync stock price data from Polygon.io S3 bucket using Boto3
        
        Args:
            symbols: List of stock symbols (defaults to DEFAULT_SYMBOLS)
            date: Specific date to sync (defaults to yesterday)
            
        Returns:
            Dictionary with sync results
        """
        if not POLYGON_API_KEY:
            logger.error("Polygon API key not found")
            return {"status": "error", "message": "Polygon API key not found"}
            
        if symbols is None:
            symbols = DEFAULT_SYMBOLS
        
        if isinstance(symbols, str):
            symbols = [s.strip() for s in symbols.split(',')]
        
        if date is None:
            # Default to yesterday
            date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Format date components for S3 path
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        year = date_obj.strftime('%Y')
        month = date_obj.strftime('%m')
        day = date_obj.strftime('%d')
        
        results = {
            "status": "success",
            "symbols_processed": 0,
            "records_stored": 0,
            "failures": []
        }
        
        try:
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
            
            # Process each symbol
            for symbol in symbols:
                try:
                    logger.info(f"Fetching price data for {symbol} from Polygon S3 for {date}")
                    
                    # Construct the S3 object key for trades data
                    # Format: us_stocks_sip/trades_v1/2024/03/2024-03-07.csv.gz
                    # Note: Polygon.io now provides daily files, not per-symbol files
                    object_key = f"{POLYGON_PREFIX_STOCKS}/trades_v1/{year}/{month}/{year}-{month}-{day}.csv.gz"
                    
                    # Create a temporary file to store the downloaded data
                    with tempfile.NamedTemporaryFile(suffix='.csv.gz', delete=False) as temp_file:
                        temp_path = temp_file.name
                    
                    try:
                        # Download the file from S3
                        s3_client.download_file(POLYGON_S3_BUCKET, object_key, temp_path)
                        
                        # Read the gzipped CSV file
                        with gzip.open(temp_path, 'rt') as f:
                            # Read into pandas DataFrame
                            df = pd.read_csv(f)
                            
                            if df.empty:
                                logger.warning(f"Empty CSV file for {date}")
                                continue
                            
                            # Filter for the specific symbol
                            symbol_df = df[df['symbol'] == symbol.upper()]
                            
                            if symbol_df.empty:
                                logger.warning(f"No data for {symbol} in the file")
                                continue
                            
                            # Process the data to create OHLCV
                            # Convert timestamp to datetime (Polygon uses nanoseconds)
                            symbol_df['timestamp'] = pd.to_datetime(symbol_df['timestamp_nanoseconds'], unit='ns')
                            symbol_df.set_index('timestamp', inplace=True)
                            
                            # Resample to daily OHLCV
                            daily_df = pd.DataFrame()
                            daily_df['open'] = symbol_df['price'].resample('D').first()
                            daily_df['high'] = symbol_df['price'].resample('D').max()
                            daily_df['low'] = symbol_df['price'].resample('D').min()
                            daily_df['close'] = symbol_df['price'].resample('D').last()
                            daily_df['volume'] = symbol_df['size'].resample('D').sum()
                            
                            # Store in the database
                            records = self.db.store_stock_prices(daily_df, symbol, source='polygon_s3')
                            results["records_stored"] += records
                            results["symbols_processed"] += 1
                    
                    except ClientError as e:
                        error_code = e.response.get('Error', {}).get('Code')
                        if error_code == '404' or error_code == 'NoSuchKey':
                            logger.warning(f"Data file for {date} not found in S3")
                        elif error_code == '403' or error_code == 'AccessDenied':
                            logger.error(f"Access denied to Polygon S3: {str(e)}")
                            # This is likely an authentication issue, so try fallback if enabled
                            if POLYGON_S3_FALLBACK:
                                logger.info(f"Falling back to Polygon.io API for {symbol}")
                                # Fall back to API for this specific symbol
                                api_result = self.sync_stock_prices_from_polygon_api(
                                    symbols=[symbol], 
                                    start_date=date, 
                                    end_date=date
                                )
                                if api_result.get("status") == "success" and api_result.get("symbols_processed", 0) > 0:
                                    results["symbols_processed"] += api_result.get("symbols_processed", 0)
                                    results["records_stored"] += api_result.get("records_stored", 0)
                                    continue
                            
                            # If fallback is disabled or fallback failed, return error
                            if not POLYGON_S3_FALLBACK:
                                return {
                                    "status": "error",
                                    "message": f"Access denied to Polygon S3: {str(e)}"
                                }
                        else:
                            logger.error(f"S3 error for {symbol}: {str(e)}")
                        
                        results["failures"].append({
                            "symbol": symbol,
                            "reason": f"S3 error: {str(e)}"
                        })
                    
                    finally:
                        # Clean up the temporary file
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                
                except Exception as e:
                    logger.error(f"Error processing {symbol} from Polygon S3: {str(e)}")
                    results["failures"].append({
                        "symbol": symbol,
                        "reason": str(e)
                    })
        
        except Exception as e:
            logger.error(f"Error initializing Polygon S3 client: {str(e)}")
            
            # Check if fallback to API is enabled
            if POLYGON_S3_FALLBACK:
                logger.info("Falling back to Polygon.io API for data retrieval")
                return self.sync_stock_prices_from_polygon_api(symbols, start_date=date, end_date=date)
            else:
                return {
                    "status": "error",
                    "message": f"Failed to initialize S3 client: {str(e)}"
                }
            
        # Log the sync activity
        self.db.log_data_sync(
            source='polygon_s3',
            data_type='prices',
            start_date=date,
            end_date=date,
            symbols=','.join(symbols),
            status='success' if not results["failures"] else 'partial',
            records_processed=results["records_stored"],
            error_message=json.dumps(results["failures"]) if results["failures"] else None
        )
        
        logger.info(f"Polygon S3 sync completed: {results['symbols_processed']} symbols, {results['records_stored']} records")
        return results
    
    def sync_stock_fundamentals_from_yahoo(self, symbols=None):
        """
        Sync stock fundamental data from Yahoo Finance
        
        Args:
            symbols: List of stock symbols (defaults to DEFAULT_SYMBOLS)
            
        Returns:
            Dictionary with sync results
        """
        if symbols is None:
            symbols = DEFAULT_SYMBOLS
        
        if isinstance(symbols, str):
            symbols = [s.strip() for s in symbols.split(',')]
        
        today = datetime.now().strftime('%Y-%m-%d')
        
        results = {
            "status": "success",
            "symbols_processed": 0,
            "records_stored": 0,
            "failures": []
        }
        
        for symbol in symbols:
            try:
                logger.info(f"Fetching fundamental data for {symbol} from Yahoo Finance")
                
                # Get stock data from Yahoo Finance
                stock = yf.Ticker(symbol)
                info = stock.info
                
                if not info:
                    logger.warning(f"No fundamental data returned for {symbol}")
                    results["failures"].append({
                        "symbol": symbol,
                        "reason": "No data returned from Yahoo Finance"
                    })
                    continue
                
                # Store in the database
                if self.db.store_stock_fundamentals(symbol, info, date=today, source='yahoo'):
                    results["records_stored"] += 1
                    results["symbols_processed"] += 1
                else:
                    results["failures"].append({
                        "symbol": symbol,
                        "reason": "Failed to store in database"
                    })
            
            except Exception as e:
                logger.error(f"Error syncing fundamentals for {symbol}: {str(e)}")
                results["failures"].append({
                    "symbol": symbol,
                    "reason": str(e)
                })
        
        # Log the sync activity
        self.db.log_data_sync(
            source='yahoo',
            data_type='fundamentals',
            start_date=today,
            end_date=today,
            symbols=','.join(symbols),
            status='success' if not results["failures"] else 'partial',
            records_processed=results["records_stored"],
            error_message=json.dumps(results["failures"]) if results["failures"] else None
        )
        
        logger.info(f"Yahoo fundamentals sync completed: {results['symbols_processed']} symbols, {results['records_stored']} records")
        return results
    
    def run_scheduled_sync(self):
        """Run the scheduled data synchronization tasks"""
        logger.info("Running scheduled data synchronization")
        
        # Try to use Polygon S3 first
        try:
            yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            s3_results = self.sync_stock_prices_from_polygon_s3(date=yesterday)
            
            if s3_results["symbols_processed"] == 0:
                # If S3 sync failed, fall back to API
                logger.info("S3 sync failed, falling back to Polygon API")
                self.sync_stock_prices_from_polygon_api(
                    start_date=yesterday,
                    end_date=yesterday
                )
        except Exception as e:
            logger.error(f"Error in S3 sync, falling back to API: {str(e)}")
            # Fall back to API
            self.sync_stock_prices_from_polygon_api(
                start_date=(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
                end_date=(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            )
        
        # Sync fundamentals weekly (on Monday)
        if datetime.now().weekday() == 0:  # Monday
            self.sync_stock_fundamentals_from_yahoo()
    
    def start_scheduler(self):
        """Start the scheduler for periodic data synchronization"""
        if self.is_running:
            logger.warning("Scheduler is already running")
            return False
        
        # Schedule the sync to run every hour
        schedule.every().hour.do(self.run_scheduled_sync)
        
        # Also run at market close (4:30 PM ET)
        schedule.every().day.at("16:30").do(self.run_scheduled_sync)
        
        # Run the scheduler in a separate thread
        def run_scheduler():
            self.is_running = True
            logger.info("Scheduler started")
            
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        
        self.scheduler_thread = threading.Thread(target=run_scheduler)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        
        logger.info("Data sync scheduler started")
        return True
    
    def stop_scheduler(self):
        """Stop the scheduler"""
        if not self.is_running:
            logger.warning("Scheduler is not running")
            return False
        
        self.is_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        logger.info("Data sync scheduler stopped")
        return True

# Singleton instance
_data_sync_instance = None

def get_data_synchronizer():
    """Get the singleton instance of DataSynchronizer"""
    global _data_sync_instance
    if _data_sync_instance is None:
        _data_sync_instance = DataSynchronizer()
    return _data_sync_instance

def start_data_sync_scheduler():
    """Start the data synchronization scheduler"""
    synchronizer = get_data_synchronizer()
    return synchronizer.start_scheduler()

def stop_data_sync_scheduler():
    """Stop the data synchronization scheduler"""
    synchronizer = get_data_synchronizer()
    return synchronizer.stop_scheduler()

def run_manual_sync(source='polygon_api', data_type='prices', symbols=None, start_date=None, end_date=None):
    """
    Run a manual data synchronization
    
    Args:
        source: Data source ('polygon_api', 'polygon_s3', 'yahoo')
        data_type: Type of data ('prices', 'fundamentals')
        symbols: List of symbols to sync
        start_date: Start date for price data
        end_date: End date for price data
        
    Returns:
        Dictionary with sync results
    """
    synchronizer = get_data_synchronizer()
    
    if source == 'polygon_api' and data_type == 'prices':
        return synchronizer.sync_stock_prices_from_polygon_api(symbols, start_date, end_date)
    elif source == 'polygon_s3' and data_type == 'prices':
        return synchronizer.sync_stock_prices_from_polygon_s3(symbols, start_date)
    elif source == 'yahoo' and data_type == 'fundamentals':
        return synchronizer.sync_stock_fundamentals_from_yahoo(symbols)
    else:
        return {"status": "error", "message": f"Invalid source ({source}) or data type ({data_type})"}

# Start the scheduler when the module is imported
if __name__ != "__main__":
    # Don't start automatically when imported for testing
    pass
else:
    # Start when run directly
    start_data_sync_scheduler()
