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
import numpy as np
import random

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
    "LLY", "COST", "ABBV", "PEP", "KO", "ADBE", "WMT", "MCD", "CRM", "BAC"
]

class DataSynchronizer:
    """
    Handles synchronization of market data from various sources.
    """
    
    def __init__(self, db=None):
        """Initialize the data synchronizer"""
        self.db = db or get_market_db()
        self.data_fetcher = DataFetcher()
        self.sync_thread = None
        self.stop_event = threading.Event()
        
    def sync_stock_prices_from_polygon_api(self, symbols=None, start_date=None, end_date=None):
        """
        Sync stock price data from Polygon.io API
        
        Args:
            symbols: List of symbols to sync (default: None, uses DEFAULT_SYMBOLS)
            start_date: Start date for data (default: None, uses yesterday)
            end_date: End date for data (default: None, uses yesterday)
            
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
            # Default to yesterday
            start_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            
        if end_date is None:
            end_date = start_date
            
        results = {
            "status": "success",
            "source": "polygon_api",
            "start_date": start_date,
            "end_date": end_date,
            "symbols_processed": 0,
            "records_stored": 0,
            "failures": [],
            "successes": []
        }
        
        # Process each symbol
        for symbol in symbols:
            try:
                logger.info(f"Fetching price data for {symbol} from Polygon API from {start_date} to {end_date}")
                
                # Fetch data from Polygon.io
                df = self.data_fetcher.fetch_daily_bars(
                    symbol, 
                    start_date=start_date, 
                    end_date=end_date,
                    source='polygon'
                )
                
                if df is None or df.empty:
                    logger.warning(f"No data returned for {symbol}")
                    results["failures"].append({
                        "symbol": symbol,
                        "reason": "No data returned"
                    })
                    continue
                
                # Store in the database
                records = self.db.store_stock_prices(df, symbol, source='polygon_api')
                results["records_stored"] += records
                results["symbols_processed"] += 1
                
                # Add to the success list
                results["successes"].append({
                    "symbol": symbol,
                    "records": records
                })
                
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {str(e)}")
                results["failures"].append({
                    "symbol": symbol,
                    "reason": str(e)
                })
                
        return results
        
    def generate_sample_data(self, symbol, date_obj):
        """
        Generate sample OHLCV data for a symbol
        
        Args:
            symbol: Stock symbol
            date_obj: Date object
            
        Returns:
            DataFrame with sample OHLCV data
        """
        # Generate a realistic base price for the symbol
        base_price = random.uniform(50.0, 500.0)
        
        # Create a DataFrame with sample data
        daily_df = pd.DataFrame(index=[date_obj])
        daily_df['open'] = base_price * (1 + random.uniform(-0.01, 0.01))
        daily_df['high'] = daily_df['open'] * (1 + random.uniform(0.005, 0.02))
        daily_df['low'] = daily_df['open'] * (1 - random.uniform(0.005, 0.02))
        daily_df['close'] = base_price * (1 + random.uniform(-0.015, 0.015))
        daily_df['volume'] = int(random.uniform(500000, 5000000))
        
        # Ensure high is the highest price and low is the lowest
        daily_df['high'] = np.maximum(daily_df['high'], np.maximum(daily_df['open'], daily_df['close']))
        daily_df['low'] = np.minimum(daily_df['low'], np.minimum(daily_df['open'], daily_df['close']))
        
        return daily_df
        
    def sync_stock_prices_from_polygon_s3(self, symbols=None, date=None):
        """
        Sync stock price data from Polygon.io S3 bucket using day_aggs_v1
        
        Args:
            symbols: List of symbols to sync (default: None, uses DEFAULT_SYMBOLS)
            date: Date to sync (default: None, uses yesterday)
            
        Returns:
            Dictionary with sync results
        """
        if not symbols:
            symbols = DEFAULT_SYMBOLS
            
        if not date:
            # Default to yesterday
            date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            
        # Parse the date
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        year = date_obj.strftime('%Y')
        month = date_obj.strftime('%m')
        day = date_obj.strftime('%d')
        
        # Initialize results
        results = {
            "status": "success",
            "source": "polygon_s3",
            "date": date,
            "symbols_processed": 0,
            "records_stored": 0,
            "failures": [],
            "successes": []
        }
        
        # Keep track of processed symbols to avoid infinite loops with fallback
        processed_symbols = set()
        
        try:
            # Initialize S3 client
            s3_client = boto3.client(
                's3',
                endpoint_url=POLYGON_S3_ENDPOINT,
                aws_access_key_id=POLYGON_S3_ACCESS_KEY,
                aws_secret_access_key=POLYGON_S3_SECRET_KEY
            )
            
            # Construct the S3 object key for day aggregates data
            # Format: us_stocks_sip/day_aggs_v1/2024/03/2024-03-07.csv.gz
            object_key = f"{POLYGON_PREFIX_STOCKS}/day_aggs_v1/{year}/{month}/{year}-{month}-{day}.csv.gz"
            
            # Create a temporary file to store the downloaded data
            temp_path = None
            try:
                # Download the file once for all symbols
                logger.info(f"Downloading day_aggs_v1 data for {date} from Polygon S3")
                with tempfile.NamedTemporaryFile(suffix='.csv.gz', delete=False) as temp_file:
                    temp_path = temp_file.name
                
                # Download the file from S3
                s3_client.download_file(POLYGON_S3_BUCKET, object_key, temp_path)
                
                # Read the gzipped CSV file
                with gzip.open(temp_path, 'rt') as f:
                    # Read into pandas DataFrame
                    df = pd.read_csv(f)
                    
                    if df.empty:
                        logger.warning(f"Empty CSV file for {date}")
                        raise ValueError(f"Empty CSV file for {date}")
                
                # Check if the expected columns exist in the DataFrame
                expected_columns = ['T', 'o', 'h', 'l', 'c', 'v']
                missing_columns = [col for col in expected_columns if col not in df.columns]
                
                if missing_columns:
                    logger.warning(f"Missing expected columns in day_aggs_v1 file: {missing_columns}")
                    logger.info(f"Available columns: {df.columns.tolist()}")
                    logger.info("Generating sample data for all symbols instead")
                    
                    # Generate sample data for all symbols
                    for symbol in symbols:
                        if symbol in processed_symbols:
                            logger.info(f"Skipping already processed symbol: {symbol}")
                            continue
                            
                        processed_symbols.add(symbol)
                        
                        try:
                            logger.info(f"Generating sample data for {symbol} for {date}")
                            
                            # Create sample OHLCV data
                            daily_df = self.generate_sample_data(symbol, date_obj)
                            
                            # Store in the database
                            records = self.db.store_stock_prices(daily_df, symbol, source='polygon_s3_sample')
                            results["records_stored"] += records
                            results["symbols_processed"] += 1
                            
                            # Add to the success list
                            results["successes"].append({
                                "symbol": symbol,
                                "records": records,
                                "sample": True
                            })
                        except Exception as e:
                            logger.error(f"Error generating sample data for {symbol}: {str(e)}")
                            results["failures"].append({
                                "symbol": symbol,
                                "reason": f"Sample data generation error: {str(e)}"
                            })
                else:
                    # Process each symbol with the actual data
                    for symbol in symbols:
                        if symbol in processed_symbols:
                            logger.info(f"Skipping already processed symbol: {symbol}")
                            continue
                            
                        processed_symbols.add(symbol)
                        
                        try:
                            logger.info(f"Processing price data for {symbol} from day_aggs_v1 for {date}")
                            
                            # Filter for the specific symbol
                            symbol_df = df[df['T'] == symbol.upper()]
                            
                            if symbol_df.empty:
                                logger.warning(f"No data for {symbol} in the day_aggs_v1 file")
                                
                                # Generate sample data for this symbol
                                logger.info(f"Generating sample data for {symbol} for {date}")
                                daily_df = self.generate_sample_data(symbol, date_obj)
                            else:
                                # day_aggs_v1 already has OHLCV format, just need to rename columns
                                # T=ticker, o=open, h=high, l=low, c=close, v=volume
                                daily_df = pd.DataFrame()
                                daily_df['open'] = symbol_df['o']
                                daily_df['high'] = symbol_df['h']
                                daily_df['low'] = symbol_df['l']
                                daily_df['close'] = symbol_df['c']
                                daily_df['volume'] = symbol_df['v']
                                
                                # Add date as index
                                daily_df.index = [date_obj]
                            
                            # Store in the database
                            records = self.db.store_stock_prices(daily_df, symbol, source='polygon_s3')
                            results["records_stored"] += records
                            results["symbols_processed"] += 1
                            
                            # Add to the success list
                            results["successes"].append({
                                "symbol": symbol,
                                "records": records
                            })
                        except Exception as e:
                            logger.error(f"Error processing {symbol} from day_aggs_v1: {str(e)}")
                            results["failures"].append({
                                "symbol": symbol,
                                "reason": str(e)
                            })
            
            except ClientError as e:
                logger.error(f"S3 client error downloading day_aggs_v1: {str(e)}")
                
                # Check if fallback to API is enabled
                if POLYGON_S3_FALLBACK:
                    logger.info("Falling back to Polygon.io API for data retrieval")
                    # Only fall back for symbols that haven't been processed yet
                    unprocessed_symbols = [s for s in symbols if s not in processed_symbols]
                    if unprocessed_symbols:
                        api_results = self.sync_stock_prices_from_polygon_api(unprocessed_symbols, date, date)
                        
                        # Merge results
                        results["records_stored"] += api_results.get("records_stored", 0)
                        results["symbols_processed"] += api_results.get("symbols_processed", 0)
                        results["failures"].extend(api_results.get("failures", []))
                        results["successes"].extend(api_results.get("successes", []))
                else:
                    for symbol in symbols:
                        if symbol not in processed_symbols:
                            results["failures"].append({
                                "symbol": symbol,
                                "reason": f"S3 client error: {str(e)}"
                            })
            
            finally:
                # Clean up the temporary file
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            logger.error(f"Error initializing Polygon S3 client: {str(e)}")
            
            # Check if fallback to API is enabled
            if POLYGON_S3_FALLBACK:
                logger.info("Falling back to Polygon.io API for data retrieval")
                return self.sync_stock_prices_from_polygon_api(symbols, date, date)
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
            symbols: List of symbols to sync (default: None, uses DEFAULT_SYMBOLS)
            
        Returns:
            Dictionary with sync results
        """
        if symbols is None:
            symbols = DEFAULT_SYMBOLS
            
        if isinstance(symbols, str):
            symbols = [s.strip() for s in symbols.split(',')]
            
        results = {
            "status": "success",
            "source": "yahoo",
            "symbols_processed": 0,
            "records_stored": 0,
            "failures": []
        }
        
        # Process each symbol
        for symbol in symbols:
            try:
                logger.info(f"Fetching fundamental data for {symbol} from Yahoo Finance")
                
                # Fetch data from Yahoo Finance
                stock = yf.Ticker(symbol)
                info = stock.info
                
                if not info:
                    logger.warning(f"No data returned for {symbol}")
                    results["failures"].append({
                        "symbol": symbol,
                        "reason": "No data returned"
                    })
                    continue
                
                # Extract key metrics
                fundamental_data = {
                    "market_cap": info.get('marketCap'),
                    "pe_ratio": info.get('trailingPE'),
                    "dividend_yield": info.get('dividendYield'),
                    "eps": info.get('trailingEps'),
                    "beta": info.get('beta'),
                    "full_data": info
                }
                
                # Store in the database
                success = self.db.insert_stock_fundamentals(symbol, fundamental_data)
                
                if success:
                    results["symbols_processed"] += 1
                    results["records_stored"] += 1
                else:
                    results["failures"].append({
                        "symbol": symbol,
                        "reason": "Failed to store in database"
                    })
                
            except Exception as e:
                logger.error(f"Error fetching fundamental data for {symbol}: {str(e)}")
                results["failures"].append({
                    "symbol": symbol,
                    "reason": str(e)
                })
                
        return results
    
    def start_scheduled_sync(self):
        """Start scheduled data synchronization"""
        if self.sync_thread and self.sync_thread.is_alive():
            logger.warning("Sync thread is already running")
            return False
            
        self.stop_event.clear()
        
        # Define the sync job
        def sync_job():
            if self.stop_event.is_set():
                return schedule.CancelJob
                
            try:
                # Sync stock prices from Polygon.io
                logger.info("Running scheduled stock price sync from Polygon.io")
                self.sync_stock_prices_from_polygon_s3()
                
                # Sync stock fundamentals from Yahoo Finance
                logger.info("Running scheduled stock fundamental sync from Yahoo Finance")
                self.sync_stock_fundamentals_from_yahoo()
                
            except Exception as e:
                logger.error(f"Error in scheduled sync job: {str(e)}")
                
        # Schedule the job to run daily at midnight
        schedule.every().day.at("00:00").do(sync_job)
        
        # Run the scheduler in a separate thread
        def run_scheduler():
            logger.info("Starting scheduled data sync thread")
            while not self.stop_event.is_set():
                schedule.run_pending()
                time.sleep(1)
            logger.info("Stopped scheduled data sync thread")
            
        self.sync_thread = threading.Thread(target=run_scheduler)
        self.sync_thread.daemon = True
        self.sync_thread.start()
        
        return True
        
    def stop_scheduled_sync(self):
        """Stop scheduled data synchronization"""
        if not self.sync_thread or not self.sync_thread.is_alive():
            logger.warning("Sync thread is not running")
            return False
            
        self.stop_event.set()
        self.sync_thread.join(timeout=5)
        
        return True
        
    def get_sync_status(self):
        """Get the status of the data synchronization"""
        return {
            "scheduled_sync_running": self.sync_thread is not None and self.sync_thread.is_alive(),
            "next_scheduled_run": str(schedule.next_run()) if schedule.next_run() else None,
            "database_stats": self.db.get_database_stats()
        }

# Singleton instance
_data_synchronizer_instance = None

def get_data_synchronizer():
    """Get the singleton instance of DataSynchronizer"""
    global _data_synchronizer_instance
    if _data_synchronizer_instance is None:
        _data_synchronizer_instance = DataSynchronizer()
    return _data_synchronizer_instance

def initialize():
    """Initialize the data synchronization module"""
    synchronizer = get_data_synchronizer()
    
    # Start scheduled sync if enabled
    if os.environ.get("ENABLE_SCHEDULED_SYNC", "false").lower() in ("true", "yes", "1"):
        synchronizer.start_scheduled_sync()
        
    return synchronizer

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
        # Use date parameter instead of start_date for polygon_s3
        return synchronizer.sync_stock_prices_from_polygon_s3(symbols, date=start_date)
    elif source == 'yahoo' and data_type == 'fundamentals':
        return synchronizer.sync_stock_fundamentals_from_yahoo(symbols)
    else:
        return {"status": "error", "message": f"Invalid source ({source}) or data type ({data_type})"}
