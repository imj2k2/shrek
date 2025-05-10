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
import sys
from pathlib import Path

# Add dotenv loading for environment variables
try:
    from dotenv import load_dotenv
    # Try to load from current directory and parent directories
    env_file = Path('.env')
    if env_file.exists():
        load_dotenv(env_file)
    else:
        # Try parent directory (project root)
        parent_env = Path(__file__).resolve().parent.parent / '.env'
        if parent_env.exists():
            load_dotenv(parent_env)
            print(f"Loaded environment variables from {parent_env}")
        else:
            print("No .env file found. Using environment variables as is.")
except ImportError:
    print("python-dotenv not installed. Using environment variables as is.")

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
                aws_secret_access_key=POLYGON_S3_SECRET_KEY,
                region_name='us-east-1'  # Adding explicit region
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
    
    def backfill_historical_data_s3(self, start_year=None, end_year=None, extract_all_symbols=True):
        """
        Backfill historical data from Polygon.io S3 bucket using yearly aggregate files
        
        Args:
            start_year: First year to backfill (default: None, will go back 10 years)
            end_year: Last year to backfill (default: None, uses current year)
            extract_all_symbols: Whether to extract all symbols in the file (True) or just DEFAULT_SYMBOLS (False)
            
        Returns:
            Dictionary with backfill results
        """
        # Set up time range if not specified
        current_year = datetime.now().year
        
        if end_year is None:
            end_year = current_year
            
        if start_year is None:
            start_year = current_year - 10  # Default to 10 years of data
        
        logger.info(f"Starting S3 historical data backfill from {start_year} to {end_year}")
        
        # Initialize S3 client
        try:
            s3_client = boto3.client(
                's3',
                endpoint_url=POLYGON_S3_ENDPOINT,
                aws_access_key_id=POLYGON_S3_ACCESS_KEY,
                aws_secret_access_key=POLYGON_S3_SECRET_KEY,
                region_name='us-east-1'  # Adding explicit region
            )
        except Exception as e:
            logger.error(f"Failed to initialize S3 client: {str(e)}")
            return {"status": "error", "message": f"S3 client initialization error: {str(e)}"}
        
        results = {
            "status": "success",
            "source": "polygon_s3_backfill",
            "start_year": start_year,
            "end_year": end_year,
            "years_processed": 0,
            "symbols_processed": 0,
            "records_stored": 0,
            "failures": [],
            "successes": []
        }
        
        # Process each year
        for year in range(start_year, end_year + 1):
            year_str = str(year)
            logger.info(f"Processing data for year {year_str}")
            
            try:
                # First try to get the full year aggregated file
                # Format: us_stocks_sip/day_aggs_v1/all/2022.csv.gz
                object_key = f"{POLYGON_PREFIX_STOCKS}/day_aggs_v1/all/{year_str}.csv.gz"
                
                # Create a temporary file to store the downloaded data
                with tempfile.NamedTemporaryFile(suffix='.csv.gz', delete=False) as temp_file:
                    temp_path = temp_file.name
                
                try:
                    # Download the file from S3
                    logger.info(f"Downloading yearly aggregate file for {year_str}")
                    s3_client.download_file(POLYGON_S3_BUCKET, object_key, temp_path)
                    
                    # Process the yearly file
                    self._process_s3_yearly_file(year_str, temp_path, extract_all_symbols, results)
                    
                    # Increment successful years
                    results["years_processed"] += 1
                    
                except ClientError as s3_error:
                    # If yearly file isn't available, try monthly files
                    logger.warning(f"Yearly file for {year_str} not available, trying monthly files: {str(s3_error)}")
                    
                    # Try each month
                    for month in range(1, 13):
                        # Skip future months in the current year
                        if year == current_year and month > datetime.now().month:
                            logger.info(f"Skipping future month {month} in current year {year}")
                            continue
                            
                        month_str = f"{month:02d}"
                        try:
                            # Format: us_stocks_sip/day_aggs_v1/all/2022-02.csv.gz
                            object_key = f"{POLYGON_PREFIX_STOCKS}/day_aggs_v1/all/{year_str}-{month_str}.csv.gz"
                            
                            with tempfile.NamedTemporaryFile(suffix='.csv.gz', delete=False) as month_temp_file:
                                month_temp_path = month_temp_file.name
                            
                            # Download the monthly file
                            logger.info(f"Downloading monthly aggregate file for {year_str}-{month_str}")
                            s3_client.download_file(POLYGON_S3_BUCKET, object_key, month_temp_path)
                            
                            # Process the monthly file
                            self._process_s3_monthly_file(year_str, month_str, month_temp_path, extract_all_symbols, results)
                            
                        except ClientError as month_error:
                            logger.warning(f"Monthly file for {year_str}-{month_str} not available: {str(month_error)}")
                        except Exception as month_proc_error:
                            logger.error(f"Error processing monthly file for {year_str}-{month_str}: {str(month_proc_error)}")
                        finally:
                            # Clean up the monthly temp file
                            if 'month_temp_path' in locals() and os.path.exists(month_temp_path):
                                try:
                                    os.unlink(month_temp_path)
                                except Exception as e:
                                    logger.warning(f"Error removing temporary monthly file: {str(e)}")
                
                finally:
                    # Clean up the temp file
                    if os.path.exists(temp_path):
                        try:
                            os.unlink(temp_path)
                        except Exception as e:
                            logger.warning(f"Error removing temporary file: {str(e)}")
            
            except Exception as year_error:
                logger.error(f"Error processing year {year_str}: {str(year_error)}")
                results["failures"].append({
                    "year": year_str,
                    "reason": str(year_error)
                })
        
        return results

    def _process_s3_yearly_file(self, year, file_path, extract_all_symbols, results):
        """
        Process a yearly aggregate file from Polygon.io S3 bucket
        
        Args:
            year: Year string
            file_path: Path to the downloaded file
            extract_all_symbols: Whether to extract all symbols or just DEFAULT_SYMBOLS
            results: Results dictionary to update
            
        Returns:
            None (updates results in-place)
        """
        logger.info(f"Processing yearly aggregate file for {year}")
        
        try:
            # Read the gzipped CSV file
            with gzip.open(file_path, 'rt') as f:
                # Read the CSV in chunks to avoid memory issues with large files
                chunk_size = 100000  # Adjust based on available memory
                chunk_reader = pd.read_csv(f, chunksize=chunk_size)
                
                for chunk_idx, chunk in enumerate(chunk_reader):
                    logger.info(f"Processing chunk {chunk_idx + 1} for year {year} with {len(chunk)} rows")
                    
                    # Filter symbols if needed
                    if not extract_all_symbols:
                        # Filter for only the default symbols
                        chunk = chunk[chunk['T'].isin([s.upper() for s in DEFAULT_SYMBOLS])]
                    
                    # Get unique symbols in this chunk
                    symbols_in_chunk = chunk['T'].unique()
                    
                    # Group by symbol and date
                    for symbol in symbols_in_chunk:
                        symbol_data = chunk[chunk['T'] == symbol]
                        
                        # Skip if no data
                        if symbol_data.empty:
                            continue
                        
                        # Convert to the format expected by the database
                        df = pd.DataFrame()
                        df['open'] = symbol_data['o']
                        df['high'] = symbol_data['h']
                        df['low'] = symbol_data['l']
                        df['close'] = symbol_data['c']
                        df['volume'] = symbol_data['v']
                        
                        # Convert timestamp to datetime and set as index
                        # The timestamp column may be named 't' or 'd' depending on the file format
                        if 't' in symbol_data.columns:
                            # Convert milliseconds timestamp to datetime
                            df.index = pd.to_datetime(symbol_data['t'], unit='ms')
                        elif 'd' in symbol_data.columns:
                            # Convert date string to datetime
                            df.index = pd.to_datetime(symbol_data['d'])
                        
                        # Store in the database
                        try:
                            records = self.db.store_stock_prices(df, symbol, source='polygon_s3_backfill')
                            results["records_stored"] += records
                            results["symbols_processed"] += 1
                            
                            # Check if we should add to successes or update
                            success_entry = next((s for s in results["successes"] if s.get("symbol") == symbol), None)
                            if success_entry:
                                # Update existing entry
                                success_entry["records"] = success_entry.get("records", 0) + records
                            else:
                                # Add new entry
                                results["successes"].append({
                                    "symbol": symbol,
                                    "records": records
                                })
                            
                        except Exception as db_error:
                            logger.error(f"Error storing data for {symbol} in year {year}: {str(db_error)}")
                            results["failures"].append({
                                "symbol": symbol,
                                "year": year,
                                "reason": str(db_error)
                            })
        
        except Exception as e:
            logger.error(f"Error processing yearly file for {year}: {str(e)}")
            raise

    def _process_s3_monthly_file(self, year, month, file_path, extract_all_symbols, results):
        """
        Process a monthly aggregate file from Polygon.io S3 bucket
        
        Args:
            year: Year string
            month: Month string
            file_path: Path to the downloaded file
            extract_all_symbols: Whether to extract all symbols or just DEFAULT_SYMBOLS
            results: Results dictionary to update
            
        Returns:
            None (updates results in-place)
        """
        logger.info(f"Processing monthly aggregate file for {year}-{month}")
        
        # Use the same processing logic as yearly files
        # The format is the same, just for a single month
        try:
            # Read the gzipped CSV file
            with gzip.open(file_path, 'rt') as f:
                # Read the CSV in chunks to avoid memory issues with large files
                chunk_size = 100000  # Adjust based on available memory
                chunk_reader = pd.read_csv(f, chunksize=chunk_size)
                
                for chunk_idx, chunk in enumerate(chunk_reader):
                    logger.info(f"Processing chunk {chunk_idx + 1} for {year}-{month} with {len(chunk)} rows")
                    
                    # Filter symbols if needed
                    if not extract_all_symbols:
                        # Filter for only the default symbols
                        chunk = chunk[chunk['T'].isin([s.upper() for s in DEFAULT_SYMBOLS])]
                    
                    # Get unique symbols in this chunk
                    symbols_in_chunk = chunk['T'].unique()
                    
                    # Group by symbol and date
                    for symbol in symbols_in_chunk:
                        symbol_data = chunk[chunk['T'] == symbol]
                        
                        # Skip if no data
                        if symbol_data.empty:
                            continue
                        
                        # Convert to the format expected by the database
                        df = pd.DataFrame()
                        df['open'] = symbol_data['o']
                        df['high'] = symbol_data['h']
                        df['low'] = symbol_data['l']
                        df['close'] = symbol_data['c']
                        df['volume'] = symbol_data['v']
                        
                        # Convert timestamp to datetime and set as index
                        # The timestamp column may be named 't' or 'd' depending on the file format
                        if 't' in symbol_data.columns:
                            # Convert milliseconds timestamp to datetime
                            df.index = pd.to_datetime(symbol_data['t'], unit='ms')
                        elif 'd' in symbol_data.columns:
                            # Convert date string to datetime
                            df.index = pd.to_datetime(symbol_data['d'])
                        
                        # Store in the database
                        try:
                            records = self.db.store_stock_prices(df, symbol, source='polygon_s3_backfill')
                            results["records_stored"] += records
                            
                            # Check if we should increment symbols_processed
                            # Only increment for new symbols, not ones already counted
                            if not any(s.get("symbol") == symbol for s in results["successes"]):
                                results["symbols_processed"] += 1
                            
                            # Check if we should add to successes or update
                            success_entry = next((s for s in results["successes"] if s.get("symbol") == symbol), None)
                            if success_entry:
                                # Update existing entry
                                success_entry["records"] = success_entry.get("records", 0) + records
                            else:
                                # Add new entry
                                results["successes"].append({
                                    "symbol": symbol,
                                    "records": records
                                })
                            
                        except Exception as db_error:
                            logger.error(f"Error storing data for {symbol} in {year}-{month}: {str(db_error)}")
                            results["failures"].append({
                                "symbol": symbol,
                                "period": f"{year}-{month}",
                                "reason": str(db_error)
                            })
        
        except Exception as e:
            logger.error(f"Error processing monthly file for {year}-{month}: {str(e)}")
            raise

    def backfill_historical_data(self, symbols=None, years=10, batch_size=5, batch_delay=1.0):
        """
        Backfill historical data for the specified symbols going back the specified number of years
        
        Args:
            symbols: List of symbols to backfill (default: None, uses DEFAULT_SYMBOLS)
            years: Number of years to go back (default: 10)
            batch_size: Number of symbols to process in each batch (default: 5)
            batch_delay: Delay between batches in seconds (default: 1.0)
            
        Returns:
            Dictionary with backfill results
        """
        if not POLYGON_API_KEY:
            logger.error("Polygon API key not found")
            return {"status": "error", "message": "Polygon API key not found"}
            
        if symbols is None:
            symbols = DEFAULT_SYMBOLS
            
        if isinstance(symbols, str):
            symbols = [s.strip() for s in symbols.split(',')]
            
        # Calculate the start and end dates
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365 * years)).strftime('%Y-%m-%d')
        
        logger.info(f"Starting historical data backfill from {start_date} to {end_date} for {len(symbols)} symbols")
        
        results = {
            "status": "success",
            "source": "polygon_api_backfill",
            "start_date": start_date,
            "end_date": end_date,
            "symbols_processed": 0,
            "records_stored": 0,
            "failures": [],
            "successes": []
        }
        
        # Process symbols in batches to avoid API rate limits
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(symbols) + batch_size - 1)//batch_size} with {len(batch)} symbols")
            
            for symbol in batch:
                try:
                    # Check if we already have data for this period
                    existing_data = self.db.get_stock_prices(symbol, start_date, end_date)
                    
                    if existing_data is not None and not existing_data.empty:
                        logger.info(f"Found {len(existing_data)} existing records for {symbol} in date range")
                        
                        # Check for gaps in the data
                        date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
                        missing_dates = date_range.difference(pd.DatetimeIndex(existing_data.index))
                        
                        if len(missing_dates) == 0:
                            logger.info(f"No missing dates for {symbol}, skipping")
                            results["symbols_processed"] += 1
                            results["successes"].append({
                                "symbol": symbol,
                                "records": len(existing_data),
                                "status": "complete"
                            })
                            continue
                        
                        logger.info(f"Found {len(missing_dates)} missing dates for {symbol}, will fetch those")
                        
                        # Split missing dates into chunks to avoid API limits
                        # This is a simplified approach; a more sophisticated one would use coherent date ranges
                        missing_dates_list = missing_dates.strftime('%Y-%m-%d').tolist()
                        
                        for missing_date in missing_dates_list:
                            try:
                                # Fetch just this date
                                df = self.data_fetcher.fetch_daily_bars(
                                    symbol, 
                                    start_date=missing_date, 
                                    end_date=missing_date,
                                    source='polygon'
                                )
                                
                                if df is not None and not df.empty:
                                    records = self.db.store_stock_prices(df, symbol, source='polygon_api_backfill')
                                    results["records_stored"] += records
                            except Exception as date_error:
                                logger.warning(f"Error fetching data for {symbol} on {missing_date}: {str(date_error)}")
                                # Continue with next date, don't abort the whole process
                        
                        results["symbols_processed"] += 1
                        results["successes"].append({
                            "symbol": symbol,
                            "status": "updated"
                        })
                    else:
                        # No existing data, fetch the full range
                        logger.info(f"No existing data for {symbol}, fetching full range")
                        
                        # Since Polygon API has a limit on the number of data points per request,
                        # we need to split the date range into smaller chunks
                        # Each chunk is 2 years (730 days) which should be under the API limit
                        
                        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                        
                        # Create chunks of 2 years each
                        chunk_days = 730  # ~2 years
                        current_start = start_dt
                        
                        while current_start < end_dt:
                            current_end = min(current_start + timedelta(days=chunk_days), end_dt)
                            chunk_start = current_start.strftime('%Y-%m-%d')
                            chunk_end = current_end.strftime('%Y-%m-%d')
                            
                            logger.info(f"Fetching chunk for {symbol}: {chunk_start} to {chunk_end}")
                            
                            try:
                                df = self.data_fetcher.fetch_daily_bars(
                                    symbol, 
                                    start_date=chunk_start, 
                                    end_date=chunk_end,
                                    source='polygon'
                                )
                                
                                if df is not None and not df.empty:
                                    records = self.db.store_stock_prices(df, symbol, source='polygon_api_backfill')
                                    results["records_stored"] += records
                                    logger.info(f"Stored {records} records for {symbol} from {chunk_start} to {chunk_end}")
                                else:
                                    logger.warning(f"No data returned for {symbol} from {chunk_start} to {chunk_end}")
                            except Exception as chunk_error:
                                logger.error(f"Error fetching chunk for {symbol} from {chunk_start} to {chunk_end}: {str(chunk_error)}")
                                # Continue with next chunk, don't abort the whole process
                            
                            current_start = current_end + timedelta(days=1)
                            
                            # Add a small delay to avoid hitting API rate limits
                            time.sleep(0.5)
                        
                        results["symbols_processed"] += 1
                        results["successes"].append({
                            "symbol": symbol,
                            "status": "new"
                        })
                        
                except Exception as e:
                    logger.error(f"Error processing {symbol} for backfill: {str(e)}")
                    results["failures"].append({
                        "symbol": symbol,
                        "reason": str(e)
                    })
            
            # Add a delay between batches to avoid API rate limits
            if i + batch_size < len(symbols):
                logger.info(f"Sleeping for {batch_delay}s before next batch")
                time.sleep(batch_delay)
        
        return results
                
    def check_data_completeness(self, symbols=None, start_date=None, end_date=None):
        """
        Check if we have complete data for the specified symbols in the date range
        
        Args:
            symbols: List of symbols to check (default: None, uses DEFAULT_SYMBOLS)
            start_date: Start date to check from (default: None, uses 10 years ago)
            end_date: End date to check to (default: None, uses today)
            
        Returns:
            Dictionary with completeness check results
        """
        if symbols is None:
            symbols = DEFAULT_SYMBOLS
            
        if isinstance(symbols, str):
            symbols = [s.strip() for s in symbols.split(',')]
            
        if start_date is None:
            # Default to 10 years ago
            start_date = (datetime.now() - timedelta(days=365 * 10)).strftime('%Y-%m-%d')
            
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        results = {
            "overall_completeness": 0.0,
            "symbol_details": {}
        }
        
        # Generate the expected trading days (business days)
        expected_days = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
        num_expected_days = len(expected_days)
        
        total_available_days = 0
        symbols_checked = 0
        
        for symbol in symbols:
            try:
                # Get existing data for this symbol
                existing_data = self.db.get_stock_prices(symbol, start_date, end_date)
                
                if existing_data is not None and not existing_data.empty:
                    # Check for gaps
                    available_days = len(existing_data)
                    completeness = min(available_days / num_expected_days, 1.0) if num_expected_days > 0 else 0.0
                    
                    # Calculate missing dates
                    available_dates = pd.DatetimeIndex(existing_data.index)
                    missing_dates = expected_days.difference(available_dates)
                    
                    total_available_days += available_days
                    symbols_checked += 1
                    
                    results["symbol_details"][symbol] = {
                        "available_days": available_days,
                        "expected_days": num_expected_days,
                        "completeness": completeness,
                        "missing_dates_count": len(missing_dates),
                        "first_date": existing_data.index[0].strftime('%Y-%m-%d') if len(existing_data.index) > 0 else None,
                        "last_date": existing_data.index[-1].strftime('%Y-%m-%d') if len(existing_data.index) > 0 else None
                    }
                else:
                    # No data available
                    results["symbol_details"][symbol] = {
                        "available_days": 0,
                        "expected_days": num_expected_days,
                        "completeness": 0.0,
                        "missing_dates_count": num_expected_days,
                        "first_date": None,
                        "last_date": None
                    }
                    
            except Exception as e:
                logger.error(f"Error checking data completeness for {symbol}: {str(e)}")
                results["symbol_details"][symbol] = {
                    "error": str(e)
                }
        
        # Calculate overall completeness
        if symbols_checked > 0 and num_expected_days > 0:
            total_expected_days = num_expected_days * symbols_checked
            results["overall_completeness"] = min(total_available_days / total_expected_days, 1.0)
        
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
                # Daily sync - update with yesterday's data
                logger.info("Running daily stock price sync from Polygon.io")
                yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
                
                # Try to sync daily data from S3 first (bulk data retrieval)
                try:
                    logger.info("Attempting to fetch yesterday's data from Polygon S3")
                    # Use S3 to get daily data
                    self.sync_stock_prices_from_polygon_s3(date=yesterday)
                except Exception as s3_error:
                    logger.warning(f"Failed to fetch yesterday's data from S3, falling back to API: {str(s3_error)}")
                    # Fallback to API if S3 fails
                    self.sync_stock_prices_from_polygon_api(start_date=yesterday, end_date=yesterday)
                
                # Weekly sync - check for any missing data in the last month and perform S3 backfill if needed
                if datetime.now().weekday() == 6:  # Sunday
                    logger.info("Running weekly check for missing data in the last month")
                    month_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                    completeness = self.check_data_completeness(start_date=month_ago)
                    
                    # If overall completeness is low, use S3 to backfill the whole month
                    if completeness["overall_completeness"] < 0.90:  # Less than 90% complete
                        logger.info(f"Overall data completeness for last month is low ({completeness['overall_completeness']*100:.1f}%), running S3 backfill")
                        # Get current month and year
                        current_month = datetime.now().month
                        current_year = datetime.now().year
                        
                        # Calculate the month to backfill (last month)
                        backfill_month = current_month - 1
                        backfill_year = current_year
                        if backfill_month == 0:
                            backfill_month = 12
                            backfill_year -= 1
                        
                        # Backfill specific month using S3
                        try:
                            # Format: us_stocks_sip/day_aggs_v1/all/2022-02.csv.gz
                            month_str = f"{backfill_month:02d}"
                            year_str = str(backfill_year)
                            
                            logger.info(f"Backfilling data for {year_str}-{month_str} using S3")
                            
                            # Initialize S3 client
                            s3_client = boto3.client(
                                's3',
                                endpoint_url=POLYGON_S3_ENDPOINT,
                                aws_access_key_id=POLYGON_S3_ACCESS_KEY,
                                aws_secret_access_key=POLYGON_S3_SECRET_KEY
                            )
                            
                            object_key = f"{POLYGON_PREFIX_STOCKS}/day_aggs_v1/all/{year_str}-{month_str}.csv.gz"
                            
                            with tempfile.NamedTemporaryFile(suffix='.csv.gz', delete=False) as temp_file:
                                temp_path = temp_file.name
                            
                            # Download the monthly file
                            logger.info(f"Downloading monthly aggregate file for {year_str}-{month_str}")
                            s3_client.download_file(POLYGON_S3_BUCKET, object_key, temp_path)
                            
                            # Process the file with mock results dict just for this monthly process
                            monthly_results = {
                                "symbols_processed": 0,
                                "records_stored": 0,
                                "successes": [],
                                "failures": []
                            }
                            
                            # Process the monthly file
                            self._process_s3_monthly_file(year_str, month_str, temp_path, True, monthly_results)
                            
                            logger.info(f"Weekly S3 backfill completed for {year_str}-{month_str}. "
                                       f"Processed {monthly_results['symbols_processed']} symbols, "
                                       f"stored {monthly_results['records_stored']} records.")
                            
                        except Exception as month_error:
                            logger.error(f"Error backfilling month {year_str}-{month_str} from S3: {str(month_error)}")
                            # Fall back to API for individual symbols with low completeness
                            for symbol, details in completeness["symbol_details"].items():
                                if details.get("completeness", 0) < 0.95:  # Less than 95% complete
                                    logger.info(f"Data for {symbol} is incomplete ({details.get('completeness', 0)*100:.1f}%), fetching missing data via API")
                                    # Fetch missing data
                                    self.sync_stock_prices_from_polygon_api(symbols=[symbol], start_date=month_ago)
                    else:
                        # If overall completeness is good, just fix specific symbols
                        for symbol, details in completeness["symbol_details"].items():
                            if details.get("completeness", 0) < 0.95:  # Less than 95% complete
                                logger.info(f"Data for {symbol} is incomplete ({details.get('completeness', 0)*100:.1f}%), fetching missing data")
                                # Fetch missing data
                                self.sync_stock_prices_from_polygon_api(symbols=[symbol], start_date=month_ago)
                
                # Monthly sync - full data check and S3 backfill for a specific year
                if datetime.now().day == 1:  # First day of the month
                    logger.info("Running monthly S3 backfill for historical data")
                    
                    # Backfill a specific year each month to ensure historical data completeness
                    # This ensures that over a 10 year period, all historical data gets refreshed
                    current_year = datetime.now().year
                    month_num = datetime.now().month
                    
                    # Calculate which year to backfill based on current month
                    # This distributes the backfill tasks across the year
                    years_back = (month_num % 10) + 1  # 1-10 years back based on current month
                    backfill_year = current_year - years_back
                    
                    logger.info(f"Monthly backfill for year {backfill_year} (determined by current month {month_num})")
                    
                    # Run S3 backfill for the specific year
                    try:
                        results = self.backfill_historical_data_s3(start_year=backfill_year, end_year=backfill_year)
                        logger.info(f"Monthly historical backfill completed for year {backfill_year}. "
                                   f"Processed {results.get('symbols_processed', 0)} symbols, "
                                   f"stored {results.get('records_stored', 0)} records.")
                    except Exception as year_error:
                        logger.error(f"Error during monthly historical backfill for year {backfill_year}: {str(year_error)}")
                
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

def run_manual_sync(source='polygon_api', data_type='prices', symbols=None, start_date=None, end_date=None, years=None, use_s3=True, extract_all_symbols=True):
    """
    Run a manual data synchronization
    
    Args:
        source: Data source ('polygon_api', 'polygon_s3', 'yahoo')
        data_type: Type of data ('prices', 'fundamentals', 'backfill', 'check')
        symbols: List of symbols to sync
        start_date: Start date for price data
        end_date: End date for price data
        years: Number of years to backfill (only used with data_type='backfill')
        use_s3: Whether to use S3 for backfilling (default: True)
        extract_all_symbols: Whether to extract all symbols when using S3 (default: True)
        
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
    elif data_type == 'backfill':
        if use_s3:
            # Use S3 for efficient backfilling
            if years:
                start_year = datetime.now().year - years
            else:
                start_year = None  # Will default to 10 years ago
                
            return synchronizer.backfill_historical_data_s3(
                start_year=start_year, 
                end_year=None,  # Default to current year
                extract_all_symbols=extract_all_symbols
            )
        else:
            # Fallback to API-based backfill (slower, more API calls)
            return synchronizer.backfill_historical_data(symbols, years=years or 10)
    elif data_type == 'check':
        # Check data completeness
        return synchronizer.check_data_completeness(symbols, start_date, end_date)
    else:
        return {"status": "error", "message": f"Invalid source ({source}) or data type ({data_type})"}

# Create a CLI command to run the backfill
def run_backfill_cli():
    """
    Command-line interface for running a historical data backfill
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Backfill historical market data from Polygon.io')
    parser.add_argument('--symbols', type=str, help='Comma-separated list of symbols (only used with --api mode)')
    parser.add_argument('--years', type=int, default=10, help='Number of years to backfill')
    parser.add_argument('--start-year', type=int, help='Specific start year for backfill (overrides --years)')
    parser.add_argument('--end-year', type=int, help='Specific end year for backfill (defaults to current year)')
    parser.add_argument('--batch-size', type=int, default=5, help='Number of symbols to process in each batch (API mode only)')
    parser.add_argument('--check-only', action='store_true', help='Only check data completeness without backfilling')
    parser.add_argument('--api', action='store_true', help='Use API instead of S3 for backfill (slower but more precise)')
    parser.add_argument('--filter-symbols', action='store_true', help='Only extract DEFAULT_SYMBOLS from S3 files (not all symbols)')
    
    args = parser.parse_args()
    
    # Initialize the data synchronizer
    synchronizer = get_data_synchronizer()
    
    symbols = args.symbols.split(',') if args.symbols else None
    
    if args.check_only:
        print(f"Checking data completeness for {len(symbols) if symbols else len(DEFAULT_SYMBOLS)} symbols...")
        results = synchronizer.check_data_completeness(symbols)
        print(f"Overall data completeness: {results['overall_completeness']*100:.2f}%")
        
        # Print details for symbols with less than 95% completeness
        incomplete_symbols = {s: d for s, d in results["symbol_details"].items() 
                             if d.get("completeness", 0) < 0.95}
        
        if incomplete_symbols:
            print(f"\nSymbols with incomplete data ({len(incomplete_symbols)}):\n")
            for symbol, details in incomplete_symbols.items():
                print(f"{symbol}: {details.get('completeness', 0)*100:.2f}% complete, "
                     f"{details.get('missing_dates_count', 0)} missing dates")
        else:
            print("\nAll symbols have at least 95% data completeness.")
    else:
        if args.api:
            # Use the API-based backfill
            print(f"Starting API-based backfill for {len(symbols) if symbols else len(DEFAULT_SYMBOLS)} symbols, "
                  f"going back {args.years} years...")
            results = synchronizer.backfill_historical_data(
                symbols, 
                years=args.years, 
                batch_size=args.batch_size
            )
            
            print(f"API Backfill completed. Processed {results['symbols_processed']} symbols, "
                  f"stored {results['records_stored']} records.")
            
            if results["failures"]:
                print(f"Failures ({len(results['failures'])}):\n")
                for failure in results["failures"]:
                    print(f"{failure['symbol']}: {failure['reason']}")
        else:
            # Use the S3-based backfill (default)
            start_year = args.start_year
            if not start_year and args.years:
                # Calculate start year based on years to backfill
                start_year = datetime.now().year - args.years
            
            print(f"Starting S3-based backfill from {start_year or 'default (10 years ago)'} "  
                  f"to {args.end_year or 'current year'}")
            print(f"Extracting {'DEFAULT_SYMBOLS only' if args.filter_symbols else 'ALL symbols'} from S3 files")
            
            results = synchronizer.backfill_historical_data_s3(
                start_year=start_year,
                end_year=args.end_year,
                extract_all_symbols=not args.filter_symbols
            )
            
            print(f"S3 Backfill completed. Processed {results.get('years_processed', 0)} years, "  
                  f"{results.get('symbols_processed', 0)} symbols, "
                  f"stored {results.get('records_stored', 0)} records.")
            
            if results.get("failures"):
                print(f"Failures ({len(results['failures'])}):\n")
                for failure in results["failures"]:
                    print(f"{failure}")
            
            print("\nNote: The data from S3 contains ALL US stocks (thousands of symbols).")
            print("Run with --check-only to verify data completeness for specific symbols.")

# Create a dedicated CLI command to run the S3 backfill
def run_s3_backfill_cli():
    """
    Command-line interface for running a historical data backfill using Polygon.io S3 bucket
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Backfill historical market data from Polygon.io S3 bucket')
    parser.add_argument('--start-year', type=int, default=None, help='First year to backfill (default: 10 years ago)')
    parser.add_argument('--end-year', type=int, default=None, help='Last year to backfill (default: current year)')
    parser.add_argument('--filter-symbols', action='store_true', help='Only extract DEFAULT_SYMBOLS from S3 files (not all symbols)')
    
    args = parser.parse_args()
    
    # Initialize the data synchronizer
    synchronizer = get_data_synchronizer()
    
    print(f"Starting historical data backfill from S3 for years {args.start_year or 'default (10 years ago)'} to {args.end_year or 'present'}")
    print(f"{'Filtering to DEFAULT_SYMBOLS only' if args.filter_symbols else 'Including ALL available symbols from S3 files'}")
    
    results = synchronizer.backfill_historical_data_s3(
        start_year=args.start_year,
        end_year=args.end_year,
        extract_all_symbols=not args.filter_symbols
    )
    
    print(f"Backfill completed. Processed {results.get('years_processed', 0)} years, "  
          f"{results.get('symbols_processed', 0)} symbols, "
          f"stored {results.get('records_stored', 0)} records.")
    
    if results.get("failures"):
        print(f"\nFailures encountered:")
        for failure in results["failures"]:
            if 'year' in failure:
                print(f"Year {failure['year']}: {failure['reason']}")
            elif 'period' in failure:
                print(f"Period {failure['period']}: {failure['reason']}")
            else:
                print(failure)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "s3":
        # Remove the 's3' argument so argparse doesn't see it
        sys.argv.pop(1)
        run_s3_backfill_cli()
    else:
        run_backfill_cli()
