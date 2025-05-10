#!/usr/bin/env python3
"""
An alternative approach to backfill data from Polygon.io's S3 bucket
using the daily files we know we can list (but not the yearly aggregates).
"""
import os
import boto3
import logging
import tempfile
import pandas as pd
import gzip
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to load from .env file
env_file = Path(__file__).resolve().parent.parent / '.env'
if env_file.exists():
    load_dotenv(env_file)
    logger.info(f"Loaded environment variables from {env_file}")

# Get credentials from environment
POLYGON_S3_ACCESS_KEY = os.environ.get("POLYGON_S3_ACCESS_KEY", "")
POLYGON_S3_SECRET_KEY = os.environ.get("POLYGON_S3_SECRET_KEY", "")

# Polygon.io S3 configuration
POLYGON_S3_ENDPOINT = "https://files.polygon.io"
POLYGON_S3_BUCKET = "flatfiles"
POLYGON_PREFIX_STOCKS = "us_stocks_sip"

def backfill_daily_data(start_year=2015, end_year=None, symbols=None):
    """
    Backfill historical data from Polygon.io S3 bucket using daily files
    instead of yearly/monthly aggregates.
    
    Args:
        start_year: First year to backfill
        end_year: Last year to backfill (default: current year)
        symbols: List of specific symbols to filter (default: None, gets all symbols)
    
    Returns:
        Dictionary with backfill results
    """
    if end_year is None:
        end_year = datetime.now().year
        
    logger.info(f"Starting daily backfill for years {start_year} to {end_year}")
    
    # Initialize results dictionary
    results = {
        "status": "success", 
        "days_processed": 0,
        "symbols_processed": 0,
        "records_stored": 0,
        "failures": []
    }
    
    try:
        # Initialize S3 client
        s3_client = boto3.client(
            service_name='s3',
            endpoint_url=POLYGON_S3_ENDPOINT,
            aws_access_key_id=POLYGON_S3_ACCESS_KEY,
            aws_secret_access_key=POLYGON_S3_SECRET_KEY,
            region_name='us-east-1'
        )
        
        # Process each year
        for year in range(start_year, end_year + 1):
            year_str = str(year)
            logger.info(f"Listing data for year {year_str}")
            
            # List all daily files for this year
            try:
                # Since we cannot access yearly aggregates, we'll list the directory
                # to find available months
                months_response = s3_client.list_objects_v2(
                    Bucket=POLYGON_S3_BUCKET,
                    Prefix=f"{POLYGON_PREFIX_STOCKS}/day_aggs_v1/{year_str}/",
                    Delimiter='/'
                )
                
                if 'CommonPrefixes' in months_response:
                    for prefix in months_response['CommonPrefixes']:
                        month_prefix = prefix['Prefix']
                        month = month_prefix.split('/')[-2]
                        logger.info(f"Processing month: {year_str}-{month}")
                        
                        # List all daily files for this month
                        days_response = s3_client.list_objects_v2(
                            Bucket=POLYGON_S3_BUCKET,
                            Prefix=month_prefix
                        )
                        
                        if 'Contents' in days_response:
                            for day_obj in days_response['Contents']:
                                day_key = day_obj['Key']
                                day_file = day_key.split('/')[-1]
                                
                                if day_file.endswith('.csv.gz'):
                                    logger.info(f"Processing file: {day_file}")
                                    
                                    # Download the daily file
                                    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                                        temp_path = temp_file.name
                                    
                                    try:
                                        s3_client.download_file(
                                            POLYGON_S3_BUCKET, 
                                            day_key, 
                                            temp_path
                                        )
                                        
                                        # Process the daily file
                                        logger.info(f"Processing data from {day_file}")
                                        
                                        # Read the compressed CSV
                                        with gzip.open(temp_path, 'rt') as f:
                                            df = pd.read_csv(f)
                                        
                                        # Filter by symbols if specified
                                        if symbols:
                                            df = df[df['T'].isin(symbols)]
                                        
                                        if not df.empty:
                                            logger.info(f"Found {len(df)} records for {day_file}")
                                            # Here you would normally store to database
                                            # self.db.store_stock_prices(df, 'ALL', 'polygon_s3_backfill_daily')
                                            
                                            results['days_processed'] += 1
                                            results['records_stored'] += len(df)
                                            
                                            # Count unique symbols
                                            unique_symbols = df['T'].unique()
                                            results['symbols_processed'] = len(unique_symbols)
                                            
                                    except Exception as e:
                                        logger.error(f"Error processing {day_file}: {str(e)}")
                                        results['failures'].append({
                                            'file': day_file,
                                            'reason': str(e)
                                        })
                                    finally:
                                        # Clean up temp file
                                        if os.path.exists(temp_path):
                                            os.unlink(temp_path)
                        else:
                            logger.warning(f"No daily files found for {year_str}-{month}")
                else:
                    logger.warning(f"No months found for year {year_str}")
                    
            except Exception as e:
                logger.error(f"Error listing daily files for {year_str}: {str(e)}")
                results['failures'].append({
                    'year': year_str,
                    'reason': str(e)
                })
                
    except Exception as e:
        logger.error(f"Error initializing S3 client: {str(e)}")
        results['status'] = 'error'
        results['message'] = f"S3 client initialization error: {str(e)}"
    
    return results

def test_list_available_paths():
    """Test function to list available paths in Polygon's S3 bucket"""
    logger.info("Testing available S3 paths...")
    
    try:
        # Initialize S3 client
        s3_client = boto3.client(
            service_name='s3',
            endpoint_url=POLYGON_S3_ENDPOINT,
            aws_access_key_id=POLYGON_S3_ACCESS_KEY,
            aws_secret_access_key=POLYGON_S3_SECRET_KEY,
            region_name='us-east-1'
        )
        
        # List the root level paths
        logger.info("Listing root level directories...")
        response = s3_client.list_objects_v2(
            Bucket=POLYGON_S3_BUCKET,
            Delimiter='/'
        )
        
        if 'CommonPrefixes' in response:
            for prefix in response['CommonPrefixes']:
                root_prefix = prefix['Prefix']
                logger.info(f"Root directory: {root_prefix}")
                
                # List subdirectories for each root
                sub_response = s3_client.list_objects_v2(
                    Bucket=POLYGON_S3_BUCKET,
                    Prefix=root_prefix,
                    Delimiter='/'
                )
                
                if 'CommonPrefixes' in sub_response:
                    for sub_prefix in sub_response['CommonPrefixes']:
                        sub_dir = sub_prefix['Prefix']
                        logger.info(f"  Subdirectory: {sub_dir}")
        else:
            logger.info("No common prefixes found. Listing first few objects...")
            response = s3_client.list_objects_v2(
                Bucket=POLYGON_S3_BUCKET,
                MaxKeys=10
            )
            
            if 'Contents' in response:
                for item in response['Contents']:
                    logger.info(f"Object: {item['Key']}")
            else:
                logger.info("No objects found.")
                
    except Exception as e:
        logger.error(f"Error: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Backfill data from Polygon.io S3 bucket using daily files')
    parser.add_argument('--start-year', type=int, default=2020, help='First year to backfill')
    parser.add_argument('--end-year', type=int, default=None, help='Last year to backfill (default: current year)')
    parser.add_argument('--symbols', type=str, default=None, help='Comma-separated list of symbols to filter')
    parser.add_argument('--list-paths', action='store_true', help='Only list available paths in S3 bucket')
    
    args = parser.parse_args()
    
    if args.list_paths:
        test_list_available_paths()
    else:
        symbol_list = None
        if args.symbols:
            symbol_list = [s.strip() for s in args.symbols.split(',')]
            
        backfill_daily_data(
            start_year=args.start_year,
            end_year=args.end_year,
            symbols=symbol_list
        )
