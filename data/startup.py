"""
Startup script for initializing the database and data synchronization.
This should be imported and run when the application starts.
"""
import logging
import os
import time
from datetime import datetime, timedelta

from data.database import get_market_db
from data.data_sync import get_data_synchronizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_database():
    """Initialize the database and ensure it's ready for use"""
    logger.info("Initializing market database")
    db = get_market_db()
    
    # Verify database is working
    try:
        # Simple test query
        conn = db._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM sqlite_master")
        count = cursor.fetchone()[0]
        logger.info(f"Database initialized with {count} tables")
        return True
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        return False

def initialize_data_sync():
    """Initialize data synchronization"""
    logger.info("Initializing data synchronization")
    
    # Get the data synchronizer instance
    synchronizer = get_data_synchronizer()
    
    # Start the scheduler
    success = synchronizer.start_scheduled_sync()
    
    if success:
        logger.info("Data synchronization scheduler started")
    else:
        logger.error("Failed to start data synchronization scheduler")
    
    return success

def perform_initial_data_sync():
    """Perform an initial data sync if needed"""
    logger.info("Checking if initial data sync is needed")
    
    db = get_market_db()
    synchronizer = get_data_synchronizer()
    
    # Check when the last sync was performed
    last_sync_date = db.get_last_sync_date('polygon_api', 'prices')
    
    # If no sync in the last 24 hours, perform an initial sync
    if not last_sync_date or (datetime.now() - datetime.strptime(last_sync_date, '%Y-%m-%d')).days >= 1:
        logger.info("Performing initial data sync")
        
        # Sync the last 200 days of price data
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=200)).strftime('%Y-%m-%d')
        
        try:
            # Try S3 first for yesterday's data
            s3_results = synchronizer.sync_stock_prices_from_polygon_s3(
                date=end_date
            )
            
            # If S3 failed or had no results, use API for the full range
            if s3_results["symbols_processed"] == 0:
                synchronizer.sync_stock_prices_from_polygon_api(
                    start_date=start_date,
                    end_date=end_date
                )
        except Exception as e:
            logger.error(f"S3 sync failed, using API: {str(e)}")
            synchronizer.sync_stock_prices_from_polygon_api(
                start_date=start_date,
                end_date=end_date
            )
        
        # Sync fundamental data
        synchronizer.sync_stock_fundamentals_from_yahoo()
        
        logger.info("Initial data sync completed")
        return True
    else:
        logger.info(f"Skipping initial data sync, last sync was on {last_sync_date}")
        return False

def refresh_database(data_source='polygon_s3', symbols=None, days=7, reset_db=False):
    """Refresh the database with data from the specified source
    
    Args:
        data_source (str): The data source to use ('polygon_s3', 'polygon_api', or 'yahoo')
        symbols (list): List of symbols to refresh. If None, use default universe.
        days (int): Number of days to sync (for historical data)
        reset_db (bool): Whether to reset the database before refreshing
        
    Returns:
        dict: Results of the refresh operation
    """
    logger.info(f"Refreshing database from {data_source} source")
    
    db = get_market_db()
    synchronizer = get_data_synchronizer()
    
    # Reset database if requested
    if reset_db:
        try:
            logger.info("Resetting database tables")
            db.reset_tables()
            logger.info("Database reset successful")
        except Exception as e:
            error_msg = f"Database reset failed: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}
    
    # Calculate date range
    end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    
    results = {"status": "success", "operations": []}
    
    try:
        # Sync price data based on selected source
        if data_source == 'polygon_s3':
            s3_result = synchronizer.sync_stock_prices_from_polygon_s3(
                symbols=symbols,
                date=end_date
            )
            results["operations"].append({"source": "polygon_s3", "result": s3_result})
            
            # If S3 had no results, fall back to API
            if s3_result.get("symbols_processed", 0) == 0:
                logger.warning("No data from Polygon S3, falling back to API")
                api_result = synchronizer.sync_stock_prices_from_polygon_api(
                    symbols=symbols,
                    start_date=start_date,
                    end_date=end_date
                )
                results["operations"].append({"source": "polygon_api_fallback", "result": api_result})
        
        elif data_source == 'polygon_api':
            api_result = synchronizer.sync_stock_prices_from_polygon_api(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date
            )
            results["operations"].append({"source": "polygon_api", "result": api_result})
        
        # Always sync fundamentals from Yahoo
        if data_source in ['polygon_s3', 'polygon_api', 'yahoo']:
            yahoo_result = synchronizer.sync_stock_fundamentals_from_yahoo(symbols=symbols)
            results["operations"].append({"source": "yahoo_fundamentals", "result": yahoo_result})
        
        logger.info(f"Database refresh from {data_source} completed successfully")
        results["message"] = f"Database refreshed successfully from {data_source}"
        return results
    
    except Exception as e:
        error_msg = f"Database refresh failed: {str(e)}"
        logger.error(error_msg)
        return {"status": "error", "message": error_msg}

def initialize():
    """Initialize all data components"""
    logger.info("Initializing data components")
    
    # Initialize database
    db_success = initialize_database()
    if not db_success:
        logger.error("Database initialization failed")
        return False
    
    # Initialize data sync
    sync_success = initialize_data_sync()
    if not sync_success:
        logger.error("Data sync initialization failed")
        return False
    
    # Perform initial data sync in a separate thread to not block startup
    import threading
    sync_thread = threading.Thread(target=perform_initial_data_sync)
    sync_thread.daemon = True
    sync_thread.start()
    
    logger.info("Data components initialized successfully")
    return True

# Run initialization when imported
if __name__ != "__main__":
    # Don't initialize automatically when imported for testing
    pass
else:
    # Initialize when run directly
    initialize()
