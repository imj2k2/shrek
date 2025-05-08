"""
Database module for storing and retrieving historical market data.
Provides a persistent layer for storing data from Polygon.io and other sources.
"""
import os
import logging
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import json
import threading
import time
from threading import local as threading_local

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database file path
DB_PATH = os.environ.get("DB_PATH", "data/market_data.db")

class MarketDatabase:
    """
    Provides a persistent database layer for market data.
    Uses SQLite for simplicity and portability.
    """
    
    def __init__(self, db_path=DB_PATH):
        """Initialize the database connection"""
        self.db_path = db_path
        self._ensure_db_dir_exists()
        self._thread_local = threading_local()
        self._create_tables()
        logger.info(f"Market database initialized at {self.db_path}")
    
    def _ensure_db_dir_exists(self):
        """Ensure the directory for the database file exists"""
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
            logger.info(f"Created directory for database: {db_dir}")
    
    def _get_connection(self):
        """Get a thread-local connection to the SQLite database"""
        try:
            if not hasattr(self._thread_local, 'conn') or self._thread_local.conn is None:
                self._thread_local.conn = sqlite3.connect(self.db_path)
                logger.debug(f"Created new SQLite connection for thread {threading.current_thread().ident}")
            return self._thread_local.conn
        except Exception as e:
            logger.error(f"Error creating database connection: {str(e)}")
            # Create a new connection if there was an error
            try:
                self._thread_local.conn = sqlite3.connect(self.db_path)
                return self._thread_local.conn
            except Exception as e2:
                logger.error(f"Failed to create new connection after error: {str(e2)}")
                # Return None and let the calling code handle it
                return None
    
    def _create_tables(self):
        """Create necessary tables if they don't exist and handle schema migrations"""
        try:
            conn = self._get_connection()
            if conn is None:
                logger.error("Cannot create tables: No database connection")
                return
                
            cursor = conn.cursor()
            
            # Check database version - create version table if it doesn't exist
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS db_version (
                id INTEGER PRIMARY KEY,
                version INTEGER NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            # Get current version
            cursor.execute("SELECT version FROM db_version ORDER BY version DESC LIMIT 1")
            result = cursor.fetchone()
            current_version = result[0] if result else 0
            
            # Migrations - run if needed
            if current_version < 1:
                logger.info("Running database migration to version 1")
                self._migration_v1(cursor)
                
                # Update version
                cursor.execute("INSERT INTO db_version (version) VALUES (1)")
                conn.commit()
                
            # Create standard tables (regardless of version)
            self._create_standard_tables(cursor)
            
            conn.commit()
        except Exception as e:
            logger.error(f"Error creating database tables: {str(e)}")
            # Try to recover if possible
            try:
                if conn:
                    conn.rollback()
            except Exception as rollback_error:
                logger.error(f"Error rolling back transaction: {str(rollback_error)}")
                
    def _create_standard_tables(self, cursor):
        """Create the standard tables required by the application"""
        # Create data_sync_log table to track synchronization activities
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS data_sync_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            data_type TEXT NOT NULL,
            start_date TEXT,
            end_date TEXT,
            symbols TEXT,
            status TEXT NOT NULL,
            records_processed INTEGER,
            error_message TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Stock prices table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            vwap REAL,
            timestamp INTEGER,
            source TEXT,
            created_at TIMESTAMP,
            updated_at TIMESTAMP,
            UNIQUE(symbol, date)
        )
        ''')
        
        # Stock fundamentals table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_fundamentals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            market_cap REAL,
            pe_ratio REAL,
            dividend_yield REAL,
            eps REAL,
            beta REAL,
            full_data TEXT,
            UNIQUE(symbol, date)
        )
        ''')
        
        # Options data table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS options_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            expiration_date TEXT NOT NULL,
            strike_price REAL NOT NULL,
            option_type TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            open_interest INTEGER,
            implied_volatility REAL,
            delta REAL,
            gamma REAL,
            theta REAL,
            vega REAL,
            UNIQUE(symbol, expiration_date, strike_price, option_type, date)
        )
        ''')
        
        # Crypto prices table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS crypto_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            market_cap REAL,
            UNIQUE(symbol, date)
        )
        ''')
        
        # Data sources tracking table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS data_sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            data_type TEXT NOT NULL,
            source TEXT NOT NULL,
            last_updated TIMESTAMP,
            status TEXT,
            UNIQUE(symbol, data_type, source)
        )
        ''')
            
    def _migration_v1(self, cursor):
        """Apply database schema version 1 migrations"""
        try:
            # Recreate approach is safer for SQLite migrations
            # Check if stock_prices table exists and drop it to force recreation
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='stock_prices'")
            if cursor.fetchone():
                logger.info("Recreating stock_prices table with updated schema")
                # Backup existing data
                try:
                    cursor.execute("ALTER TABLE stock_prices RENAME TO stock_prices_backup")
                    logger.info("Backed up existing stock prices table")
                    
                    # Create new table with correct schema
                    cursor.execute('''
                    CREATE TABLE stock_prices (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        date TEXT NOT NULL,
                        open REAL,
                        high REAL,
                        low REAL,
                        close REAL,
                        volume INTEGER,
                        vwap REAL,
                        timestamp INTEGER,
                        source TEXT,
                        created_at TIMESTAMP,
                        updated_at TIMESTAMP,
                        UNIQUE(symbol, date)
                    )
                    ''')
                    
                    # Copy data from backup table (only columns that exist in both)
                    cursor.execute("PRAGMA table_info(stock_prices_backup)")
                    columns = [row[1] for row in cursor.fetchall()]
                    
                    # Build a column list of common columns between old and new table
                    common_columns = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
                    # Add any other columns that might exist in the backup
                    for col in ['vwap', 'timestamp', 'source']:
                        if col in columns:
                            common_columns.append(col)
                    
                    column_str = ', '.join(common_columns)
                    cursor.execute(f"INSERT INTO stock_prices ({column_str}) SELECT {column_str} FROM stock_prices_backup")
                    logger.info("Migrated data from backup table to new schema")
                    
                    # Drop backup table
                    # cursor.execute("DROP TABLE stock_prices_backup") 
                    # Keep backup for safety
                    
                except Exception as e:
                    logger.error(f"Error migrating stock_prices table: {str(e)}")
                    # If something goes wrong, recreate the table from scratch
                    cursor.execute("DROP TABLE IF EXISTS stock_prices")
                    cursor.execute('''
                    CREATE TABLE stock_prices (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        date TEXT NOT NULL,
                        open REAL,
                        high REAL,
                        low REAL,
                        close REAL,
                        volume INTEGER,
                        vwap REAL,
                        timestamp INTEGER,
                        source TEXT,
                        created_at TIMESTAMP,
                        updated_at TIMESTAMP,
                        UNIQUE(symbol, date)
                    )
                    ''')
                    logger.info("Created fresh stock_prices table")
                    
            # Apply similar migration approach to other tables if needed
            
            # Continue with creating tables - existing tables won't be affected by CREATE IF NOT EXISTS
            logger.info("Creating tables for schema v1")
        except Exception as e:
            logger.error(f"Error during schema migration v1: {str(e)}")
            # Re-raise to let the calling method handle it
            raise
            
    def store_stock_prices(self, df, symbol, source='unknown'):
        """Store stock price data in the database
        
        Args:
            df: DataFrame with price data (must have open, high, low, close, volume columns)
            symbol: Stock symbol
            source: Data source identifier
            
        Returns:
            int: Number of records stored
        """
        try:
            conn = self._get_connection()
            if conn is None:
                logger.error(f"Cannot store stock prices for {symbol}: No database connection")
                return 0
                
            cursor = conn.cursor()
            records_stored = 0
            
            # Verify the stock_prices table exists and has the necessary structure
            try:
                # Check if table exists
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='stock_prices'")
                if not cursor.fetchone():
                    # Table doesn't exist, create it
                    logger.info("Creating stock_prices table which is missing")
                    cursor.execute('''
                    CREATE TABLE stock_prices (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        date TEXT NOT NULL,
                        open REAL,
                        high REAL,
                        low REAL,
                        close REAL,
                        volume INTEGER,
                        vwap REAL,
                        timestamp INTEGER,
                        source TEXT,
                        created_at TIMESTAMP,
                        updated_at TIMESTAMP,
                        UNIQUE(symbol, date)
                    )
                    ''')
                    conn.commit()
                else:
                    # Check if id column exists
                    cursor.execute("PRAGMA table_info(stock_prices)")
                    columns = [row[1] for row in cursor.fetchall()]
                    if 'id' not in columns:
                        # The id column is missing - recreate table with correct schema
                        logger.warning("The id column is missing from stock_prices table. Recreating table.")
                        cursor.execute("ALTER TABLE stock_prices RENAME TO stock_prices_old")
                        
                        # Create the proper table
                        cursor.execute('''
                        CREATE TABLE stock_prices (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            symbol TEXT NOT NULL,
                            date TEXT NOT NULL,
                            open REAL,
                            high REAL,
                            low REAL,
                            close REAL,
                            volume INTEGER,
                            vwap REAL,
                            timestamp INTEGER,
                            source TEXT,
                            created_at TIMESTAMP,
                            updated_at TIMESTAMP,
                            UNIQUE(symbol, date)
                        )
                        ''')
                        
                        # Try to migrate data if possible
                        try:
                            cursor.execute("PRAGMA table_info(stock_prices_old)")
                            old_columns = [row[1] for row in cursor.fetchall()]
                            common_cols = [col for col in old_columns if col in ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'timestamp', 'source']]
                            
                            if common_cols:
                                col_str = ', '.join(common_cols)
                                cursor.execute(f"INSERT INTO stock_prices ({col_str}) SELECT {col_str} FROM stock_prices_old")
                                logger.info("Migrated data from old table structure")
                        except Exception as e:
                            logger.warning(f"Could not migrate data: {str(e)}")
                            
                        # Drop old table
                        cursor.execute("DROP TABLE stock_prices_old")
                        conn.commit()
            except Exception as e:
                logger.error(f"Error checking/creating table: {str(e)}")
                raise
            
            # Process each row in the DataFrame
            for date, row in df.iterrows():
                # Convert date to string if it's a datetime
                if isinstance(date, (datetime, pd.Timestamp)):
                    date_str = date.strftime('%Y-%m-%d')
                else:
                    date_str = str(date)
                
                try:
                    # First try to insert directly - faster if the record doesn't exist
                    cursor.execute(
                        """INSERT OR IGNORE INTO stock_prices 
                        (symbol, date, open, high, low, close, volume, source, created_at, updated_at) 
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)""",
                        (symbol, date_str, float(row['open']), float(row['high']), float(row['low']), 
                         float(row['close']), int(row['volume']), source)
                    )
                    
                    # If insert didn't work, then update instead
                    if cursor.rowcount == 0:
                        cursor.execute(
                            """UPDATE stock_prices 
                            SET open = ?, high = ?, low = ?, close = ?, volume = ?, source = ?, updated_at = CURRENT_TIMESTAMP 
                            WHERE symbol = ? AND date = ?""",
                            (float(row['open']), float(row['high']), float(row['low']), float(row['close']), 
                             int(row['volume']), source, symbol, date_str)
                        )
                    
                    records_stored += 1
                except Exception as e:
                    logger.warning(f"Error storing price record for {symbol} on {date_str}: {str(e)}")
            
            conn.commit()
            logger.info(f"Stored {records_stored} price records for {symbol} from {source}")
            return records_stored
            
        except Exception as e:
            logger.error(f"Error storing price data for {symbol}: {str(e)}")
            return 0
    
    def get_stock_prices(self, symbol, start_date=None, end_date=None):
        """Get stock price data for a symbol with optional date range"""
        try:
            conn = self._get_connection()
            if conn is None:
                logger.error(f"Cannot get stock prices for {symbol}: No database connection")
                return pd.DataFrame()  # Return empty DataFrame
                
            query = "SELECT * FROM stock_prices WHERE symbol = ?"
            params = [symbol]
            
            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)
            
            query += " ORDER BY date ASC"
            
            df = pd.read_sql_query(query, conn, params=params)
            
            if df.empty:
                logger.warning(f"No price data found for {symbol}")
            else:
                logger.info(f"Retrieved {len(df)} price records for {symbol}")
            
            return df
        except Exception as e:
            logger.error(f"Error retrieving price data for {symbol}: {str(e)}")
            return pd.DataFrame()  # Return empty DataFrame on error
    
    def get_stock_fundamentals(self, symbol, date=None):
        """Get fundamental data for a stock"""
        query = "SELECT * FROM stock_fundamentals WHERE symbol = ?"
        params = [symbol]
        
        if date:
            query += " AND date = ?"
            params.append(date)
        else:
            query += " ORDER BY date DESC LIMIT 1"
        
        try:
            conn = self._get_connection()
            if conn is None:
                logger.error(f"Cannot get stock fundamentals for {symbol}: No database connection")
                return None
                
            cursor = conn.cursor()
            cursor.execute(query, params)
            row = cursor.fetchone()
            
            if not row:
                logger.warning(f"No fundamental data found for {symbol}")
                return None
            
            # Convert row to dictionary
            columns = [col[0] for col in cursor.description]
            data = dict(zip(columns, row))
            
            # Parse JSON fields
            if 'full_data' in data and data['full_data']:
                try:
                    data['full_data'] = json.loads(data['full_data'])
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON data for {symbol}")
            
            return data
        except Exception as e:
            logger.error(f"Error retrieving fundamental data for {symbol}: {str(e)}")
            return None
    
    def insert_stock_prices(self, symbol, df):
        """Insert stock price data into the database"""
        if df.empty:
            logger.warning(f"No data to insert for {symbol}")
            return 0
        
        try:
            conn = self._get_connection()
            if conn is None:
                logger.error(f"Cannot insert stock prices for {symbol}: No database connection")
                return 0
                
            # Make a copy to avoid modifying the original DataFrame
            df_copy = df.copy()
            
            # Ensure the DataFrame has the required columns
            required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df_copy.columns:
                    if col == 'date':
                        logger.error(f"Missing required column 'date' for {symbol}")
                        return 0
                    else:
                        logger.warning(f"Missing column {col} for {symbol}, filling with NaN")
                        df_copy[col] = None
            
            # Add symbol column if not present
            if 'symbol' not in df_copy.columns:
                df_copy['symbol'] = symbol
            
            # Convert date to string format if it's a datetime
            if pd.api.types.is_datetime64_any_dtype(df_copy['date']):
                df_copy['date'] = df_copy['date'].dt.strftime('%Y-%m-%d')
            
            # Insert data
            cursor = conn.cursor()
            rows_inserted = 0
            
            for _, row in df_copy.iterrows():
                try:
                    cursor.execute('''
                    INSERT OR REPLACE INTO stock_prices 
                    (symbol, date, open, high, low, close, volume, vwap, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        row.get('symbol', symbol),
                        row['date'],
                        row.get('open', None),
                        row.get('high', None),
                        row.get('low', None),
                        row.get('close', None),
                        row.get('volume', None),
                        row.get('vwap', None),
                        row.get('timestamp', None)
                    ))
                    rows_inserted += 1
                except Exception as e:
                    logger.error(f"Error inserting row for {symbol} on {row['date']}: {str(e)}")
            
            conn.commit()
            logger.info(f"Inserted {rows_inserted} price records for {symbol}")
            return rows_inserted
        except Exception as e:
            logger.error(f"Error inserting price data for {symbol}: {str(e)}")
            return 0
    
    def insert_stock_fundamentals(self, symbol, data, date=None):
        """Insert fundamental data for a stock"""
        if not data:
            logger.warning(f"No fundamental data to insert for {symbol}")
            return False
        
        try:
            conn = self._get_connection()
            if conn is None:
                logger.error(f"Cannot insert stock fundamentals for {symbol}: No database connection")
                return False
                
            # Use provided date or today's date
            if not date:
                date = datetime.now().strftime('%Y-%m-%d')
            
            # Convert full_data to JSON if it's a dict
            full_data = data.get('full_data', {})
            if isinstance(full_data, dict):
                full_data = json.dumps(full_data)
            
            cursor = conn.cursor()
            cursor.execute('''
            INSERT OR REPLACE INTO stock_fundamentals 
            (symbol, date, market_cap, pe_ratio, dividend_yield, eps, beta, full_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                date,
                data.get('market_cap', None),
                data.get('pe_ratio', None),
                data.get('dividend_yield', None),
                data.get('eps', None),
                data.get('beta', None),
                full_data
            ))
            
            conn.commit()
            logger.info(f"Inserted fundamental data for {symbol} on {date}")
            return True
        except Exception as e:
            logger.error(f"Error inserting fundamental data for {symbol}: {str(e)}")
            return False
    
    def update_data_source(self, symbol, data_type, source, start_date=None, end_date=None):
        """Update the data source tracking table"""
        try:
            conn = self._get_connection()
            if conn is None:
                logger.error(f"Cannot update data source for {symbol}: No database connection")
                return False
                
            cursor = conn.cursor()
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            cursor.execute('''
            INSERT OR REPLACE INTO data_sources 
            (symbol, data_type, source, start_date, end_date, last_updated)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                data_type,
                source,
                start_date,
                end_date,
                now
            ))
            
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error updating data source for {symbol}: {str(e)}")
            return False
    
    def get_data_sources(self, symbol=None, data_type=None):
        """Get data source information"""
        try:
            conn = self._get_connection()
            if conn is None:
                logger.error("Cannot get data sources: No database connection")
                return pd.DataFrame()
                
            query = "SELECT * FROM data_sources"
            params = []
            
            if symbol:
                query += " WHERE symbol = ?"
                params.append(symbol)
                
                if data_type:
                    query += " AND data_type = ?"
                    params.append(data_type)
            elif data_type:
                query += " WHERE data_type = ?"
                params.append(data_type)
            
            df = pd.read_sql_query(query, conn, params=params)
            return df
        except Exception as e:
            logger.error(f"Error retrieving data sources: {str(e)}")
            return pd.DataFrame()
    
    def clear_data(self, table=None):
        """Clear data from specified table or all tables"""
        try:
            conn = self._get_connection()
            if conn is None:
                logger.error("Cannot clear data: No database connection")
                return False
                
            cursor = conn.cursor()
            
            if table:
                cursor.execute(f"DELETE FROM {table}")
                logger.info(f"Cleared all data from {table}")
            else:
                tables = ['stock_prices', 'stock_fundamentals', 'options_data', 'crypto_prices', 'data_sources']
                for t in tables:
                    cursor.execute(f"DELETE FROM {t}")
                logger.info("Cleared all data from all tables")
            
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error clearing data: {str(e)}")
            return False
    
    def get_database_stats(self):
        """Get statistics about the database"""
        try:
            conn = self._get_connection()
            if conn is None:
                return {"error": "No database connection"}
                
            cursor = conn.cursor()
            
            # Get list of tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            # Get count of records in each table
            stats = {}
            for table in tables:
                table_name = table[0]
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                stats[table_name] = count
            
            return stats
        except Exception as e:
            logger.error(f"Error getting database stats: {str(e)}")
            return {"error": str(e)}
            
    def log_data_sync(self, source, data_type, start_date=None, end_date=None, symbols=None, status="success", records_processed=0, error_message=None):
        """Log a data synchronization activity
        
        Args:
            source: Data source (e.g., 'polygon_api', 'polygon_s3', 'yahoo')
            data_type: Type of data (e.g., 'prices', 'fundamentals')
            start_date: Start date for the sync period
            end_date: End date for the sync period
            symbols: Comma-separated list of symbols that were synced
            status: Status of the sync operation ('success', 'partial', 'error')
            records_processed: Number of records processed
            error_message: Error message if any
            
        Returns:
            bool: True if log was created successfully, False otherwise
        """
        try:
            conn = self._get_connection()
            if conn is None:
                logger.error("Failed to get database connection for logging data sync")
                return False
                
            cursor = conn.cursor()
            
            cursor.execute("""
            INSERT INTO data_sync_log 
            (source, data_type, start_date, end_date, symbols, status, records_processed, error_message) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (source, data_type, start_date, end_date, symbols, status, records_processed, error_message))
            
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error logging data sync: {str(e)}")
            return False
            
    def get_last_sync_date(self, source, data_type):
        """Get the date of the last successful data synchronization
        
        Args:
            source: Data source (e.g., 'polygon_api', 'polygon_s3', 'yahoo')
            data_type: Type of data (e.g., 'prices', 'fundamentals')
            
        Returns:
            str: Date of the last successful sync in YYYY-MM-DD format, or None if no sync found
        """
        try:
            conn = self._get_connection()
            if conn is None:
                logger.error("Failed to get database connection for retrieving last sync date")
                return None
                
            cursor = conn.cursor()
            
            cursor.execute("""
            SELECT end_date FROM data_sync_log 
            WHERE source = ? AND data_type = ? AND status IN ('success', 'partial')
            ORDER BY end_date DESC LIMIT 1
            """, (source, data_type))
            
            result = cursor.fetchone()
            
            if result and result[0]:
                return result[0]
            else:
                return None
        except Exception as e:
            logger.error(f"Error getting last sync date: {str(e)}")
            return None

# Singleton instance
_market_db_instance = None

def get_market_db():
    """Get the singleton instance of MarketDatabase"""
    global _market_db_instance
    if _market_db_instance is None:
        _market_db_instance = MarketDatabase()
    return _market_db_instance
