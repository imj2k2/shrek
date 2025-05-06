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
        self._thread_local.conn = None
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
        if not hasattr(self._thread_local, 'conn') or self._thread_local.conn is None:
            self._thread_local.conn = sqlite3.connect(self.db_path)
            logger.debug(f"Created new SQLite connection for thread {threading.current_thread().ident}")
        return self._thread_local.conn
    
    def _create_tables(self):
        """Create necessary tables if they don't exist"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Stock price data table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_prices (
            symbol TEXT,
            date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            source TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (symbol, date)
        )
        ''')
        
        # Fundamental data table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_fundamentals (
            symbol TEXT,
            date TEXT,
            pe_ratio REAL,
            eps REAL,
            dividend_yield REAL,
            market_cap REAL,
            revenue REAL,
            profit_margin REAL,
            debt_to_equity REAL,
            roe REAL,
            source TEXT,
            data_json TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (symbol, date)
        )
        ''')
        
        # Data sync log table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS data_sync_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            data_type TEXT,
            start_date TEXT,
            end_date TEXT,
            symbols TEXT,
            status TEXT,
            records_processed INTEGER,
            error_message TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create indexes for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_stock_prices_symbol ON stock_prices (symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_stock_prices_date ON stock_prices (date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_stock_fundamentals_symbol ON stock_fundamentals (symbol)')
        
        conn = self._get_connection()
        conn.commit()
        logger.info("Database tables created or verified")
    
    def store_stock_prices(self, df, symbol, source='polygon'):
        """
        Store stock price data in the database
        
        Args:
            df: Pandas DataFrame with stock price data
            symbol: Stock symbol
            source: Data source (e.g., 'polygon', 'yahoo')
        """
        if df is None or df.empty:
            logger.warning(f"No data to store for {symbol}")
            return 0
        
        # Ensure the DataFrame has the expected columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Convert column names to lowercase
        df.columns = df.columns.str.lower()
        
        # Check if all required columns exist
        if not all(col in df.columns for col in required_columns):
            logger.error(f"Missing required columns for {symbol}. Required: {required_columns}, Got: {df.columns.tolist()}")
            return 0
        
        # Ensure the index is a datetime and convert to string format for storage
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning(f"Index for {symbol} is not DatetimeIndex, attempting to convert")
            df.index = pd.to_datetime(df.index)
        
        # Create a copy of the DataFrame with the date as a column
        df_to_store = df.copy()
        df_to_store['date'] = df_to_store.index.strftime('%Y-%m-%d')
        df_to_store['symbol'] = symbol
        df_to_store['source'] = source
        
        # Select only the columns we want to store
        columns_to_store = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'source']
        df_to_store = df_to_store[columns_to_store]
        
        try:
            # Use pandas to_sql with 'replace' if the record exists
            conn = self._get_connection()
            df_to_store.to_sql('stock_prices', conn, if_exists='append', index=False)
            conn.commit()
            logger.info(f"Stored {len(df_to_store)} price records for {symbol} from {source}")
            return len(df_to_store)
        except sqlite3.IntegrityError:
            # Handle duplicate records
            logger.info(f"Some records for {symbol} already exist, updating...")
            records_updated = 0
            
            for _, row in df_to_store.iterrows():
                try:
                    conn = self._get_connection()
                    cursor = conn.cursor()
                    cursor.execute('''
                    INSERT OR REPLACE INTO stock_prices 
                    (symbol, date, open, high, low, close, volume, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        row['symbol'], row['date'], row['open'], row['high'], 
                        row['low'], row['close'], row['volume'], row['source']
                    ))
                    records_updated += 1
                except Exception as e:
                    logger.error(f"Error updating record for {symbol} on {row['date']}: {str(e)}")
            
            conn = self._get_connection()
            conn.commit()
            logger.info(f"Updated {records_updated} price records for {symbol}")
            return records_updated
        except Exception as e:
            logger.error(f"Error storing price data for {symbol}: {str(e)}")
            return 0
    
    def store_stock_fundamentals(self, symbol, data, date=None, source='yahoo'):
        """
        Store stock fundamental data in the database
        
        Args:
            symbol: Stock symbol
            data: Dictionary with fundamental data
            date: Date for the data (defaults to today)
            source: Data source (e.g., 'yahoo', 'polygon')
        """
        if not data:
            logger.warning(f"No fundamental data to store for {symbol}")
            return False
        
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            # Extract key metrics from the data
            pe_ratio = data.get('trailingPE') or data.get('forwardPE')
            eps = data.get('trailingEPS') or data.get('forwardEPS')
            dividend_yield = data.get('dividendYield', 0)
            if dividend_yield:
                dividend_yield = dividend_yield * 100  # Convert to percentage
            
            market_cap = data.get('marketCap')
            revenue = data.get('totalRevenue')
            profit_margin = data.get('profitMargins')
            if profit_margin:
                profit_margin = profit_margin * 100  # Convert to percentage
            
            debt_to_equity = data.get('debtToEquity')
            roe = data.get('returnOnEquity')
            if roe:
                roe = roe * 100  # Convert to percentage
            
            # Store the full data as JSON
            data_json = json.dumps(data)
            
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute('''
            INSERT OR REPLACE INTO stock_fundamentals 
            (symbol, date, pe_ratio, eps, dividend_yield, market_cap, revenue, 
             profit_margin, debt_to_equity, roe, source, data_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol, date, pe_ratio, eps, dividend_yield, market_cap, revenue,
                profit_margin, debt_to_equity, roe, source, data_json
            ))
            
            conn.commit()
            logger.info(f"Stored fundamental data for {symbol} from {source}")
            return True
        except Exception as e:
            logger.error(f"Error storing fundamental data for {symbol}: {str(e)}")
            return False
    
    def log_data_sync(self, source, data_type, start_date, end_date, symbols, status, records_processed=0, error_message=None):
        """
        Log data synchronization activity
        
        Args:
            source: Data source (e.g., 'polygon', 'yahoo')
            data_type: Type of data (e.g., 'prices', 'fundamentals')
            start_date: Start date for the sync period
            end_date: End date for the sync period
            symbols: Symbols that were synced (can be a list or comma-separated string)
            status: Status of the sync (e.g., 'success', 'error')
            records_processed: Number of records processed
            error_message: Error message if status is 'error'
        """
        if isinstance(symbols, list):
            symbols = ','.join(symbols)
        
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute('''
            INSERT INTO data_sync_log 
            (source, data_type, start_date, end_date, symbols, status, records_processed, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                source, data_type, start_date, end_date, symbols, status, records_processed, error_message
            ))
            
            conn.commit()
            logger.info(f"Logged data sync: {source} {data_type} {status}")
            return True
        except Exception as e:
            logger.error(f"Error logging data sync: {str(e)}")
            return False
    
    def get_stock_prices(self, symbol, start_date=None, end_date=None):
        """
        Retrieve stock price data from the database
        
        Args:
            symbol: Stock symbol
            start_date: Start date (format: 'YYYY-MM-DD')
            end_date: End date (format: 'YYYY-MM-DD')
            
        Returns:
            Pandas DataFrame with stock price data
        """
        query = "SELECT * FROM stock_prices WHERE symbol = ?"
        params = [symbol]
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        
        query += " ORDER BY date ASC"
        
        try:
            conn = self._get_connection()
            df = pd.read_sql_query(query, conn, params=params)
            
            if df.empty:
                logger.warning(f"No price data found for {symbol}")
                return None
            
            # Set the date as the index
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            logger.info(f"Retrieved {len(df)} price records for {symbol}")
            return df
        except Exception as e:
            logger.error(f"Error retrieving price data for {symbol}: {str(e)}")
            return None
    
    def get_stock_fundamentals(self, symbol, date=None):
        """
        Retrieve stock fundamental data from the database
        
        Args:
            symbol: Stock symbol
            date: Specific date (format: 'YYYY-MM-DD'), if None returns the latest
            
        Returns:
            Dictionary with fundamental data
        """
        query = "SELECT * FROM stock_fundamentals WHERE symbol = ?"
        params = [symbol]
        
        if date:
            query += " AND date = ?"
            params.append(date)
        else:
            query += " ORDER BY date DESC LIMIT 1"
        
        try:
            cursor = self.conn.cursor()
            cursor.execute(query, params)
            row = cursor.fetchone()
            
            if not row:
                logger.warning(f"No fundamental data found for {symbol}")
                return None
            
            # Convert row to dictionary
            columns = [desc[0] for desc in cursor.description]
            data = dict(zip(columns, row))
            
            # Parse the JSON data if available
            if 'data_json' in data and data['data_json']:
                try:
                    data['full_data'] = json.loads(data['data_json'])
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse JSON data for {symbol}")
            
            logger.info(f"Retrieved fundamental data for {symbol}")
            return data
        except Exception as e:
            logger.error(f"Error retrieving fundamental data for {symbol}: {str(e)}")
            return None
    
    def get_last_sync_date(self, source, data_type, symbol=None):
        """
        Get the date of the last successful data sync
        
        Args:
            source: Data source (e.g., 'polygon', 'yahoo')
            data_type: Type of data (e.g., 'prices', 'fundamentals')
            symbol: Specific symbol to check (optional)
            
        Returns:
            Date of the last successful sync or None
        """
        query = """
        SELECT MAX(end_date) FROM data_sync_log 
        WHERE source = ? AND data_type = ? AND status = 'success'
        """
        params = [source, data_type]
        
        if symbol:
            query += " AND symbols LIKE ?"
            params.append(f"%{symbol}%")
        
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(query, params)
            result = cursor.fetchone()[0]
            
            return result
        except Exception as e:
            logger.error(f"Error getting last sync date: {str(e)}")
            return None
    
    def reset_tables(self):
        """Reset all database tables (delete all data)"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get list of tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        # Delete data from each table except sqlite_sequence
        for table in tables:
            table_name = table[0]
            if table_name != 'sqlite_sequence':
                try:
                    cursor.execute(f"DELETE FROM {table_name}")
                    logger.info(f"Deleted all data from table {table_name}")
                except Exception as e:
                    logger.error(f"Error deleting data from table {table_name}: {str(e)}")
        
        # Commit changes
        conn.commit()
        
        # Log the reset
        self.log_data_sync(
            source="system",
            data_type="reset",
            status="success",
            message="Database tables reset"
        )
        
        return True
        
    def close(self):
        """Close the database connection for the current thread"""
        if hasattr(self._thread_local, 'conn') and self._thread_local.conn is not None:
            self._thread_local.conn.close()
            self._thread_local.conn = None
            logger.info(f"Database connection closed for thread {threading.current_thread().ident}")
            
    def close_all(self):
        """Close all database connections"""
        self.close()
        logger.info("All database connections closed")

# Singleton instance
_market_db_instance = None

def get_market_db():
    """Get the singleton instance of MarketDatabase"""
    global _market_db_instance
    if _market_db_instance is None:
        _market_db_instance = MarketDatabase()
    return _market_db_instance
