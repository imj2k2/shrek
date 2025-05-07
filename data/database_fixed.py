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
        """Create necessary tables if they don't exist"""
        try:
            conn = self._get_connection()
            if conn is None:
                logger.error("Cannot create tables: No database connection")
                return
                
            cursor = conn.cursor()
            
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
                volume REAL,
                market_cap REAL,
                UNIQUE(symbol, date)
            )
            ''')
            
            # Data source tracking table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                data_type TEXT NOT NULL,
                source TEXT NOT NULL,
                start_date TEXT,
                end_date TEXT,
                last_updated TEXT,
                UNIQUE(symbol, data_type, source)
            )
            ''')
            
            conn.commit()
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating database tables: {str(e)}")
    
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
                logger.error("Cannot get database stats: No database connection")
                return {}
                
            cursor = conn.cursor()
            
            stats = {}
            tables = ['stock_prices', 'stock_fundamentals', 'options_data', 'crypto_prices', 'data_sources']
            
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                stats[f"{table}_count"] = count
                
                # Get unique symbols for relevant tables
                if table in ['stock_prices', 'stock_fundamentals', 'options_data', 'crypto_prices']:
                    cursor.execute(f"SELECT COUNT(DISTINCT symbol) FROM {table}")
                    unique_symbols = cursor.fetchone()[0]
                    stats[f"{table}_symbols"] = unique_symbols
                    
                    # Get date range for stock_prices
                    if table == 'stock_prices' and count > 0:
                        cursor.execute(f"SELECT MIN(date), MAX(date) FROM {table}")
                        min_date, max_date = cursor.fetchone()
                        stats[f"{table}_date_range"] = f"{min_date} to {max_date}"
            
            return stats
        except Exception as e:
            logger.error(f"Error getting database stats: {str(e)}")
            return {}

# Singleton instance
_market_db_instance = None

def get_market_db():
    """Get the singleton instance of MarketDatabase"""
    global _market_db_instance
    if _market_db_instance is None:
        _market_db_instance = MarketDatabase()
    return _market_db_instance
