# Data fetcher for stock, crypto, and options data
import requests
import os
import pandas as pd
import yfinance as yf
import logging
from datetime import datetime, timedelta

class DataFetcher:
    def __init__(self):
        self.polygon_key = os.getenv("POLYGON_API_KEY", "")
        self.logger = logging.getLogger("DataFetcher")
        
    def fetch_stock_data(self, symbol: str, start_date=None, end_date=None, source: str = None):
        """Fetch stock data from either Polygon.io or Yahoo Finance"""
        try:
            # Format dates if provided
            if start_date and isinstance(start_date, datetime):
                start_str = start_date.strftime('%Y-%m-%d')
            else:
                start_str = "2020-01-01"
                
            if end_date and isinstance(end_date, datetime):
                end_str = end_date.strftime('%Y-%m-%d')
            else:
                end_str = datetime.now().strftime('%Y-%m-%d')
            
            # If source is not specified, use polygon if key is available
            if source is None:
                source = "polygon" if self.polygon_key else "yahoo"
                
            # Log which API key we're using
            self.logger.info(f"Using API source: {source} (Polygon key available: {bool(self.polygon_key)})")
            
            if source == "polygon" and self.polygon_key:
                url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_str}/{end_str}?adjusted=true&sort=asc&limit=500&apiKey={self.polygon_key}"
                self.logger.info(f"Fetching {symbol} data from Polygon.io")
                resp = requests.get(url)
                
                if resp.ok:
                    data = resp.json()
                    # Convert to pandas DataFrame
                    if 'results' in data and data['results']:
                        df = pd.DataFrame(data['results'])
                        df['date'] = pd.to_datetime(df['t'], unit='ms')
                        df.set_index('date', inplace=True)
                        df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'}, inplace=True)
                        # Set data source attribute
                        df.data_source = 'polygon'
                        return df
                    else:
                        self.logger.warning(f"No results found for {symbol} in Polygon.io response")
                else:
                    self.logger.warning(f"Failed to fetch {symbol} from Polygon.io: {resp.status_code}")
            
            # Fallback to Yahoo Finance
            self.logger.info(f"Fetching {symbol} data from Yahoo Finance")
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_str, end=end_str)
            
            if not df.empty:
                # Set data source attribute
                df.data_source = 'yahoo'
                return df
            else:
                self.logger.warning(f"No data found for {symbol} in Yahoo Finance")
                self.logger.info(f"Falling back to mock data for {symbol}")
                return self._generate_mock_data(symbol, start_str, end_str)
                
        except Exception as e:
            self.logger.error(f"Error fetching stock data for {symbol}: {str(e)}")
            self.logger.info(f"Falling back to mock data for {symbol} after error")
            return self._generate_mock_data(symbol, start_str, end_str)
    
    def fetch_crypto_data(self, symbol: str, start_date=None, end_date=None, interval="1d"):
        """Fetch cryptocurrency data"""
        try:
            # Format dates if provided
            if start_date and isinstance(start_date, datetime):
                start_str = start_date.strftime('%Y-%m-%d')
            else:
                start_str = "2020-01-01"
                
            if end_date and isinstance(end_date, datetime):
                end_str = end_date.strftime('%Y-%m-%d')
            else:
                end_str = datetime.now().strftime('%Y-%m-%d')
            
            # Try Yahoo Finance first (more reliable for crypto)
            crypto_symbol = f"{symbol}-USD"
            self.logger.info(f"Fetching {crypto_symbol} data from Yahoo Finance")
            ticker = yf.Ticker(crypto_symbol)
            df = ticker.history(start=start_str, end=end_str, interval=interval)
            
            if not df.empty:
                return df
            
            # Fallback to Polygon if we have an API key
            if self.polygon_key:
                url = f"https://api.polygon.io/v2/aggs/ticker/X:{symbol}USD/range/1/day/{start_str}/{end_str}?adjusted=true&sort=asc&limit=500&apiKey={self.polygon_key}"
                self.logger.info(f"Fetching {symbol} data from Polygon.io")
                resp = requests.get(url)
                
                if resp.ok:
                    data = resp.json()
                    # Convert to pandas DataFrame
                    if 'results' in data and data['results']:
                        df = pd.DataFrame(data['results'])
                        df['date'] = pd.to_datetime(df['t'], unit='ms')
                        df.set_index('date', inplace=True)
                        df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'}, inplace=True)
                        # Set data source attribute
                        df.data_source = 'polygon'
                        return df
            
            self.logger.warning(f"No data found for {symbol}")
            self.logger.info(f"Falling back to mock data for crypto {symbol}")
            return self._generate_mock_data(symbol, start_str, end_str, is_crypto=True)
                
        except Exception as e:
            self.logger.error(f"Error fetching crypto data for {symbol}: {str(e)}")
            self.logger.info(f"Falling back to mock data for crypto {symbol} after error")
            return self._generate_mock_data(symbol, start_str, end_str, is_crypto=True)
    
    def fetch_options_data(self, symbol: str):
        """Fetch options data for a symbol"""
        try:
            if not self.polygon_key:
                self.logger.warning("No Polygon API key provided for options data")
                return {}
                
            url = f"https://api.polygon.io/v3/reference/options/contracts?underlying_ticker={symbol}&apiKey={self.polygon_key}"
            self.logger.info(f"Fetching options data for {symbol} from Polygon.io")
            resp = requests.get(url)
            
            if resp.ok:
                return resp.json()
            else:
                self.logger.warning(f"Failed to fetch options data for {symbol}: {resp.status_code}")
                return {}
                
        except Exception as e:
            self.logger.error(f"Error fetching options data for {symbol}: {str(e)}")
            return {}
            
    def _generate_mock_data(self, symbol, start_date_str, end_date_str, is_crypto=False):
        """Generate mock stock or crypto data for backtesting when APIs fail"""
        import numpy as np
        
        self.logger.info(f"Generating mock data for {symbol} from {start_date_str} to {end_date_str}")
        
        # Convert date strings to datetime objects
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
        
        # Generate a date range
        date_range = [start_date + timedelta(days=i) for i in range((end_date - start_date).days + 1)]
        
        # For crypto, include weekends; for stocks, filter out weekends
        if not is_crypto:
            date_range = [date for date in date_range if date.weekday() < 5]  # 0-4 are Monday to Friday
        
        # Set initial price based on symbol
        if is_crypto:
            # Crypto prices and volatility
            if symbol == 'BTC':
                initial_price = 35000.0
                volatility = 0.03
            elif symbol == 'ETH':
                initial_price = 2000.0
                volatility = 0.035
            else:
                initial_price = 500.0
                volatility = 0.04
        else:
            # Stock prices and volatility
            if symbol == 'AAPL':
                initial_price = 150.0
                volatility = 0.015
            elif symbol == 'MSFT':
                initial_price = 250.0
                volatility = 0.012
            elif symbol == 'GOOG':
                initial_price = 2500.0
                volatility = 0.018
            else:
                # Default values for other symbols
                initial_price = 100.0
                volatility = 0.02
        
        # Generate price data with random walk
        np.random.seed(hash(symbol) % 10000)  # Seed based on symbol for consistency
        returns = np.random.normal(0.0005, volatility, len(date_range))  # Slight upward bias
        prices = [initial_price]
        
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        prices = prices[1:]  # Remove the initial seed price
        
        # Create DataFrame
        df = pd.DataFrame({
            'Open': prices,
            'High': [p * (1 + np.random.uniform(0, 0.01)) for p in prices],
            'Low': [p * (1 - np.random.uniform(0, 0.01)) for p in prices],
            'Close': [p * (1 + np.random.normal(0, 0.003)) for p in prices],
            'Volume': [int(np.random.uniform(1000000, 10000000)) for _ in prices]
        }, index=date_range)
        
        # Set data source attribute
        df.data_source = 'mock'
        
        self.logger.info(f"Generated mock data for {symbol} with {len(df)} data points")
        return df
