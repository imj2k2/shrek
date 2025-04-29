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
        
    def fetch_stock_data(self, symbol: str, start_date=None, end_date=None, source: str = "yahoo"):
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
                return df
            else:
                self.logger.warning(f"No data found for {symbol} in Yahoo Finance")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Error fetching stock data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
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
                        return df
            
            self.logger.warning(f"No data found for {symbol}")
            return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Error fetching crypto data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
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
