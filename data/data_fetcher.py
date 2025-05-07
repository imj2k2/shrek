# Data fetcher for stock, crypto, and options data
import requests
import os
import time
import pandas as pd
import yfinance as yf
import logging
from datetime import datetime, timedelta

class DataFetcher:
    def __init__(self):
        self.polygon_key = os.getenv("POLYGON_API_KEY", "")
        self.logger = logging.getLogger("DataFetcher")
        
    def fetch_daily_bars(self, symbol: str, start_date=None, end_date=None, source: str = None):
        """Fetch daily price bars for a stock symbol
        
        Args:
            symbol: Stock symbol
            start_date: Start date for data
            end_date: End date for data
            source: Data source ('polygon' or 'yahoo')
            
        Returns:
            DataFrame with daily OHLCV data
        """
        # This is a wrapper around fetch_stock_data to maintain compatibility with data_sync.py
        return self.fetch_stock_data(symbol, start_date, end_date, source, use_mock_data=True)
        
    def fetch_stock_data(self, symbol: str, start_date=None, end_date=None, source: str = None, use_mock_data=True):
        """Fetch stock data from either Polygon.io or Yahoo Finance
        
        Args:
            symbol: Stock symbol
            start_date: Start date for data
            end_date: End date for data
            source: Data source ('polygon' or 'yahoo')
            use_mock_data: Whether to fallback to mock data if real data is unavailable
            
        Returns:
            DataFrame with stock data or None if data unavailable and mock data disabled
        """
        try:
            # Format dates if provided
            if start_date and isinstance(start_date, str):
                start_str = start_date
            elif start_date and isinstance(start_date, datetime):
                start_str = start_date.strftime('%Y-%m-%d')
            else:
                start_str = "2020-01-01"
                
            if end_date and isinstance(end_date, str):
                end_str = end_date
            elif end_date and isinstance(end_date, datetime):
                end_str = end_date.strftime('%Y-%m-%d')
            else:
                end_str = datetime.now().strftime('%Y-%m-%d')
            
            # If source is not specified, use polygon if key is available
            if source is None:
                source = "polygon" if self.polygon_key else "yahoo"
                
            # Log which API key we're using
            self.logger.info(f"Using API source: {source} (Polygon key available: {bool(self.polygon_key)})")
            
            if source == "polygon" and self.polygon_key:
                try:
                    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_str}/{end_str}?adjusted=true&sort=asc&limit=500&apiKey={self.polygon_key}"
                    self.logger.info(f"Fetching {symbol} data from Polygon.io")
                    
                    # Add timeout to avoid hanging requests
                    resp = requests.get(url, timeout=10)
                    
                    if resp.ok:
                        data = resp.json()
                        # Convert to pandas DataFrame
                        if 'results' in data and data['results']:
                            df = pd.DataFrame(data['results'])
                            df['date'] = pd.to_datetime(df['t'], unit='ms')
                            df.set_index('date', inplace=True)
                            df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'}, inplace=True)
                            
                            # Convert column names to lowercase to match Backtrader expectations
                            df.columns = df.columns.str.lower()
                            
                            # Set data source attribute
                            df.data_source = 'polygon'
                            return df
                        else:
                            self.logger.warning(f"No results found for {symbol} in Polygon.io response. Response: {data}")
                    else:
                        # For 403 errors, include masked API key info to help with debugging
                        if resp.status_code == 403:
                            # Mask the API key for security (show only first 4 and last 4 chars if long enough)
                            masked_key = self.polygon_key
                            if len(masked_key) > 8:
                                masked_key = masked_key[:4] + '*' * (len(masked_key) - 8) + masked_key[-4:]
                            elif len(masked_key) > 0:
                                masked_key = masked_key[:2] + '*' * (len(masked_key) - 2)
                            
                            error_msg = f"Failed to fetch {symbol} from Polygon.io: HTTP 403 Forbidden. API Key used: {masked_key}"
                            error_msg += ". This typically indicates an invalid or expired API key."
                        else:
                            error_msg = f"Failed to fetch {symbol} from Polygon.io: HTTP {resp.status_code}"
                        
                        try:
                            error_detail = resp.json().get('error', '')
                            if error_detail:
                                error_msg += f", Error: {error_detail}"
                        except:
                            pass
                        self.logger.error(error_msg)
                except requests.exceptions.Timeout:
                    self.logger.error(f"Polygon API timeout for {symbol}. Check your network connection or API status.")
                except requests.exceptions.ConnectionError:
                    self.logger.error(f"Polygon API connection error for {symbol}. Check your network connection.")
                except Exception as e:
                    self.logger.error(f"Polygon API error for {symbol}: {str(e)}")
            
            # Fallback to direct data download which is more reliable in Docker
            self.logger.info(f"Fetching {symbol} data directly")
            
            # Try direct download approach first (most reliable in Docker environments)
            try:
                # Convert dates to proper format for URL
                start_dt = pd.to_datetime(start_str)
                end_dt = pd.to_datetime(end_str)
                
                # Add one day to end date to ensure inclusive range
                end_dt = end_dt + pd.Timedelta(days=1)
                
                # Convert to Unix timestamp (seconds since epoch)
                start_ts = int(start_dt.timestamp())
                end_ts = int(end_dt.timestamp())
                
                # Yahoo Finance direct CSV download URL
                interval = "1d"  # daily data
                url = f"https://query1.finance.yahoo.com/v7/finance/download/{symbol}?period1={start_ts}&period2={end_ts}&interval={interval}&events=history"
                
                self.logger.info(f"Trying direct download from {url}")
                
                # Use pandas to download CSV directly (more reliable than yfinance in Docker)
                df = pd.read_csv(
                    url,
                    parse_dates=['Date'],
                    index_col='Date'
                )
                
                if not df.empty:
                    # Convert column names to lowercase to match Backtrader expectations
                    df.columns = df.columns.str.lower()
                    # Set data source attribute
                    df.data_source = 'yahoo_direct'
                    self.logger.info(f"Successfully fetched {len(df)} rows of data for {symbol} via direct download")
                    return df
                    
            except Exception as e:
                self.logger.warning(f"Direct download failed for {symbol}: {str(e)}")
            
            # If direct download fails, try with yfinance as fallback
            self.logger.info(f"Trying yfinance API for {symbol}")
            max_retries = 3
            retry_delay = 1  # seconds
            
            for retry in range(max_retries):
                try:
                    # Create a new ticker for each retry
                    ticker = yf.Ticker(symbol)
                    # Use basic parameters that work with older yfinance versions
                    df = ticker.history(start=start_str, end=end_str)
                    
                    if not df.empty:
                        # Convert column names to lowercase to match Backtrader expectations
                        df.columns = df.columns.str.lower()
                        # Set data source attribute
                        df.data_source = 'yahoo'
                        self.logger.info(f"Successfully fetched {len(df)} rows of data for {symbol} via yfinance")
                        return df
                    else:
                        if retry < max_retries - 1:
                            self.logger.warning(f"Retry {retry+1}/{max_retries}: Empty dataframe from Yahoo Finance for {symbol}")
                            time.sleep(retry_delay)
                        else:
                            self.logger.warning(f"No data found for {symbol} in Yahoo Finance after {max_retries} attempts")
                except Exception as e:
                    if retry < max_retries - 1:
                        self.logger.warning(f"Retry {retry+1}/{max_retries}: Yahoo Finance error for {symbol}: {str(e)}")
                        time.sleep(retry_delay)
                    else:
                        self.logger.warning(f"Failed to fetch {symbol} via yfinance after {max_retries} attempts: {str(e)}")
            
            # If we reached here, all attempts failed
            if use_mock_data:
                self.logger.info(f"Falling back to mock data for {symbol} after all data source attempts failed")
                return self._generate_mock_data(symbol, start_str, end_str)
            else:
                self.logger.error(f"No data available for {symbol} and mock data is disabled. Enable mock data in UI settings to continue.")
                return None
                
        except Exception as e:
            self.logger.error(f"Error fetching stock data for {symbol}: {str(e)}")
            if use_mock_data:
                self.logger.info(f"Falling back to mock data for {symbol} after error")
                return self._generate_mock_data(symbol, start_str, end_str)
            else:
                self.logger.error(f"Cannot fallback to mock data as it is disabled in UI settings. Enable mock data to continue.")
                return None
    
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
                try:
                    url = f"https://api.polygon.io/v2/aggs/ticker/X:{symbol}USD/range/1/day/{start_str}/{end_str}?adjusted=true&sort=asc&limit=500&apiKey={self.polygon_key}"
                    self.logger.info(f"Fetching {symbol} data from Polygon.io")
                    # Add timeout to avoid hanging requests
                    resp = requests.get(url, timeout=10)
                    
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
                            self.logger.warning(f"No results found for crypto {symbol} in Polygon.io response. Response: {data}")
                    else:
                        # For 403 errors, include masked API key info to help with debugging
                        if resp.status_code == 403:
                            # Mask the API key for security (show only first 4 and last 4 chars if long enough)
                            masked_key = self.polygon_key
                            if len(masked_key) > 8:
                                masked_key = masked_key[:4] + '*' * (len(masked_key) - 8) + masked_key[-4:]
                            elif len(masked_key) > 0:
                                masked_key = masked_key[:2] + '*' * (len(masked_key) - 2)
                            
                            error_msg = f"Failed to fetch crypto {symbol} from Polygon.io: HTTP 403 Forbidden. API Key used: {masked_key}"
                            error_msg += ". This typically indicates an invalid or expired API key."
                        else:
                            error_msg = f"Failed to fetch crypto {symbol} from Polygon.io: HTTP {resp.status_code}"
                        
                        try:
                            error_detail = resp.json().get('error', '')
                            if error_detail:
                                error_msg += f", Error: {error_detail}"
                        except:
                            pass
                        self.logger.error(error_msg)
                except requests.exceptions.Timeout:
                    self.logger.error(f"Polygon API timeout for crypto {symbol}. Check your network connection or API status.")
                except requests.exceptions.ConnectionError:
                    self.logger.error(f"Polygon API connection error for crypto {symbol}. Check your network connection.")
                except Exception as e:
                    self.logger.error(f"Polygon API error for crypto {symbol}: {str(e)}")
            
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
            
            try:
                # Add timeout to avoid hanging requests
                resp = requests.get(url, timeout=10)
                
                if resp.ok:
                    return resp.json()
                else:
                    # For 403 errors, include masked API key info to help with debugging
                    if resp.status_code == 403:
                        # Mask the API key for security (show only first 4 and last 4 chars if long enough)
                        masked_key = self.polygon_key
                        if len(masked_key) > 8:
                            masked_key = masked_key[:4] + '*' * (len(masked_key) - 8) + masked_key[-4:]
                        elif len(masked_key) > 0:
                            masked_key = masked_key[:2] + '*' * (len(masked_key) - 2)
                        
                        error_msg = f"Failed to fetch options data for {symbol}: HTTP 403 Forbidden. API Key used: {masked_key}"
                        error_msg += ". This typically indicates an invalid or expired API key."
                    else:
                        error_msg = f"Failed to fetch options data for {symbol}: HTTP {resp.status_code}"
                    
                    try:
                        error_detail = resp.json().get('error', '')
                        if error_detail:
                            error_msg += f", Error: {error_detail}"
                    except:
                        pass
                    self.logger.error(error_msg)
                    return {}
            except requests.exceptions.Timeout:
                self.logger.error(f"Polygon API timeout for options data for {symbol}. Check your network connection or API status.")
                return {}
            except requests.exceptions.ConnectionError:
                self.logger.error(f"Polygon API connection error for options data for {symbol}. Check your network connection.")
                return {}
            except Exception as e:
                self.logger.error(f"Polygon API error for options data for {symbol}: {str(e)}")
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
        
        # Generate price data with more realistic patterns
        np.random.seed(hash(symbol) % 10000)  # Seed based on symbol for consistency
        
        # Create a trend component (long-term direction)
        trend = np.linspace(0, 0.2, len(date_range))  # 20% trend over the period
        
        # Create a cyclical component (market cycles)
        cycle_period = len(date_range) // 3  # Three cycles over the period
        cycle = 0.1 * np.sin(np.linspace(0, cycle_period * np.pi, len(date_range)))
        
        # Create a seasonal component (e.g., quarterly patterns)
        seasonal_period = min(len(date_range) // 4, 60)  # Quarterly or 60-day seasonality
        seasonal = 0.05 * np.sin(np.linspace(0, seasonal_period * 2 * np.pi, len(date_range)))
        
        # Create a random component (daily noise)
        noise = np.random.normal(0, volatility, len(date_range))
        
        # Combine components
        combined_returns = trend + cycle + seasonal + noise
        
        # Generate prices
        prices = [initial_price]
        for ret in combined_returns:
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
