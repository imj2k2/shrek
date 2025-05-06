"""
Stock screener module for the Shrek trading platform
Provides functionality to screen stocks based on various criteria
"""
import pandas as pd
import numpy as np
import yfinance as yf
import logging
import os
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class StockScreener:
    """Stock screener with customizable criteria"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.polygon_key = os.getenv("POLYGON_API_KEY", "")
        
    def screen_stocks(self, criteria, universe=None, max_stocks=50):
        """
        Screen stocks based on specified criteria
        
        Args:
            criteria: Dict of screening criteria
            universe: List of stock symbols to screen (if None, uses default universe)
            max_stocks: Maximum number of stocks to return
            
        Returns:
            DataFrame with screened stocks and their data
        """
        try:
            # Get universe of stocks to screen
            if universe is None or len(universe) == 0:
                universe = self._get_default_universe()
                
            self.logger.info(f"Screening {len(universe)} stocks with criteria: {criteria}")
            
            # Fetch data for all stocks in universe
            stock_data = self._fetch_stock_data(universe)
            
            # Apply screening criteria
            screened_stocks = self._apply_criteria(stock_data, criteria)
            
            # Limit to max_stocks
            if len(screened_stocks) > max_stocks:
                screened_stocks = screened_stocks[:max_stocks]
                
            self.logger.info(f"Screening complete. Found {len(screened_stocks)} matching stocks.")
            return screened_stocks
            
        except Exception as e:
            self.logger.error(f"Error in stock screening: {str(e)}")
            return pd.DataFrame()
            
    def _get_default_universe(self):
        """Get a default universe of stocks to screen"""
        try:
            # Try to get S&P 500 components
            sp500 = self._get_sp500_symbols()
            if len(sp500) > 0:
                return sp500
                
            # Fallback to a predefined list of major stocks
            return [
                "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "JPM", 
                "V", "JNJ", "WMT", "PG", "MA", "UNH", "HD", "BAC", "XOM", "PFE",
                "CSCO", "VZ", "NFLX", "ADBE", "CRM", "INTC", "CMCSA", "PEP", "KO",
                "ABT", "TMO", "ACN", "AVGO", "MRK", "DHR", "NKE", "TXN", "NEE", 
                "LLY", "QCOM", "MDT", "UNP", "HON", "PM", "T", "LIN", "ORCL", 
                "COST", "CVX", "IBM", "AMD", "PYPL", "SBUX", "GS"
            ]
        except Exception as e:
            self.logger.error(f"Error getting default universe: {str(e)}")
            # Return a minimal set of stocks if everything else fails
            return ["AAPL", "MSFT", "AMZN", "GOOGL", "META"]
            
    def _get_sp500_symbols(self):
        """Get current S&P 500 components"""
        try:
            # Try to get S&P 500 components from Wikipedia
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            df = tables[0]
            return df['Symbol'].tolist()
        except Exception as e:
            self.logger.error(f"Error fetching S&P 500 symbols: {str(e)}")
            return []
            
    def _fetch_stock_data(self, symbols, lookback_days=30):
        """
        Fetch data for multiple stocks
        
        Args:
            symbols: List of stock symbols
            lookback_days: Number of days of historical data to fetch
            
        Returns:
            Dict mapping symbols to their data DataFrames
        """
        stock_data = {}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        # Use ThreadPoolExecutor for parallel fetching
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Submit tasks for each symbol
            futures = {executor.submit(self._fetch_single_stock, symbol, start_date, end_date): symbol for symbol in symbols}
            
            # Process results as they complete
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    data = future.result()
                    if data is not None and not data.empty:
                        stock_data[symbol] = data
                except Exception as e:
                    self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
        
        return stock_data
        
    def _fetch_single_stock(self, symbol, start_date, end_date):
        """Fetch data for a single stock"""
        try:
            # Try yfinance first
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            
            if not df.empty:
                # Calculate additional metrics
                df = self._calculate_metrics(df)
                return df
                
            return None
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
            
    def _calculate_metrics(self, df):
        """Calculate additional metrics for screening"""
        try:
            # Ensure dataframe has required columns
            if 'Close' not in df.columns:
                df['Close'] = df['close'] if 'close' in df.columns else None
            if 'Volume' not in df.columns:
                df['Volume'] = df['volume'] if 'volume' in df.columns else None
                
            # Calculate daily returns
            df['daily_return'] = df['Close'].pct_change()
            
            # Calculate volatility (20-day)
            if len(df) >= 20:
                df['volatility_20d'] = df['daily_return'].rolling(window=20).std() * np.sqrt(252)
            
            # Calculate moving averages
            df['sma_10'] = df['Close'].rolling(window=10).mean()
            df['sma_50'] = df['Close'].rolling(window=50).mean()
            df['sma_200'] = df['Close'].rolling(window=200).mean()
            
            # Calculate RSI (14-day)
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            
            # Avoid division by zero
            rs = gain / loss.replace(0, np.nan)
            df['rsi_14'] = 100 - (100 / (1 + rs))
            
            # Calculate MACD
            df['ema_12'] = df['Close'].ewm(span=12, adjust=False).mean()
            df['ema_26'] = df['Close'].ewm(span=26, adjust=False).mean()
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # Calculate Bollinger Bands (20-day, 2 std)
            if len(df) >= 20:
                df['bb_middle'] = df['Close'].rolling(window=20).mean()
                df['bb_std'] = df['Close'].rolling(window=20).std()
                df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
                df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
                
            # Calculate average volume
            df['avg_volume_20d'] = df['Volume'].rolling(window=20).mean()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            return df
            
    def _apply_criteria(self, stock_data, criteria):
        """
        Apply screening criteria to stock data
        
        Args:
            stock_data: Dict mapping symbols to their data DataFrames
            criteria: Dict of screening criteria
            
        Returns:
            DataFrame with screened stocks and their data
        """
        results = []
        
        for symbol, df in stock_data.items():
            try:
                # Skip if dataframe is empty or too short
                if df is None or df.empty or len(df) < 2:
                    continue
                    
                # Get the most recent data point
                latest = df.iloc[-1]
                
                # Initialize as True - stock passes unless it fails a criterion
                passes_screen = True
                
                # Price criteria
                if 'min_price' in criteria and criteria['min_price'] is not None:
                    if latest['Close'] < criteria['min_price']:
                        passes_screen = False
                        
                if 'max_price' in criteria and criteria['max_price'] is not None:
                    if latest['Close'] > criteria['max_price']:
                        passes_screen = False
                
                # Volume criteria
                if 'min_volume' in criteria and criteria['min_volume'] is not None:
                    if latest['Volume'] < criteria['min_volume']:
                        passes_screen = False
                
                # Volatility criteria
                if 'min_volatility' in criteria and criteria['min_volatility'] is not None:
                    if 'volatility_20d' in latest and not np.isnan(latest['volatility_20d']):
                        if latest['volatility_20d'] < criteria['min_volatility']:
                            passes_screen = False
                            
                if 'max_volatility' in criteria and criteria['max_volatility'] is not None:
                    if 'volatility_20d' in latest and not np.isnan(latest['volatility_20d']):
                        if latest['volatility_20d'] > criteria['max_volatility']:
                            passes_screen = False
                
                # Moving average criteria
                if 'price_above_sma50' in criteria and criteria['price_above_sma50']:
                    if 'sma_50' in latest and not np.isnan(latest['sma_50']):
                        if latest['Close'] <= latest['sma_50']:
                            passes_screen = False
                            
                if 'price_below_sma50' in criteria and criteria['price_below_sma50']:
                    if 'sma_50' in latest and not np.isnan(latest['sma_50']):
                        if latest['Close'] >= latest['sma_50']:
                            passes_screen = False
                
                if 'price_above_sma200' in criteria and criteria['price_above_sma200']:
                    if 'sma_200' in latest and not np.isnan(latest['sma_200']):
                        if latest['Close'] <= latest['sma_200']:
                            passes_screen = False
                            
                if 'price_below_sma200' in criteria and criteria['price_below_sma200']:
                    if 'sma_200' in latest and not np.isnan(latest['sma_200']):
                        if latest['Close'] >= latest['sma_200']:
                            passes_screen = False
                
                # RSI criteria
                if 'min_rsi' in criteria and criteria['min_rsi'] is not None:
                    if 'rsi_14' in latest and not np.isnan(latest['rsi_14']):
                        if latest['rsi_14'] < criteria['min_rsi']:
                            passes_screen = False
                            
                if 'max_rsi' in criteria and criteria['max_rsi'] is not None:
                    if 'rsi_14' in latest and not np.isnan(latest['rsi_14']):
                        if latest['rsi_14'] > criteria['max_rsi']:
                            passes_screen = False
                
                # MACD criteria
                if 'macd_positive' in criteria and criteria['macd_positive']:
                    if 'macd' in latest and not np.isnan(latest['macd']):
                        if latest['macd'] <= 0:
                            passes_screen = False
                            
                if 'macd_negative' in criteria and criteria['macd_negative']:
                    if 'macd' in latest and not np.isnan(latest['macd']):
                        if latest['macd'] >= 0:
                            passes_screen = False
                
                # If the stock passes all criteria, add it to results
                if passes_screen:
                    # Create a result row with key metrics
                    result = {
                        'Symbol': symbol,
                        'Price': latest['Close'],
                        'Volume': latest['Volume'],
                        'Daily_Change_%': latest.get('daily_return', 0) * 100 if 'daily_return' in latest else 0,
                    }
                    
                    # Add additional metrics if available
                    if 'volatility_20d' in latest and not np.isnan(latest['volatility_20d']):
                        result['Volatility_20d'] = latest['volatility_20d'] * 100
                        
                    if 'rsi_14' in latest and not np.isnan(latest['rsi_14']):
                        result['RSI_14'] = latest['rsi_14']
                        
                    if 'macd' in latest and not np.isnan(latest['macd']):
                        result['MACD'] = latest['macd']
                        
                    results.append(result)
                    
            except Exception as e:
                self.logger.error(f"Error applying criteria to {symbol}: {str(e)}")
                continue
        
        # Convert results to DataFrame and sort by criteria
        if results:
            results_df = pd.DataFrame(results)
            
            # Sort by specified field if provided
            if 'sort_by' in criteria and criteria['sort_by'] in results_df.columns:
                ascending = criteria.get('sort_ascending', False)
                results_df = results_df.sort_values(by=criteria['sort_by'], ascending=ascending)
                
            return results_df
        else:
            return pd.DataFrame()
            
    def get_fundamental_data(self, symbols):
        """
        Get fundamental data for a list of symbols
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            DataFrame with fundamental data
        """
        results = []
        
        for symbol in symbols:
            try:
                # Use yfinance to get fundamental data
                ticker = yf.Ticker(symbol)
                
                # Get basic info
                info = ticker.info
                
                # Create a result row with key metrics
                result = {
                    'Symbol': symbol,
                    'Name': info.get('shortName', 'N/A'),
                    'Sector': info.get('sector', 'N/A'),
                    'Industry': info.get('industry', 'N/A'),
                    'Market_Cap': info.get('marketCap', 0),
                    'P/E': info.get('trailingPE', 0),
                    'Forward_P/E': info.get('forwardPE', 0),
                    'PEG_Ratio': info.get('pegRatio', 0),
                    'P/S': info.get('priceToSalesTrailing12Months', 0),
                    'P/B': info.get('priceToBook', 0),
                    'EPS': info.get('trailingEps', 0),
                    'Dividend_Yield_%': info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
                    'Beta': info.get('beta', 0),
                    '52W_High': info.get('fiftyTwoWeekHigh', 0),
                    '52W_Low': info.get('fiftyTwoWeekLow', 0),
                }
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error getting fundamental data for {symbol}: {str(e)}")
                continue
        
        # Convert results to DataFrame
        if results:
            return pd.DataFrame(results)
        else:
            return pd.DataFrame()
