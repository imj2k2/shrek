"""
Robust backtesting engine using Backtrader library
"""
import backtrader as bt
import pandas as pd
import numpy as np
import datetime
from datetime import date, datetime as dt
from typing import List, Dict, Union, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class ShrekStrategy(bt.Strategy):
    """Base Backtrader strategy for the Shrek trading platform"""
    
    params = (
        ('rsi_period', 14),
        ('rsi_overbought', 70),
        ('rsi_oversold', 30),
        ('sma_period_fast', 10),
        ('sma_period_slow', 30),
        ('debug', False),
        ('risk_pct', 0.02),  # Risk 2% per trade
        ('stop_loss', 0.05),  # 5% stop loss
        ('take_profit', 0.10),  # 10% take profit
        ('trailing_stop', False),  # Use trailing stop
        ('momentum_weight', 0.25),
        ('mean_reversion_weight', 0.25),
        ('breakout_weight', 0.25),
        ('volatility_weight', 0.25),
        ('allow_short', False),  # Whether to allow short positions
        ('position_type', None),  # 'long', 'short', or None (auto-determine)
    )
    
    def __init__(self):
        # Initialize indicators
        self.sma_fast = bt.indicators.SMA(self.data.close, 
                                          period=self.params.sma_period_fast)
        self.sma_slow = bt.indicators.SMA(self.data.close, 
                                           period=self.params.sma_period_slow)
        self.rsi = bt.indicators.RSI(self.data.close, 
                                     period=self.params.rsi_period)
        self.atr = bt.indicators.ATR(self.data)
        
        # Track orders
        self.orders = []
        self.trades = []
        
        # Track portfolio values
        self.equity_curve = []
        
        # Track trade management
        self.buy_price = 0
        self.stop_price = 0
        self.target_price = 0
        self.trailing_price = 0
        
    def log(self, txt, dt=None):
        """Logging function for strategy"""
        dt = dt or self.datas[0].datetime.date(0)
        if self.params.debug:
            # Ensure dt is a date object
            if isinstance(dt, float):
                dt = date.fromordinal(int(dt))
            elif not hasattr(dt, 'isoformat'):
                dt = date.today()
            print(f'{dt.isoformat()}: {txt}')
            
    def notify_order(self, order):
        """Called when an order is filled"""
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, {order.executed.price:.2f}')
                self.buy_price = order.executed.price
                # Set stop loss and take profit
                if self.params.stop_loss > 0:
                    self.stop_price = self.buy_price * (1 - self.params.stop_loss)
                if self.params.take_profit > 0:
                    self.target_price = self.buy_price * (1 + self.params.take_profit)
                self.trailing_price = self.buy_price
            elif order.issell():
                self.log(f'SELL EXECUTED, {order.executed.price:.2f}')
                
            # Store order details
            order_date = self.datas[0].datetime.date(0)
            # Ensure date is a proper date object
            if isinstance(order_date, float):
                order_date = date.fromordinal(int(order_date))
            elif not hasattr(order_date, 'isoformat'):
                order_date = date.today()
                
            self.orders.append({
                'date': order_date.isoformat(),
                'symbol': self.data._name,
                'type': 'buy' if order.isbuy() else 'sell',
                'price': order.executed.price,
                'size': order.executed.size,
                'value': order.executed.value,
                'commission': order.executed.comm
            })
                
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'Order Canceled/Margin/Rejected: {order.status}')
            
    def notify_trade(self, trade):
        """Called when a trade is completed"""
        if trade.isclosed:
            self.log(f'TRADE PROFIT, GROSS: {trade.pnl:.2f}, NET: {trade.pnlcomm:.2f}')
            
            # Store trade details
            # Ensure entry and exit dates are proper datetime objects
            entry_date = trade.dtopen
            exit_date = trade.dtclose
            
            if isinstance(entry_date, float):
                entry_date = dt.fromtimestamp(entry_date)
            elif not hasattr(entry_date, 'isoformat'):
                entry_date = dt.now()
                
            if isinstance(exit_date, float):
                exit_date = dt.fromtimestamp(exit_date)
            elif not hasattr(exit_date, 'isoformat'):
                exit_date = dt.now()
                
            self.trades.append({
                'symbol': self.data._name,
                'entry_date': entry_date.isoformat(),
                'exit_date': exit_date.isoformat(),
                'entry_price': trade.price,
                'exit_price': trade.pnlcomm / trade.price if trade.price else 0,
                'profit_loss': trade.pnlcomm,
                'profit_loss_pct': (trade.pnlcomm / trade.price) * 100 if trade.price else 0
            })
            
    def next(self):
        """Main strategy logic executed on each bar"""
        # Get current date and ensure it's a proper date object
        bar_date = self.datas[0].datetime.date(0)
        if isinstance(bar_date, float):
            bar_date = date.fromordinal(int(bar_date))
        elif not hasattr(bar_date, 'isoformat'):
            bar_date = date.today()
            
        # Store portfolio value for equity curve
        self.equity_curve.append({
            'date': bar_date.isoformat(),
            'value': self.broker.getvalue()
        })
        
        # Iterate through each data feed (symbol)
        for i, data in enumerate(self.datas):
            # Skip if we're still in the warmup period
            if len(data) < 30:  # Use a reasonable warmup period
                continue
                
            # Get the current position for this symbol
            pos = self.getposition(data)
            
            # Get the position type for this symbol
            # First check if the engine has a position_types dictionary
            symbol_position_type = self.params.position_type  # Default to strategy parameter
            
            # Try to get the position type from the engine if available
            cerebro = self._env
            if hasattr(cerebro, 'position_types') and data._name in cerebro.position_types:
                symbol_position_type = cerebro.position_types[data._name]
                self.log(f'Using position type {symbol_position_type} for {data._name}')
            
            # Skip if we already have a position in this symbol
            if pos.size != 0:
                # Check if we need to exit based on stop loss or take profit
                if pos.size > 0:  # Long position
                    # For long positions
                    if pos.price > 0:  # Ensure we have a valid entry price
                        # Check stop loss
                        if self.params.stop_loss > 0 and data.close[0] <= pos.price * (1 - self.params.stop_loss):
                            self.log(f'STOP LOSS: {data._name}, Price: {data.close[0]:.2f}')
                            self.close(data)
                        
                        # Check take profit
                        elif self.params.take_profit > 0 and data.close[0] >= pos.price * (1 + self.params.take_profit):
                            self.log(f'TAKE PROFIT: {data._name}, Price: {data.close[0]:.2f}')
                            self.close(data)
                            
                        # Check trailing stop if enabled
                        elif self.params.trailing_stop:
                            if not hasattr(self, 'highest_prices'):
                                self.highest_prices = {}
                                
                            if data._name not in self.highest_prices:
                                self.highest_prices[data._name] = data.close[0]
                            else:
                                # Update highest price seen
                                if data.close[0] > self.highest_prices[data._name]:
                                    self.highest_prices[data._name] = data.close[0]
                                
                                # Check if price has fallen below our trailing stop level
                                trail_price = self.highest_prices[data._name] * (1 - self.params.stop_loss)
                                if data.close[0] <= trail_price:
                                    self.log(f'TRAILING STOP: {data._name}, Price: {data.close[0]:.2f}, High: {self.highest_prices[data._name]:.2f}')
                                    self.close(data)
                elif pos.size < 0:  # Short position
                    # For short positions
                    if pos.price > 0:  # Ensure we have a valid entry price
                        # Check stop loss (price goes up too much)
                        if self.params.stop_loss > 0 and data.close[0] >= pos.price * (1 + self.params.stop_loss):
                            self.log(f'STOP LOSS (SHORT): {data._name}, Price: {data.close[0]:.2f}')
                            self.close(data)
                        
                        # Check take profit (price goes down enough)
                        elif self.params.take_profit > 0 and data.close[0] <= pos.price * (1 - self.params.take_profit):
                            self.log(f'TAKE PROFIT (SHORT): {data._name}, Price: {data.close[0]:.2f}')
                            self.close(data)
                            
                        # Check trailing stop if enabled
                        elif self.params.trailing_stop:
                            if not hasattr(self, 'lowest_prices'):
                                self.lowest_prices = {}
                                
                            if data._name not in self.lowest_prices:
                                self.lowest_prices[data._name] = data.close[0]
                            else:
                                # Update lowest price seen
                                if data.close[0] < self.lowest_prices[data._name]:
                                    self.lowest_prices[data._name] = data.close[0]
                                
                                # Check if price has risen above our trailing stop level
                                trail_price = self.lowest_prices[data._name] * (1 + self.params.stop_loss)
                                if data.close[0] >= trail_price:
                                    self.log(f'TRAILING STOP (SHORT): {data._name}, Price: {data.close[0]:.2f}, Low: {self.lowest_prices[data._name]:.2f}')
                                    self.close(data)
                
                continue
            
            # Calculate signals for this symbol
            # Use the appropriate indicators for this data feed
            sma_fast = bt.indicators.SMA(data.close, period=self.params.sma_period_fast)
            sma_slow = bt.indicators.SMA(data.close, period=self.params.sma_period_slow)
            rsi = bt.indicators.RSI(data.close, period=self.params.rsi_period)
            
            # Calculate signals
            signal = 0
            
            # Momentum strategy
            momentum_signal = 1 if sma_fast[0] > sma_slow[0] else -1 if sma_fast[0] < sma_slow[0] else 0
            signal += momentum_signal * self.params.momentum_weight
            
            # Mean reversion strategy (RSI)
            mean_reversion_signal = 1 if rsi[0] < self.params.rsi_oversold else -1 if rsi[0] > self.params.rsi_overbought else 0
            signal += mean_reversion_signal * self.params.mean_reversion_weight
            
            # If position_type is explicitly set for this symbol, use that direction
            if symbol_position_type == 'short':
                # For short positions, we want to enter when signal is negative
                if signal < -0.5:
                    self.log(f'SHORT ENTRY: {data._name}, Price: {data.close[0]:.2f}, Signal: {signal:.2f}')
                    # Calculate position size based on risk
                    cash = self.broker.getcash()
                    risk_amount = cash * self.params.risk_pct
                    price = data.close[0]
                    size = risk_amount / price
                    
                    # Enter short position
                    self.sell(data=data, size=size)
            elif symbol_position_type == 'long':
                # For long positions, we want to enter when signal is positive
                if signal > 0.5:
                    self.log(f'LONG ENTRY: {data._name}, Price: {data.close[0]:.2f}, Signal: {signal:.2f}')
                    # Calculate position size based on risk
                    cash = self.broker.getcash()
                    risk_amount = cash * self.params.risk_pct
                    price = data.close[0]
                    size = risk_amount / price
                    
                    # Enter long position
                    self.buy(data=data, size=size)
            else:
                # Auto-determine based on signal
                if signal > 0.5:  # Strong bullish signal
                    self.log(f'AUTO LONG ENTRY: {data._name}, Price: {data.close[0]:.2f}, Signal: {signal:.2f}')
                    # Calculate position size based on risk
                    cash = self.broker.getcash()
                    risk_amount = cash * self.params.risk_pct
                    price = data.close[0]
                    size = risk_amount / price
                    
                    # Enter long position
                    self.buy(data=data, size=size)
                elif signal < -0.5 and self.params.allow_short:  # Strong bearish signal and shorts allowed
                    self.log(f'AUTO SHORT ENTRY: {data._name}, Price: {data.close[0]:.2f}, Signal: {signal:.2f}')
                    # Calculate position size based on risk
                    cash = self.broker.getcash()
                    risk_amount = cash * self.params.risk_pct
                    price = data.close[0]
                    size = risk_amount / price
                    
                    # Enter short position
                    self.sell(data=data, size=size)
    
    def _handle_long_strategy(self):
        """Handle long-only strategy"""
        # Only enter if not in a position or in a short position
        if not self.position or self.position.size < 0:
            self._enter_long_position()
    
    def _handle_short_strategy(self):
        """Handle short-only strategy"""
        # Only enter if not in a position or in a long position
        if not self.position or self.position.size > 0:
            self._enter_short_position()
    
    def _enter_long_position(self):
        """Enter a long position"""
        price = self.data.close[0]
        risk_amount = self.broker.getvalue() * self.params.risk_pct
        size = max(1, int(risk_amount / price))
        
        self.log(f'BUY LONG, {price:.2f}')
        self.buy(size=size)
        
        # Set stop loss and take profit for long position
        self.buy_price = price
        if self.params.stop_loss > 0:
            self.stop_price = price * (1 - self.params.stop_loss)
        if self.params.take_profit > 0:
            self.target_price = price * (1 + self.params.take_profit)
        self.trailing_price = price
    
    def _enter_short_position(self):
        """Enter a short position"""
        price = self.data.close[0]
        risk_amount = self.broker.getvalue() * self.params.risk_pct
        size = max(1, int(risk_amount / price))
        
        self.log(f'SELL SHORT, {price:.2f}')
        self.sell(size=size)  # Sell to open short position
        
        # Set stop loss and take profit for short position (opposite direction)
        self.buy_price = price  # Still track entry price
        if self.params.stop_loss > 0:
            self.stop_price = price * (1 + self.params.stop_loss)  # Stop is ABOVE entry for shorts
        if self.params.take_profit > 0:
            self.target_price = price * (1 - self.params.take_profit)  # Target is BELOW entry for shorts
        self.trailing_price = price


class BacktraderEngine:
    """Main backtesting engine using Backtrader"""
    
    def __init__(self, data_fetcher=None, use_mock_data=False):
        self.cerebro = bt.Cerebro()
        self.data_fetcher = data_fetcher
        self.use_mock_data = use_mock_data
        self.logger = logging.getLogger(__name__)
        
    def add_strategy(self, strategy_type='ShrekStrategy', position_type=None, allow_short=False, **kwargs):
        """Add a strategy to the backtest engine
        
        Args:
            strategy_type: Type of strategy to use
            position_type: 'long', 'short', or None (auto-determine)
            allow_short: Whether to allow short positions in auto mode
            **kwargs: Additional strategy parameters
        """
        # Add position type and short selling parameters
        kwargs['position_type'] = position_type
        kwargs['allow_short'] = allow_short
        
        self.logger.info(f"Adding strategy {strategy_type} with position_type={position_type}, allow_short={allow_short}")
        
        if strategy_type == 'ShrekStrategy' or strategy_type == 'customizable_agent':
            self.cerebro.addstrategy(ShrekStrategy, **kwargs)
        # Add other strategy types as needed
        
    def add_data(self, symbols, start_date, end_date, position_types=None):
        """Add price data for the specified symbols
        
        Args:
            symbols: List of symbols to add data for
            start_date: Start date for data
            end_date: End date for data
            position_types: Optional dict mapping symbols to position types ('long' or 'short')
        
        Returns:
            Tuple of (success, data_source)
        """
        data_source = 'real'
        position_types = position_types or {}
        
        # Store position types for each symbol
        self.position_types = {}
        
        for symbol in symbols:
            try:
                # Try to get real market data first
                df = self._get_data(symbol, start_date, end_date)
                
                # Check if data is valid
                if df is None or df.empty:
                    if not self.use_mock_data:
                        self.logger.error(f"No data available for {symbol} and mock data is disabled")
                        return False, f"No data available for {symbol}. Please enable mock data or check API keys."
                    else:
                        self.logger.warning(f"Using mock data for {symbol} as real data is unavailable")
                        df = self._generate_mock_data(symbol, start_date, end_date)
                        data_source = 'mock'
                
                # Convert to Backtrader data feed
                data = self._dataframe_to_bt_feed(df, symbol)
                data.data_source = getattr(df, 'data_source', 'real')
                
                # Store the position type for this symbol
                position_type = position_types.get(symbol, 'long')
                self.position_types[symbol] = position_type
                self.logger.info(f"Setting position type for {symbol} to {position_type}")
                
                # Add data to cerebro
                self.cerebro.adddata(data)
                
            except Exception as e:
                self.logger.error(f"Error adding data for {symbol}: {str(e)}")
                if not self.use_mock_data:
                    return False, f"Error processing data for {symbol}: {str(e)}. Enable mock data to continue."
                else:
                    # Fall back to mock data
                    self.logger.warning(f"Using mock data for {symbol} due to error: {str(e)}")
                    df = self._generate_mock_data(symbol, start_date, end_date)
                    data = self._dataframe_to_bt_feed(df, symbol)
                    data.data_source = 'mock'
                    self.cerebro.adddata(data)
                    data_source = 'mock'
                    
                    # Store the position type for this symbol
                    position_type = position_types.get(symbol, 'long')
                    self.position_types[symbol] = position_type
        
        return True, data_source
        
    def _get_data(self, symbol, start_date, end_date):
        """Get historical price data
        
        Args:
            symbol: Stock symbol
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with price data or None if no data available and mock data is disabled
        """
        try:
            if self.data_fetcher:
                # Pass the use_mock_data flag to the data fetcher
                df = self.data_fetcher.fetch_stock_data(
                    symbol, 
                    start_date, 
                    end_date, 
                    use_mock_data=self.use_mock_data
                )
                
                if df is None:
                    self.logger.error(f"No data available for {symbol} and mock data is disabled")
                    return None
                    
                if df.empty:
                    self.logger.error(f"Empty dataframe returned for {symbol}")
                    return None
                    
                # Log successful data fetch
                self.logger.info(f"Successfully fetched {len(df)} rows of data for {symbol}")
                return df
            else:
                self.logger.error("Data fetcher not initialized")
                return None
        except Exception as e:
            self.logger.error(f"Error in _get_data for {symbol}: {str(e)}")
            return None
        
    def _generate_mock_data(self, symbol, start_date, end_date):
        """Generate realistic mock data that properly represents market behavior
        
        Args:
            symbol: The stock symbol to generate data for
            start_date: Start date string in YYYY-MM-DD format
            end_date: End date string in YYYY-MM-DD format
            
        Returns:
            DataFrame with mock price data that looks realistic
        """
        # Convert date strings to datetime objects
        if isinstance(start_date, str):
            start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
            
        # Create date range for business days (no weekends)
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
        
        # If date range is empty or too short, create at least 10 days
        if len(date_range) < 10:
            self.logger.warning(f"Date range too short for {symbol}, extending to minimum 10 days")
            if isinstance(end_date, datetime.datetime) and isinstance(start_date, datetime.datetime):
                if end_date < start_date:
                    # Swap dates if end is before start
                    start_date, end_date = end_date, start_date
                # Extend end date if needed
                if (end_date - start_date).days < 14:  # Need at least 10 business days
                    end_date = start_date + datetime.timedelta(days=14)
            date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
        
        # Get default start price for symbol
        initial_price = self._get_default_price(symbol)
        volatility = self._get_default_volatility(symbol)
        
        # Generate more realistic price patterns based on symbol characteristics
        trend_direction = 1 if hash(symbol) % 2 == 0 else -1  # Some symbols up, some down
        trend_strength = np.random.uniform(0.1, 0.3)  # Different trend strengths
        
        # 1. Trend component (10-30% change over the period)
        if trend_direction > 0:
            trend = np.linspace(0, trend_strength, len(date_range))
        else:
            trend = np.linspace(0, -trend_strength, len(date_range))
        
        # 2. Cycle component (market cycles - more realistic with varying frequency)
        cycle_count = max(1, len(date_range) // 20)  # More cycles for longer periods
        cycle_phase = hash(symbol) % 100 / 100.0 * 2 * np.pi  # Different starting phases 
        cycle = 0.1 * np.sin(np.linspace(cycle_phase, cycle_phase + cycle_count * 2 * np.pi, len(date_range)))
        
        # 3. Seasonal component (quarterly patterns)
        seasonal_strength = 0.05 * (1 + hash(symbol) % 5 / 10.0)  # Different seasonal effects
        seasonal = seasonal_strength * np.sin(np.linspace(0, 4 * np.pi, len(date_range)))
        
        # 4. Random component (daily noise with volatility clustering)
        # Use GARCH-like volatility clustering
        base_noise = np.random.normal(0, 1, len(date_range))
        # Create volatility clusters
        vol_cluster = np.cumsum(np.random.normal(0, 0.1, len(date_range)))
        vol_cluster = np.exp(vol_cluster - np.mean(vol_cluster)) * volatility
        noise = base_noise * vol_cluster
        
        # 5. Add some price shocks/jumps (random significant moves)
        num_jumps = max(1, len(date_range) // 30)  # More jumps for longer periods
        jump_indices = np.random.choice(range(len(date_range)), size=num_jumps, replace=False)
        jumps = np.zeros(len(date_range))
        for idx in jump_indices:
            jumps[idx] = np.random.normal(0, volatility * 3)  # 3x normal volatility for jumps
        
        # Combine all components
        returns = trend + cycle + seasonal + noise + jumps
        
        # Generate prices (use cumulative returns to ensure no negative prices)
        cum_returns = np.cumsum(returns)
        # Scale the cumulative returns to reasonable levels
        cum_returns = cum_returns / np.max(np.abs(cum_returns)) * trend_strength
        prices = initial_price * np.exp(cum_returns)
        
        # Create DataFrame with OHLC data
        df = pd.DataFrame(index=date_range)
        df['close'] = prices
        # Create realistic open/high/low based on close
        df['open'] = df['close'].shift(1)
        df.loc[df.index[0], 'open'] = prices[0] * (1 + np.random.normal(0, volatility / 2))
        
        # Daily range varies with volatility
        daily_ranges = vol_cluster * 1.5  # High-low range scales with volatility
        df['high'] = df['close'] * (1 + daily_ranges / 2)
        df['low'] = df['close'] * (1 - daily_ranges / 2)
        
        # Ensure high is always highest and low is always lowest
        for i in range(len(df)):
            high_val = max(df['open'].iloc[i], df['close'].iloc[i]) * (1 + np.random.uniform(0.001, 0.015))
            low_val = min(df['open'].iloc[i], df['close'].iloc[i]) * (1 - np.random.uniform(0.001, 0.015))
            df['high'].iloc[i] = high_val
            df['low'].iloc[i] = low_val
        
        # Generate volume that correlates with volatility
        base_volume = self._get_default_volume(symbol)
        volume_volatility_effect = 1 + 2 * np.abs(returns)  # More volume on bigger moves
        df['volume'] = base_volume * volume_volatility_effect
        df['volume'] = df['volume'].astype(int)
        
        # Mark this as mock data
        df.data_source = 'mock'
        return df
        
    def _dataframe_to_bt_feed(self, df, name):
        """Convert pandas DataFrame to Backtrader DataFeed"""
        data = bt.feeds.PandasData(
            dataname=df,
            name=name,
            datetime=None,  # Assuming datetime is the index
            open=0,  # Column position or name
            high=1,
            low=2,
            close=3,
            volume=4,
            openinterest=-1  # -1 means not present
        )
        return data
        
    def _get_default_price(self, symbol):
        """Get default price for a symbol"""
        default_prices = {
            'AAPL': 175.0,
            'MSFT': 350.0,
            'GOOG': 140.0,
            'AMZN': 130.0,
            'META': 300.0,
            'TSLA': 250.0,
            'NVDA': 450.0,
            'SPY': 430.0,
            'QQQ': 380.0,
            'BTC': 35000.0,
            'ETH': 2000.0,
        }
        return default_prices.get(symbol, 100.0)
        
    def _get_default_volatility(self, symbol):
        """Get default volatility for a symbol"""
        default_volatility = {
            'AAPL': 0.015,
            'MSFT': 0.012,
            'GOOG': 0.018,
            'AMZN': 0.020,
            'META': 0.025,
            'TSLA': 0.035,
            'NVDA': 0.030,
            'SPY': 0.010,
            'QQQ': 0.012,
            'BTC': 0.040,
            'ETH': 0.045,
        }
        return default_volatility.get(symbol, 0.02)
    
    def _get_default_volume(self, symbol):
        """Get default trading volume for a symbol"""
        default_volume = {
            'AAPL': 80000000,
            'MSFT': 30000000,
            'GOOG': 20000000,
            'AMZN': 40000000,
            'META': 25000000,
            'TSLA': 100000000,
            'NVDA': 50000000,
            'SPY': 70000000,
            'QQQ': 50000000,
            'BTC': 5000000,
            'ETH': 3000000,
        }
        return default_volume.get(symbol, 10000000)
        
    def run_backtest(self, initial_cash=100000.0):
        """Run the backtest"""
        # Set starting cash
        self.cerebro.broker.setcash(initial_cash)
        
        # Add analyzers
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        self.cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
        
        # Run backtest
        results = self.cerebro.run()
        
        # Process results
        if not results:
            return {
                'success': False,
                'error': 'Backtest failed to run properly'
            }
            
        strategy = results[0]
        
        # Get equity curve
        equity_curve = strategy.equity_curve
        
        # Get trades
        trades = strategy.trades
        
        # Get metrics from analyzers
        sharpe = strategy.analyzers.sharpe.get_analysis().get('sharperatio', 0.0)
        drawdown = strategy.analyzers.drawdown.get_analysis()
        trade_analysis = strategy.analyzers.trades.get_analysis()
        returns = strategy.analyzers.returns.get_analysis()
        
        # Format metrics
        metrics = {
            'initial_value': initial_cash,
            'final_value': self.cerebro.broker.getvalue(),
            'total_return': (self.cerebro.broker.getvalue() / initial_cash - 1) * 100,
            'sharpe_ratio': sharpe,
            'max_drawdown': drawdown.get('max', {}).get('drawdown', 0) * 100,
            'win_rate': self._calculate_win_rate(trade_analysis),
            'profit_factor': self._calculate_profit_factor(trade_analysis),
            'total_trades': trade_analysis.get('total', {}).get('total', 0),
        }
        
        # Check data source
        data_source = 'real'
        for data in strategy.datas:
            if hasattr(data, 'data_source') and data.data_source == 'mock':
                data_source = 'mock'
                break
        
        # Final results
        return {
            'equity_curve': equity_curve,
            'trades': trades,
            'metrics': metrics,
            'data_source': data_source,
            'success': True
        }
        
    def _calculate_win_rate(self, trade_analysis):
        """Calculate win rate from trade analysis"""
        won = trade_analysis.get('won', {}).get('total', 0)
        total = trade_analysis.get('total', {}).get('total', 0)
        return (won / total * 100) if total > 0 else 0
        
    def _calculate_profit_factor(self, trade_analysis):
        """Calculate profit factor from trade analysis"""
        won = trade_analysis.get('won', {}).get('pnl', {}).get('total', 0)
        lost = abs(trade_analysis.get('lost', {}).get('pnl', {}).get('total', 0))
        return (won / lost) if lost > 0 else 0
