from indicators import rsi, macd, bollinger, atr, vwap, moving_averages
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from mcp.protocol import MCPMessage
from datetime import datetime
import logging

class StocksAgent:
    """Agent for stock trading strategies including momentum, mean reversion, and breakout"""
    
    def __init__(self, mcp=None):
        self.logger = logging.getLogger("StocksAgent")
        self.mcp = mcp  # Model Context Protocol for communication
        
        # Strategy parameters
        self.strategies = {
            "momentum": {
                "enabled": True,
                "weight": 0.4,  # Strategy weight in overall decision
                "params": {
                    "rsi_period": 14,
                    "rsi_overbought": 70,
                    "rsi_oversold": 30,
                    "macd_fast": 12,
                    "macd_slow": 26,
                    "macd_signal": 9
                }
            },
            "mean_reversion": {
                "enabled": True,
                "weight": 0.3,
                "params": {
                    "bollinger_period": 20,
                    "bollinger_std": 2,
                    "vwap_period": 14
                }
            },
            "breakout": {
                "enabled": True,
                "weight": 0.3,
                "params": {
                    "atr_period": 14,
                    "atr_multiplier": 2.0,
                    "ma_periods": [20, 50, 200]
                }
            }
        }
        
        # Position sizing parameters
        self.position_sizing = {
            "max_position_pct": 0.05,  # Max 5% of portfolio in any single position
            "risk_per_trade_pct": 0.01  # Risk 1% of portfolio per trade
        }
        
        # Tracking variables
        self.last_signals = {}
        self.performance_metrics = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0
        }
    
    def generate_signals(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trading signals based on multiple strategies"""
        if not data or 'close' not in data:
            self.logger.error("Invalid data format: missing 'close' prices")
            return {'error': 'Invalid data format: missing required price data'}
        
        try:
            # Convert data to pandas Series if it's a list
            close = pd.Series(data['close']) if isinstance(data['close'], list) else data['close']
            high = pd.Series(data['high']) if 'high' in data and isinstance(data['high'], list) else None
            low = pd.Series(data['low']) if 'low' in data and isinstance(data['low'], list) else None
            volume = pd.Series(data['volume']) if 'volume' in data and isinstance(data['volume'], list) else None
            symbol = data.get('symbol', 'UNKNOWN')
            
            # Get the current price for signal generation
            current_price = close.iloc[-1] if hasattr(close, 'iloc') else close[-1]
            
            # Check if we are running as value_agent (backward compatibility)
            agent_type = data.get('agent_type', None)
            
            if agent_type == 'value_agent':
                # Use the dedicated value strategy
                value_signals = self._value_strategy(data)
                combined_signals = {'action': 'hold', 'buy_signals': [], 'sell_signals': []}
                
                for signal_id, signal in value_signals.items():
                    if signal['action'] == 'buy':
                        combined_signals['buy_signals'].append(signal)
                    elif signal['action'] == 'sell':
                        combined_signals['sell_signals'].append(signal)
                
                # Determine overall action
                if combined_signals['buy_signals']:
                    combined_signals['action'] = 'buy'
                    combined_signals['buy_strength'] = max(s['strength'] for s in combined_signals['buy_signals'])
                    combined_signals['sell_strength'] = 0
                    combined_signals['net_strength'] = combined_signals['buy_strength']
                elif combined_signals['sell_signals']:
                    combined_signals['action'] = 'sell'
                    combined_signals['sell_strength'] = max(s['strength'] for s in combined_signals['sell_signals'])
                    combined_signals['buy_strength'] = 0
                    combined_signals['net_strength'] = -combined_signals['sell_strength']
            else:
                # Regular mode - use all enabled strategies
                momentum_signals = self._momentum_strategy(close) if self.strategies["momentum"]["enabled"] else {}
                mean_reversion_signals = self._mean_reversion_strategy(close, high, low) if self.strategies["mean_reversion"]["enabled"] else {}
                breakout_signals = self._breakout_strategy(close, high, low, volume) if self.strategies["breakout"]["enabled"] else {}
                
                # Combine signals with weights
                combined_signals = self._combine_signals({
                    "momentum": momentum_signals,
                    "mean_reversion": mean_reversion_signals,
                    "breakout": breakout_signals
                })
            
            # Add position sizing
            final_signals = self._apply_position_sizing(combined_signals, data)
            
            # Ensure we have quantity information for the backtest engine
            if final_signals['action'] in ['buy', 'sell'] and 'qty' not in final_signals:
                # Default to 10 shares if no quantity specified
                final_signals['qty'] = 10
                
            # Ensure we have a price for the backtest engine
            if 'price' not in final_signals:
                final_signals['price'] = current_price
                
            # Add strategy information for tracking
            if 'strategy' not in final_signals:
                if final_signals['action'] == 'buy' and final_signals.get('buy_signals'):
                    # Use the strongest buy signal's strategy
                    strongest_signal = max(final_signals['buy_signals'], key=lambda x: x['strength']) if final_signals['buy_signals'] else None
                    final_signals['strategy'] = strongest_signal.get('strategy', 'combined') if strongest_signal else 'combined'
                elif final_signals['action'] == 'sell' and final_signals.get('sell_signals'):
                    # Use the strongest sell signal's strategy
                    strongest_signal = max(final_signals['sell_signals'], key=lambda x: x['strength']) if final_signals['sell_signals'] else None
                    final_signals['strategy'] = strongest_signal.get('strategy', 'combined') if strongest_signal else 'combined'
                else:
                    final_signals['strategy'] = 'combined'
            
            # Add metadata
            final_signals['timestamp'] = datetime.now().isoformat()
            final_signals['symbol'] = symbol
            
            # Store last signals
            self.last_signals = final_signals
            
            # Send to MCP if available
            if self.mcp:
                message = MCPMessage(
                    sender="stocks_agent",
                    message_type="signal",
                    content=final_signals
                )
                self.mcp.send_message(message)
            
            return final_signals
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {str(e)}")
            return {'error': f'Signal generation failed: {str(e)}'}
    
    def _momentum_strategy(self, close: pd.Series) -> Dict[str, Any]:
        """Momentum strategy using RSI and MACD"""
        signals = {}
        
        # Convert to pandas Series if it's a numpy array
        if isinstance(close, np.ndarray):
            close = pd.Series(close)
        
        # Get parameters
        params = self.strategies["momentum"]["params"]
        
        # Calculate RSI
        if len(close) >= params["rsi_period"]:
            # Calculate price changes
            delta = close.diff()
            
            # Separate gains and losses
            gain = delta.copy()
            loss = delta.copy()
            gain[gain < 0] = 0
            loss[loss > 0] = 0
            loss = -loss  # Make losses positive
            
            # Calculate average gain and loss
            avg_gain = gain.rolling(window=params["rsi_period"]).mean()
            avg_loss = loss.rolling(window=params["rsi_period"]).mean()
            
            # Calculate RS and RSI
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            # Get the latest RSI value
            latest_rsi = rsi.iloc[-1] if not rsi.empty else None
            
            if latest_rsi is not None:
                # Oversold condition (potential buy) - Make more aggressive
                if latest_rsi < params["rsi_oversold"] + 10:  # Increase threshold to generate more signals
                    # Scale strength based on how oversold
                    strength = 0.5 + ((params["rsi_oversold"] + 10 - latest_rsi) / 40)
                    signals["rsi_oversold"] = {
                        "action": "buy",
                        "strength": min(0.9, strength),  # Cap at 0.9
                        "reason": f"RSI oversold ({latest_rsi:.2f})",
                        "strategy": "momentum"
                    }
                # Overbought condition (potential sell) - Make more aggressive
                elif latest_rsi > params["rsi_overbought"] - 10:  # Decrease threshold to generate more signals
                    # Scale strength based on how overbought
                    strength = 0.5 + ((latest_rsi - (params["rsi_overbought"] - 10)) / 40)
                    signals["rsi_overbought"] = {
                        "action": "sell",
                        "strength": min(0.9, strength),  # Cap at 0.9
                        "reason": f"RSI overbought ({latest_rsi:.2f})",
                        "strategy": "momentum"
                    }
        
        # MACD Strategy
        try:
            if len(close) >= params["macd_slow"]:
                # Calculate EMAs
                ema_fast = close.ewm(span=params["macd_fast"], adjust=False).mean()
                ema_slow = close.ewm(span=params["macd_slow"], adjust=False).mean()
                
                # Calculate MACD line and signal line
                macd_line = ema_fast - ema_slow
                signal_line = macd_line.ewm(span=params["macd_signal"], adjust=False).mean()
                
                # Get the latest values
                latest_macd = macd_line.iloc[-1] if not macd_line.empty else None
                latest_signal = signal_line.iloc[-1] if not signal_line.empty else None
            
                if latest_macd is not None and latest_signal is not None:
                    # MACD crosses above signal line (potential buy)
                    if len(macd_line) > 1 and len(signal_line) > 1:
                        prev_macd = macd_line.iloc[-2]
                        prev_signal = signal_line.iloc[-2]
                        
                        # Calculate distance between MACD and signal line
                        macd_distance = abs(latest_macd - latest_signal)
                        norm_distance = macd_distance / abs(latest_signal) if latest_signal != 0 else 0
                    
                    # MACD crosses or is about to cross above signal line (potential buy)
                    if (prev_macd < prev_signal and latest_macd > latest_signal) or \
                       (latest_macd < latest_signal and (latest_signal - latest_macd) / abs(latest_signal) < 0.05):
                        signals["macd_cross_above"] = {
                            "action": "buy",
                            "strength": min(0.8, 0.6 + norm_distance),
                            "reason": "MACD crossed above signal line",
                            "strategy": "momentum"
                        }
                    # MACD crosses or is about to cross below signal line (potential sell)
                    elif (prev_macd > prev_signal and latest_macd < latest_signal) or \
                         (latest_macd > latest_signal and (latest_macd - latest_signal) / abs(latest_signal) < 0.05):
                        signals["macd_cross_below"] = {
                            "action": "sell",
                            "strength": min(0.8, 0.6 + norm_distance),
                            "reason": "MACD crossed below signal line",
                            "strategy": "momentum"
                        }
                        
                    # MACD is positive and increasing (bullish momentum)
                    if latest_macd > 0 and len(macd_line) > 5 and latest_macd > macd_line.iloc[-5]:
                        signals["macd_bullish"] = {
                            "action": "buy",
                            "strength": 0.5,
                            "reason": "MACD is positive and increasing",
                            "strategy": "momentum"
                        }
                        
                    # MACD is negative and decreasing (bearish momentum)
                    elif latest_macd < 0 and len(macd_line) > 5 and latest_macd < macd_line.iloc[-5]:
                        signals["macd_bearish"] = {
                            "action": "sell",
                            "strength": 0.5,
                            "reason": "MACD is negative and decreasing",
                            "strategy": "momentum"
                        }
        except Exception as e:
            self.logger.warning(f"MACD calculation failed: {str(e)}")
        
        
        return signals
    
    def _mean_reversion_strategy(self, close: pd.Series, high: Optional[pd.Series] = None, low: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Mean reversion strategy using Bollinger Bands and VWAP"""
        signals = {}
        params = self.strategies["mean_reversion"]["params"]
        
        # Bollinger Bands Strategy
        try:
            # Convert to pandas Series if it's a numpy array
            if isinstance(close, np.ndarray):
                close = pd.Series(close)
                
            if len(close) >= params["bollinger_period"]:
                # Calculate SMA for middle band
                middle = close.rolling(window=params["bollinger_period"]).mean()
                
                # Calculate standard deviation
                stddev = close.rolling(window=params["bollinger_period"]).std()
                
                # Calculate upper and lower bands
                upper = middle + (stddev * params["bollinger_std"])
                lower = middle - (stddev * params["bollinger_std"])
                
                # Drop NaN values
                upper = upper.dropna()
                middle = middle.dropna()
                lower = lower.dropna()
            
            if not upper.empty and not lower.empty:
                last_close = close.iloc[-1]
                last_upper = upper.iloc[-1]
                last_lower = lower.iloc[-1]
                last_middle = middle.iloc[-1]
                
                # Price below lower band (potential buy)
                if last_close < last_lower:
                    # Calculate how far below the band (normalized)
                    band_width = last_upper - last_lower
                    distance = (last_lower - last_close) / band_width if band_width > 0 else 0
                    strength = min(0.9, 0.5 + distance)  # Cap at 0.9
                    
                    signals['bollinger_lower'] = {
                        'action': 'buy',
                        'strength': strength,
                        'reason': f'Price below lower Bollinger Band',
                        'distance': distance,
                        'close': last_close,
                        'lower_band': last_lower
                    }
                
                # Price above upper band (potential sell)
                elif last_close > last_upper:
                    # Calculate how far above the band (normalized)
                    band_width = last_upper - last_lower
                    distance = (last_close - last_upper) / band_width if band_width > 0 else 0
                    strength = min(0.9, 0.5 + distance)  # Cap at 0.9
                    
                    signals['bollinger_upper'] = {
                        'action': 'sell',
                        'strength': strength,
                        'reason': f'Price above upper Bollinger Band',
                        'distance': distance,
                        'close': last_close,
                        'upper_band': last_upper
                    }
        except Exception as e:
            self.logger.warning(f"Bollinger Bands calculation failed: {str(e)}")
        
        # VWAP Strategy (if we have high, low, and volume data)
        # Note: 'data' is not defined in this function, so we'll check if we have high and low data
        if high is not None and low is not None:
            try:
                # Simple implementation of VWAP calculation
                if len(close) >= params["vwap_period"]:
                    # Calculate typical price (TP): (High + Low + Close) / 3
                    typical_price = (high + low + close) / 3
                    
                    # For volume, we'll use a dummy volume if not available
                    # In a real implementation, volume should be passed as a parameter
                    volume = pd.Series(np.ones(len(close)), index=close.index)
                    
                    # Calculate VWAP: sum(TP * Volume) / sum(Volume)
                    tp_vol = typical_price * volume
                    vol_sum = volume.rolling(window=params["vwap_period"]).sum()
                    tp_vol_sum = tp_vol.rolling(window=params["vwap_period"]).sum()
                    vwap_values = tp_vol_sum / vol_sum
                    vwap_values = vwap_values.dropna()
                
                if not vwap_values.empty:
                    last_vwap = vwap_values.iloc[-1]
                    last_close = close.iloc[-1]
                    
                    # Price significantly below VWAP (potential buy)
                    if last_close < last_vwap * 0.98:  # 2% below VWAP
                        signals['vwap_below'] = {
                            'action': 'buy',
                            'strength': 0.6,
                            'reason': f'Price below VWAP',
                            'close': last_close,
                            'vwap': last_vwap
                        }
                    
                    # Price significantly above VWAP (potential sell)
                    elif last_close > last_vwap * 1.02:  # 2% above VWAP
                        signals['vwap_above'] = {
                            'action': 'sell',
                            'strength': 0.6,
                            'reason': f'Price above VWAP',
                            'close': last_close,
                            'vwap': last_vwap
                        }
            except Exception as e:
                self.logger.warning(f"VWAP calculation failed: {str(e)}")
        
        return signals
    
    def _breakout_strategy(self, close: pd.Series, high: Optional[pd.Series] = None, low: Optional[pd.Series] = None, volume: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Breakout strategy using ATR and Moving Averages"""
        signals = {}
        params = self.strategies["breakout"]["params"]
        
        # ATR for volatility-based breakouts
        if high is not None and low is not None:
            try:
                # Convert to pandas Series if they're numpy arrays
                if isinstance(close, np.ndarray):
                    close = pd.Series(close)
                if isinstance(high, np.ndarray):
                    high = pd.Series(high, index=close.index)
                if isinstance(low, np.ndarray):
                    low = pd.Series(low, index=close.index)
                
                if len(close) >= params["atr_period"]:
                    # Calculate True Range
                    tr1 = high - low  # Current high - current low
                    tr2 = abs(high - close.shift(1))  # Current high - previous close
                    tr3 = abs(low - close.shift(1))  # Current low - previous close
                    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                    
                    # Calculate ATR as simple moving average of true range
                    atr_values = tr.rolling(window=params["atr_period"]).mean()
                    atr_values = atr_values.dropna()
                
                if not atr_values.empty and len(close) > 1:
                    last_atr = atr_values.iloc[-1]
                    last_close = close.iloc[-1]
                    prev_close = close.iloc[-2]
                    
                    # Significant price move up (> ATR * multiplier)
                    if last_close > prev_close + (last_atr * params["atr_multiplier"]):
                        signals['atr_breakout_up'] = {
                            'action': 'buy',
                            'strength': 0.7,
                            'reason': f'Volatility breakout up (ATR)',
                            'atr': last_atr,
                            'move_size': last_close - prev_close
                        }
                    
                    # Significant price move down (> ATR * multiplier)
                    elif last_close < prev_close - (last_atr * params["atr_multiplier"]):
                        signals['atr_breakout_down'] = {
                            'action': 'sell',
                            'strength': 0.7,
                            'reason': f'Volatility breakout down (ATR)',
                            'atr': last_atr,
                            'move_size': prev_close - last_close
                        }
            except Exception as e:
                self.logger.warning(f"ATR calculation failed: {str(e)}")
        
        # Moving Average crossovers for trend breakouts
        try:
            # Convert to pandas Series if it's a numpy array
            if isinstance(close, np.ndarray):
                close = pd.Series(close)
                
            for period in params["ma_periods"]:
                if len(close) >= period:
                    # Calculate Simple Moving Average
                    ma_values = close.rolling(window=period).mean()
                    ma_values = ma_values.dropna()
                
                if not ma_values.empty and len(close) > 1 and len(ma_values) > 1:
                    last_ma = ma_values.iloc[-1]
                    prev_ma = ma_values.iloc[-2]
                    last_close = close.iloc[-1]
                    prev_close = close.iloc[-2]
                    
                    # Price crosses above MA (bullish)
                    if prev_close < prev_ma and last_close > last_ma:
                        signals[f'ma_{period}_cross_up'] = {
                            'action': 'buy',
                            'strength': 0.6,
                            'reason': f'Price crossed above {period}-day MA',
                            'ma': last_ma,
                            'close': last_close
                        }
                    
                    # Price crosses below MA (bearish)
                    elif prev_close > prev_ma and last_close < last_ma:
                        signals[f'ma_{period}_cross_down'] = {
                            'action': 'sell',
                            'strength': 0.6,
                            'reason': f'Price crossed below {period}-day MA',
                            'ma': last_ma,
                            'close': last_close
                        }
        except Exception as e:
            self.logger.warning(f"Moving Average calculation failed: {str(e)}")
        
        return signals
    
    def _combine_signals(self, strategy_signals: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Combine signals from different strategies with weights"""
        buy_signals = []
        sell_signals = []
        
        # Collect all buy and sell signals
        for strategy, signals in strategy_signals.items():
            strategy_weight = self.strategies[strategy]["weight"]
            
            for signal_name, signal in signals.items():
                # Apply strategy weight to signal strength
                weighted_signal = signal.copy()
                weighted_signal['strength'] = signal['strength'] * strategy_weight
                weighted_signal['strategy'] = strategy
                weighted_signal['signal_name'] = signal_name
                
                if signal['action'] == 'buy':
                    buy_signals.append(weighted_signal)
                elif signal['action'] == 'sell':
                    sell_signals.append(weighted_signal)
        
        # Sort by strength (descending)
        buy_signals.sort(key=lambda x: x['strength'], reverse=True)
        sell_signals.sort(key=lambda x: x['strength'], reverse=True)
        
        # Calculate overall buy/sell sentiment
        buy_strength = sum(signal['strength'] for signal in buy_signals) if buy_signals else 0
        sell_strength = sum(signal['strength'] for signal in sell_signals) if sell_signals else 0
        
        # Determine overall action
        action = 'hold'
        if buy_strength > 0 and buy_strength > sell_strength:
            action = 'buy'
        elif sell_strength > 0 and sell_strength > buy_strength:
            action = 'sell'
        
        # Combine into final signal
        combined = {
            'action': action,
            'buy_strength': buy_strength,
            'sell_strength': sell_strength,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'net_strength': buy_strength - sell_strength
        }
        
        return combined
    
    def _apply_position_sizing(self, signals: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply position sizing to the signals"""
        if signals['action'] == 'hold':
            return signals
        
        # In a real implementation, this would use portfolio value, current positions, etc.
        # For now, we'll use a simplified approach
        
        # Example: If we're buying, calculate position size based on risk
        if signals['action'] == 'buy':
            # Get price and calculate quantity
            price = data['close'][-1] if isinstance(data['close'], list) else data['close'].iloc[-1]
            
            # Mock portfolio value (in production, get from broker)
            portfolio_value = 100000  # $100k portfolio
            
            # Calculate position size based on max position percentage
            max_position_value = portfolio_value * self.position_sizing["max_position_pct"]
            
            # Calculate position size based on risk per trade
            risk_amount = portfolio_value * self.position_sizing["risk_per_trade_pct"]
            
            # For simplicity, use a 5% stop loss
            stop_loss_pct = 0.05
            risk_based_position = risk_amount / (price * stop_loss_pct)
            
            # Take the smaller of the two calculations
            max_shares = max_position_value / price
            shares_to_buy = min(max_shares, risk_based_position)
            
            # Round down to whole shares
            shares_to_buy = int(shares_to_buy)
            
            signals['qty'] = shares_to_buy
            signals['price'] = price
            signals['stop_loss'] = price * (1 - stop_loss_pct)
            
        # If selling, we'd determine how much of existing position to sell
        # For now, just use a fixed percentage
        elif signals['action'] == 'sell':
            signals['qty'] = 10  # Mock quantity
            signals['price'] = data['close'][-1] if isinstance(data['close'], list) else data['close'].iloc[-1]
        
        return signals
    
    def update_performance(self, trade_result: Dict[str, Any]):
        """Update agent performance metrics based on trade results"""
        self.performance_metrics["total_trades"] += 1
        
        if trade_result.get('profit', 0) > 0:
            self.performance_metrics["winning_trades"] += 1
        else:
            self.performance_metrics["losing_trades"] += 1
        
        # Update win rate
        if self.performance_metrics["total_trades"] > 0:
            self.performance_metrics["win_rate"] = self.performance_metrics["winning_trades"] / self.performance_metrics["total_trades"]
        
        # In a real implementation, we might adjust strategy weights based on performance
        self._adjust_strategies_based_on_performance()
    
    def _adjust_strategies_based_on_performance(self):
        """Dynamically adjust strategy weights based on performance"""
        # This would be implemented in a production system
        # For now, it's a placeholder
        pass
        
    def _value_strategy(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Value investing strategy specifically for the value_agent"""
        signals = {}
        
        # Extract data
        try:
            # Ensure we have the required data
            if 'close' not in data or not data['close'] or len(data['close']) < 20:
                return {'action': 'hold', 'reason': 'Insufficient data for value strategy'}
                
            # Make sure we convert arrays to pandas Series for calculations
            close_values = data['close']
            symbol = data.get('symbol', 'UNKNOWN')
            
            # Convert to pandas Series if it's a numpy array or list
            if isinstance(close_values, (list, np.ndarray)):
                close = pd.Series(close_values)
            elif isinstance(close_values, pd.Series):
                close = close_values
            else:
                self.logger.warning(f"Unexpected close price data type: {type(close_values)}")
                return {'action': 'hold', 'reason': 'Invalid data format'}
            
            # Get the current price
            current_price = close.iloc[-1] if len(close) > 0 else 0
            if current_price == 0:
                return {'action': 'hold', 'reason': 'Invalid price data'}
                
            # Calculate simple moving averages
            if len(close) >= 20:
                # Calculate MAs using pandas functions
                short_window = 5
                med_window = 10
                long_window = 20
                
                short_ma = close.rolling(window=short_window).mean()
                med_ma = close.rolling(window=med_window).mean()
                long_ma = close.rolling(window=long_window).mean()
                
                # Get latest values that aren't NaN
                if not short_ma.empty and not np.isnan(short_ma.iloc[-1]):
                    latest_short = short_ma.iloc[-1]
                    latest_med = med_ma.iloc[-1]
                    latest_long = long_ma.iloc[-1]
                    
                    # Log for debugging
                    self.logger.info(f"Value strategy MAs - Short: {latest_short:.2f}, Med: {latest_med:.2f}, Long: {latest_long:.2f}")
                    
                    # Generate signals based on MA crossovers
                    if latest_short > latest_med:
                        signals["ma_bullish"] = {
                            'action': 'buy',
                            'symbol': symbol,
                            'qty': 10,
                            'price': current_price,
                            'strength': 0.8,
                            'reason': f'Bullish MA: 5-day ({latest_short:.2f}) > 10-day ({latest_med:.2f})'
                        }
                    elif latest_short < latest_long:
                        signals["ma_bearish"] = {
                            'action': 'sell',
                            'symbol': symbol,
                            'qty': 10,
                            'price': current_price,
                            'strength': 0.8,
                            'reason': f'Bearish MA: 5-day ({latest_short:.2f}) < 20-day ({latest_long:.2f})'
                        }
            
            # Price momentum strategy - safer implementation
            if len(close) >= 10:
                try:
                    # Calculate 5-day price change safely
                    if close.iloc[-5] > 0:  # Avoid division by zero
                        price_change_5d = (close.iloc[-1] / close.iloc[-5] - 1) * 100
                        
                        # Generate signals based on momentum
                        if price_change_5d > 1.0:  # 1% up in 5 days
                            signals["momentum_up"] = {
                                'action': 'buy',
                                'symbol': symbol,
                                'qty': max(5, int(10 * (price_change_5d / 2))),  # More shares for stronger momentum
                                'price': current_price,
                                'strength': min(0.9, 0.7 + price_change_5d/10),
                                'reason': f'Strong momentum: +{price_change_5d:.1f}% in 5 days'
                            }
                        elif price_change_5d < -1.0:  # 1% down in 5 days
                            signals["momentum_down"] = {
                                'action': 'sell',
                                'symbol': symbol,
                                'qty': max(5, int(10 * (abs(price_change_5d) / 2))),
                                'price': current_price,
                                'strength': min(0.9, 0.7 + abs(price_change_5d)/10),
                                'reason': f'Negative momentum: {price_change_5d:.1f}% in 5 days'
                            }
                except Exception as e:
                    self.logger.warning(f"Error calculating momentum: {str(e)}")
            
            # Simple price pattern: buy on dips, sell on peaks - safer implementation
            if len(close) >= 3:
                try:
                    # Check for short-term price patterns safely
                    if close.iloc[-2] > 0:  # Avoid division by zero
                        two_day_change = (close.iloc[-1] / close.iloc[-2] - 1) * 100
                        
                        # Generate basic mean reversion signals - buy dips, sell peaks
                        if two_day_change < -0.5:  # 0.5% dip in a day - buy opportunity
                            signals["buy_dip"] = {
                                'action': 'buy',
                                'symbol': symbol,
                                'qty': 10,
                                'price': current_price,
                                'strength': 0.75,
                                'reason': f'Buying the dip: {two_day_change:.1f}% drop'
                            }
                        elif two_day_change > 0.5:  # 0.5% rise in a day - sell opportunity
                            signals["sell_peak"] = {
                                'action': 'sell',
                                'symbol': symbol,
                                'qty': 10,
                                'price': current_price,
                                'strength': 0.75,
                                'reason': f'Selling the peak: +{two_day_change:.1f}% rise'
                            }
                except Exception as e:
                    self.logger.warning(f"Error calculating price patterns: {str(e)}")
                    
            # Ensure we have at least one signal if there's enough data
            if not signals and len(close) > 20:
                # Generate a simple buy signal based on recent price performance
                if close.iloc[-1] > close.iloc[-20]:  # Price is higher than 20 days ago
                    signals["trend_following"] = {
                        'action': 'buy',
                        'symbol': symbol,
                        'qty': 10,
                        'price': current_price,
                        'strength': 0.6,
                        'reason': 'Following upward trend'
                    }
                else:  # Price is lower than 20 days ago
                    signals["trend_following"] = {
                        'action': 'sell',
                        'symbol': symbol,
                        'qty': 10,
                        'price': current_price,
                        'strength': 0.6,
                        'reason': 'Following downward trend'
                    }
            
        except Exception as e:
            self.logger.error(f"Error in value strategy: {str(e)}")
            
        return signals
