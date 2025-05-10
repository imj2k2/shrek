from indicators import rsi, macd, bollinger, atr, vwap, moving_averages
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime

# Utility function for safe indexing of arrays/series
def safe_idx(data: Union[pd.Series, np.ndarray, list], idx: int):
    """Safely access elements from pandas Series, NumPy arrays, or lists
    
    Args:
        data: Data structure to access (pandas Series, NumPy array, or list)
        idx: Index to access (negative indexing supported)
        
    Returns:
        The value at the specified index or None if not available
    """
    try:
        if isinstance(data, pd.Series):
            if data.empty:
                return None
            return data.iloc[idx]
        elif isinstance(data, (np.ndarray, list)):
            if len(data) == 0:
                return None
            return data[idx]
        else:
            return None
    except (IndexError, KeyError):
        return None

class StocksAgent:
    """Agent for stock trading strategies including momentum, mean reversion, and breakout"""
    
    def __init__(self, mcp=None):
        self.logger = logging.getLogger("StocksAgent")
        self.mcp = mcp  # Model Context Protocol for communication
        self.debug_enabled = os.environ.get('DEBUG_TRADING', 'false').lower() == 'true'  # Debug mode for forced trades
        
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
        # Add debugging to track the signal generation process
        self.logger.info(f"Generating signals for {data.get('symbol', 'UNKNOWN')}")        
        # Use the debug_enabled attribute instead of checking environment directly
        # This allows setting debug mode programmatically from schemas.py
        
        # In debug mode, force some trades to test the backtesting engine
        if self.debug_enabled:
            self.logger.info("Debug mode enabled, forcing test signals")
            # Get the date from the data if available for better logging
            date_str = data.get('date', datetime.now().strftime('%Y-%m-%d'))
            symbol = data.get('symbol', 'UNKNOWN')
            
            # Construct a forced buy signal that works with backtester
            # The backtester expects a specific format with action and sufficient metadata
            buy_signal = {
                'action': 'buy',
                'symbol': symbol,
                'qty': 10,  # Buy 10 shares
                'strategy': 'debug_forced',
                'timestamp': datetime.now().isoformat(),
                'reason': f'Debug mode forced buy signal for {symbol} on {date_str}',
                'net_strength': 0.8,  # Strong signal
                'buy_signals': [
                    {
                        'action': 'buy',
                        'strength': 0.8,
                        'reason': 'Debug mode forced signal'
                    }
                ],
                'sell_signals': []
            }
            
            self.logger.info(f"Generated forced buy signal: {buy_signal}")
            return buy_signal
            
        # Check if a specific strategy was requested
        requested_strategy = data.get('strategy_name', '').lower()
        if requested_strategy and requested_strategy != 'debug_forced':
            self.logger.info(f"Using requested strategy: {requested_strategy}")
            # Enable only the requested strategy if it exists in our strategies dictionary
            found_strategy = False
            for strategy_name in self.strategies:
                if strategy_name.lower() in requested_strategy:
                    self.strategies[strategy_name]['enabled'] = True
                    self.strategies[strategy_name]['weight'] = 1.0  # Give it full weight
                    self.logger.info(f"Enabled {strategy_name} strategy with weight 1.0")
                    found_strategy = True
                else:
                    self.strategies[strategy_name]['enabled'] = False
                    self.logger.info(f"Disabled {strategy_name} strategy")
            
            # If no matching strategy was found, enable all strategies
            if not found_strategy:
                self.logger.warning(f"No matching strategy found for '{requested_strategy}', enabling all strategies")
                for strategy_name in self.strategies:
                    self.strategies[strategy_name]['enabled'] = True
                    self.logger.info(f"Enabled {strategy_name} strategy with default weight")
            
            # Log the active strategies
            active_strategies = [s for s in self.strategies if self.strategies[s]['enabled']]
            self.logger.info(f"Active strategies: {active_strategies}")
            
            # If no strategies are enabled, force at least one to be active
            if not any(self.strategies[s]['enabled'] for s in self.strategies):
                self.logger.warning("No strategies enabled, enabling momentum strategy as fallback")
                self.strategies['momentum']['enabled'] = True
                self.strategies['momentum']['weight'] = 1.0
            
        if not data or 'close' not in data:
            self.logger.error("Invalid data format: missing 'close' prices")
            return {'action': 'hold', 'reason': 'Missing price data'}
        
        try:
            # More robust handling of different data types for price data
            try:
                # Always ensure we have pandas Series with proper index for all data
                if 'close' in data:
                    # First ensure close data is a pandas Series
                    if isinstance(data['close'], list):
                        close = pd.Series(data['close'])
                    elif isinstance(data['close'], np.ndarray):
                        close = pd.Series(data['close'])
                    elif isinstance(data['close'], pd.Series):
                        close = data['close'].copy()  # Make a copy to avoid modifying original
                    else:
                        close = pd.Series([data['close']] if np.isscalar(data['close']) else data['close'])
                else:
                    self.logger.error("Missing close price data")
                    return {'action': 'hold', 'reason': 'Missing close price data'}
            
                # Create a numeric index if needed
                if not hasattr(close, 'index') or len(close.index) == 0:
                    close.index = range(len(close))
                
                # Process other data with the same index as close for alignment
                date_index = close.index
                
                # High prices
                if 'high' in data:
                    if isinstance(data['high'], list):
                        high = pd.Series(data['high'], index=date_index[:len(data['high'])])
                    elif isinstance(data['high'], np.ndarray):
                        high = pd.Series(data['high'], index=date_index[:len(data['high'])])
                    elif isinstance(data['high'], pd.Series):
                        high = data['high'].copy()
                    else:
                        high = pd.Series([data['high']] if np.isscalar(data['high']) else data['high'], index=date_index[:1 if np.isscalar(data['high']) else len(data['high'])])
                else:
                    # If no high data, use close data as a fallback
                    high = close.copy()
                    self.logger.warning("No high price data, using close prices instead")
                
                # Open prices
                if 'open' in data:
                    if isinstance(data['open'], list):
                        open_data = pd.Series(data['open'], index=date_index[:len(data['open'])])
                    elif isinstance(data['open'], np.ndarray):
                        open_data = pd.Series(data['open'], index=date_index[:len(data['open'])])
                    elif isinstance(data['open'], pd.Series):
                        open_data = data['open'].copy()
                    else:
                        open_data = pd.Series([data['open']] if np.isscalar(data['open']) else data['open'], index=date_index[:1 if np.isscalar(data['open']) else len(data['open'])])
                else:
                    # If no open data, use close data as a fallback
                    open_data = close.copy()
                    self.logger.warning("No open price data, using close prices instead")
                
                # Low prices
                if 'low' in data:
                    if isinstance(data['low'], list):
                        low = pd.Series(data['low'], index=date_index[:len(data['low'])])
                    elif isinstance(data['low'], np.ndarray):
                        low = pd.Series(data['low'], index=date_index[:len(data['low'])])
                    elif isinstance(data['low'], pd.Series):
                        low = data['low'].copy()
                    else:
                        low = pd.Series([data['low']] if np.isscalar(data['low']) else data['low'], index=date_index[:1 if np.isscalar(data['low']) else len(data['low'])])
                else:
                    # If no low data, use close data as a fallback
                    low = close.copy()
                    self.logger.warning("No low price data, using close prices instead")
                
                # Volume
                if 'volume' in data:
                    if isinstance(data['volume'], list):
                        volume = pd.Series(data['volume'], index=date_index[:len(data['volume'])])
                    elif isinstance(data['volume'], np.ndarray):
                        volume = pd.Series(data['volume'], index=date_index[:len(data['volume'])])
                    elif isinstance(data['volume'], pd.Series):
                        volume = data['volume'].copy()
                    else:
                        volume = pd.Series([data['volume']] if np.isscalar(data['volume']) else data['volume'], index=date_index[:1 if np.isscalar(data['volume']) else len(data['volume'])])
                else:
                    # If no volume data, create dummy volume (1s)
                    volume = pd.Series(np.ones(len(close)), index=close.index)
                    self.logger.warning("No volume data, using placeholder values")
            except Exception as e:
                self.logger.error(f"Error preprocessing price data: {str(e)}")
                return {'action': 'hold', 'reason': f'Data preprocessing error: {str(e)}'}
                
            symbol = data.get('symbol', 'UNKNOWN')
            
            # Get the current price for signal generation
            # Use safe_idx to handle any data type
            if len(close) > 0:
                current_price = float(safe_idx(close, -1))
            elif hasattr(close, '__len__') and len(close) > 0:
                current_price = float(close[-1])
            else:
                current_price = 0.0
            
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
            
            # Log signals but don't send to MCP (removed due to missing dependency)
            self.logger.info(f"Generated signals for {symbol}: {final_signals['action']}")
            # We've removed the MCP messaging to fix import errors
            
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
            
            # Get the latest RSI value - handle both pandas Series and NumPy arrays
            if isinstance(rsi, pd.Series):
                # Use the safe_idx utility for consistent handling
                latest_rsi = safe_idx(rsi, -1)
            elif isinstance(rsi, np.ndarray) and len(rsi) > 0:
                latest_rsi = rsi[-1]
            else:
                latest_rsi = None
            
            if latest_rsi is not None:
                # Oversold condition (potential buy) - Using standard RSI thresholds
                if latest_rsi < params["rsi_oversold"]:
                    # Scale strength based on how oversold (more oversold = stronger signal)
                    strength = 0.5 + ((params["rsi_oversold"] - latest_rsi) / 30)
                    signals["rsi_oversold"] = {
                        "action": "buy",
                        "strength": min(0.9, strength),  # Cap at 0.9
                        "reason": f"RSI oversold ({latest_rsi:.2f})",
                        "strategy": "momentum"
                    }
                # Overbought condition (potential sell) - Using standard RSI thresholds
                elif latest_rsi > params["rsi_overbought"]:
                    # Scale strength based on how overbought (more overbought = stronger signal)
                    strength = 0.5 + ((latest_rsi - params["rsi_overbought"]) / 30)
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
                
                # Convert to pandas Series if needed
                if isinstance(macd_line, np.ndarray):
                    macd_line = pd.Series(macd_line)
                if isinstance(signal_line, np.ndarray):
                    signal_line = pd.Series(signal_line)
                
                # Get latest values
                latest_macd = safe_idx(macd_line, -1) if len(macd_line) > 0 else None
                latest_signal = safe_idx(signal_line, -1) if len(signal_line) > 0 else None
                
                if latest_macd is not None and latest_signal is not None and len(macd_line) > 1 and len(signal_line) > 1:
                    # Get previous values
                    prev_macd = safe_idx(macd_line, -2)
                    prev_signal = safe_idx(signal_line, -2)
                    
                    # Calculate distance between MACD and signal line
                    macd_distance = abs(latest_macd - latest_signal)
                    norm_distance = macd_distance / abs(latest_signal) if latest_signal != 0 else 0
                    
                    # MACD crosses above signal line (potential buy)
                    if prev_macd < prev_signal and latest_macd > latest_signal:
                        # Strength based on the size of the crossover movement
                        signals["macd_cross_above"] = {
                            "action": "buy",
                            "strength": min(0.8, 0.6 + norm_distance),
                            "reason": "MACD crossed above signal line",
                            "strategy": "momentum"
                        }
                    # MACD crosses below signal line (potential sell)
                    elif prev_macd > prev_signal and latest_macd < latest_signal:
                        # Strength based on the size of the crossover movement
                        signals["macd_cross_below"] = {
                            "action": "sell",
                            "strength": min(0.8, 0.6 + norm_distance),
                            "reason": "MACD crossed below signal line",
                            "strategy": "momentum"
                        }
                    
                    # MACD is positive and increasing (bullish momentum)
                    if latest_macd > 0 and len(macd_line) > 5 and latest_macd > safe_idx(macd_line, -5):
                        signals["macd_bullish"] = {
                            "action": "buy",
                            "strength": 0.5,
                            "reason": "MACD is positive and increasing",
                            "strategy": "momentum"
                        }
                    
                    # MACD is negative and decreasing (bearish momentum)
                    elif latest_macd < 0 and len(macd_line) > 5 and latest_macd < safe_idx(macd_line, -5):
                        signals["macd_bearish"] = {
                            "action": "sell",
                            "strength": 0.5,
                            "reason": "MACD is negative and decreasing",
                            "strategy": "momentum"
                        }
        except Exception as e:
            self.logger.warning(f"MACD calculation failed: {str(e)}")
            
        # Bollinger Bands Strategy (separate from MACD)
        try:
            if len(close) >= params.get("bollinger_period", 20):
                # Calculate SMA for middle band
                middle = close.rolling(window=params.get("bollinger_period", 20)).mean()
                
                # Calculate standard deviation
                stddev = close.rolling(window=params.get("bollinger_period", 20)).std()
                
                # Add Bollinger Bands signals if needed
                # (Implemented separately from MACD)
        except Exception as e:
            self.logger.warning(f"Bollinger Bands calculation failed: {str(e)}")
        
        
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
            
            # Extract the latest values and handle both pandas Series and NumPy arrays
            if isinstance(upper, pd.Series) and isinstance(lower, pd.Series):
                if not upper.empty and not lower.empty:
                    # Handle pandas Series
                    # Use safe_idx for consistent data access
                    last_upper = safe_idx(upper, -1)
                    last_lower = safe_idx(lower, -1)
                    last_middle = safe_idx(middle, -1)
                    last_close = safe_idx(close, -1)
                else:
                    return signals  # Not enough data
            elif isinstance(upper, np.ndarray) and isinstance(lower, np.ndarray) and len(upper) > 0 and len(lower) > 0:
                # Handle NumPy arrays
                last_upper = upper[-1]
                last_lower = lower[-1]
                last_middle = middle[-1] if isinstance(middle, np.ndarray) and len(middle) > 0 else None
                last_close = close[-1] if isinstance(close, np.ndarray) and len(close) > 0 else None
            else:
                return signals  # Not the right data format
                
                # Price below lower band (potential buy)
                if last_close < last_lower:
                    # Calculate how far below the band (normalized)
                    band_width = last_upper - last_lower
                    distance = (last_lower - last_close) / band_width if band_width > 0 else 0
                    # Use a more conservative strength calculation - don't artificially boost
                    strength = min(0.7, 0.4 + distance)  # Lower base and cap
                    
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
                    # Use a more conservative strength calculation - don't artificially boost
                    strength = min(0.7, 0.4 + distance)  # Lower base and cap
                    
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
                # Convert to pandas Series if they're numpy arrays
                if isinstance(high, np.ndarray):
                    high = pd.Series(high, index=close.index)
                if isinstance(low, np.ndarray):
                    low = pd.Series(low, index=close.index)
                    
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
                
                    # Handle both pandas Series and NumPy arrays
                    if isinstance(vwap_values, pd.Series) and not vwap_values.empty:
                        last_vwap = safe_idx(vwap_values, -1)
                        last_close = safe_idx(close, -1)
                        
                        # Calculate percentage difference from VWAP
                        if last_close is not None and last_vwap is not None:
                            pct_diff = (last_close - last_vwap) / last_vwap * 100
                    
                    # Price significantly below VWAP (potential buy) - using standard 3% threshold
                    if pct_diff < -3.0:  # 3% below VWAP
                        # Calculate strength based on distance from VWAP (more distance = stronger signal)
                        # but cap it to avoid overemphasizing extreme moves
                        strength = min(0.7, 0.4 + abs(pct_diff) / 10)
                        signals['vwap_below'] = {
                            'action': 'buy',
                            'strength': strength,
                            'reason': f'Price {abs(pct_diff):.2f}% below VWAP',
                            'close': last_close,
                            'vwap': last_vwap,
                            'pct_diff': pct_diff
                        }
                    
                    # Price significantly above VWAP (potential sell) - using standard 3% threshold
                    elif pct_diff > 3.0:  # 3% above VWAP
                        # Calculate strength based on distance from VWAP (more distance = stronger signal)
                        # but cap it to avoid overemphasizing extreme moves
                        strength = min(0.7, 0.4 + abs(pct_diff) / 10)
                        signals['vwap_above'] = {
                            'action': 'sell',
                            'strength': strength,
                            'reason': f'Price {pct_diff:.2f}% above VWAP',
                            'close': last_close,
                            'vwap': last_vwap,
                            'pct_diff': pct_diff
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
                
                    # Handle both pandas Series and NumPy arrays
                    if isinstance(atr_values, pd.Series) and not atr_values.empty and len(close) > 1:
                        last_atr = safe_idx(atr_values, -1)
                        # Use safe_idx for consistent data access regardless of type
                        last_close = safe_idx(close, -1)
                        prev_close = safe_idx(close, -2)
                        
                        if last_close is None or prev_close is None:
                            return signals  # Not enough data
                    elif isinstance(atr_values, np.ndarray) and len(atr_values) > 0 and len(close) > 1:
                        # Convert numpy arrays to pandas Series for consistent handling
                        atr_values_series = pd.Series(atr_values)
                        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close
                        
                        last_atr = safe_idx(atr_values_series, -1)
                        last_close = safe_idx(close_series, -1)
                        prev_close = safe_idx(close_series, -2)
                        
                        if last_close is None or prev_close is None:
                            return signals  # Not enough data
                    else:
                        return signals  # Not enough data
                    
                    # Calculate the price movement as a multiple of ATR
                    price_move = last_close - prev_close
                    move_in_atr = abs(price_move) / last_atr if last_atr > 0 else 0
                    
                    # Significant price move up (> ATR * multiplier) - true volatility breakout
                    # Note: ATR multiplier is defined in the strategy parameters
                    if price_move > 0 and move_in_atr > params["atr_multiplier"]:
                        # Calculate strength based on how much it exceeded the multiplier
                        # But use conservative approach to avoid artificially strong signals
                        strength = min(0.7, 0.5 + (move_in_atr - params["atr_multiplier"]) / 10)
                        signals['atr_breakout_up'] = {
                            'action': 'buy',
                            'strength': strength,
                            'reason': f'Volatility breakout up ({move_in_atr:.2f}x ATR)',
                            'atr': last_atr,
                            'move_size': price_move,
                            'move_in_atr': move_in_atr
                        }
                    
                    # Significant price move down (> ATR * multiplier) - true volatility breakout
                    elif price_move < 0 and move_in_atr > params["atr_multiplier"]:
                        # Calculate strength based on how much it exceeded the multiplier
                        # But use conservative approach to avoid artificially strong signals
                        strength = min(0.7, 0.5 + (move_in_atr - params["atr_multiplier"]) / 10)
                        signals['atr_breakout_down'] = {
                            'action': 'sell',
                            'strength': strength,
                            'reason': f'Volatility breakout down ({move_in_atr:.2f}x ATR)',
                            'atr': last_atr,
                            'move_size': abs(price_move),
                            'move_in_atr': move_in_atr
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
                
                    # Safely handle both pandas Series and NumPy arrays
                    if isinstance(ma_values, pd.Series) and not ma_values.empty and len(close) > 1 and len(ma_values) > 1:
                        # Pandas Series handling
                        last_ma = safe_idx(ma_values, -1)
                        prev_ma = safe_idx(ma_values, -2)
                        
                        last_close = safe_idx(close, -1)
                        prev_close = safe_idx(close, -2)
                    elif isinstance(ma_values, np.ndarray) and len(ma_values) > 1 and len(close) > 1:
                        # Convert to pandas Series for consistent handling
                        ma_values_series = pd.Series(ma_values)
                        close_series = pd.Series(close) if isinstance(close, np.ndarray) else close
                        
                        last_ma = safe_idx(ma_values_series, -1)
                        prev_ma = safe_idx(ma_values_series, -2)
                        
                        last_close = safe_idx(close_series, -1)
                        prev_close = safe_idx(close_series, -2)
                    if last_close is None or prev_close is None or last_ma is None or prev_ma is None:
                        continue
                    
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
                else:
                    # Skip this MA period if not enough data
                    continue
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
            # Get price and calculate quantity in a type-safe manner
            if isinstance(data['close'], list) or isinstance(data['close'], np.ndarray):
                # Handle list or numpy array
                price = data['close'][-1]
            elif isinstance(data['close'], pd.Series):
                # Handle pandas Series
                price = data['close'].iloc[-1] if not data['close'].empty else 0
            else:
                # Handle scalar or other types
                price = data['close']
            
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
            
            # Add the strategy name if it was specified
            if data.get('strategy_name'):
                signals['strategy'] = data.get('strategy_name')
            
            return signals
        
        # If selling, we'd determine how much of existing position to sell
        # For now, just use a fixed percentage
        elif signals['action'] == 'sell':
            signals['qty'] = 10  # Mock quantity
            # Get price in a type-safe manner
            if isinstance(data['close'], list) or isinstance(data['close'], np.ndarray):
                # Handle list or numpy array
                signals['price'] = data['close'][-1]
            elif isinstance(data['close'], pd.Series):
                # Handle pandas Series
                signals['price'] = data['close'].iloc[-1] if not data['close'].empty else 0
            else:
                # Handle scalar or other types
                signals['price'] = data['close']
        
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
            if 'close' not in data:
                return {'action': 'hold', 'reason': 'No close price data available'}
                
            close_values = data['close']
            
            # More thorough check using scalar values
            has_close_data = False
            if isinstance(close_values, (list, np.ndarray)) and len(close_values) > 0:
                has_close_data = True
            elif isinstance(close_values, pd.Series) and not close_values.empty:
                has_close_data = True
                
            if not has_close_data or (isinstance(close_values, (list, np.ndarray, pd.Series)) and len(close_values) < 20):
                return {'action': 'hold', 'reason': 'Insufficient data for value strategy'}
            
            symbol = data.get('symbol', 'UNKNOWN')
            
            # Convert to pandas Series if it's a numpy array or list
            if isinstance(close_values, (list, np.ndarray)):
                close = pd.Series(close_values)
            elif isinstance(close_values, pd.Series):
                close = close_values
            else:
                self.logger.warning(f"Unexpected close price data type: {type(close_values)}")
                return {'action': 'hold', 'reason': 'Invalid data format'}
            
            # Get the current price as a scalar value to avoid numpy array comparison issues
            try:
                current_price = float(safe_idx(close, -1) or 0.0)
                if current_price <= 0:
                    return {'action': 'hold', 'reason': 'Invalid price data (zero or negative)'}
            except (ValueError, TypeError, IndexError) as e:
                self.logger.warning(f"Error getting current price: {e}")
                return {'action': 'hold', 'reason': f'Error getting price: {e}'}
                
            # Calculate simple moving averages
            if len(close) >= 20:
                # Calculate MAs using pandas functions
                try:
                    short_window = 5
                    med_window = 10
                    long_window = 20
                    
                    short_ma = close.rolling(window=short_window).mean()
                    med_ma = close.rolling(window=med_window).mean()
                    long_ma = close.rolling(window=long_window).mean()
                    
                    # Get latest values that aren't NaN as scalars to avoid numpy array issues
                    has_valid_mas = (not short_ma.empty and 
                                    not med_ma.empty and 
                                    not long_ma.empty and 
                                    not pd.isna(safe_idx(short_ma, -1)) and 
                                    not pd.isna(safe_idx(med_ma, -1)) and 
                                    not pd.isna(safe_idx(long_ma, -1)))
                    
                    if has_valid_mas:
                        latest_short = float(safe_idx(short_ma, -1) or 0)
                        latest_med = float(safe_idx(med_ma, -1) or 0)
                        latest_long = float(safe_idx(long_ma, -1) or 0)
                except Exception as e:
                    self.logger.error(f"Error calculating moving averages: {e}")
                    return {'action': 'hold', 'reason': f'Error in MA calculation: {e}'}
                    
                    # Log for debugging
                    self.logger.info(f"Value strategy MAs - Short: {latest_short:.2f}, Med: {latest_med:.2f}, Long: {latest_long:.2f}")
                    
                    # Generate signals based on MA crossovers - safe handling for NumPy comparisons
                    # Use scalar comparison to avoid ambiguous truth value error
                    if isinstance(latest_short, (int, float)) and isinstance(latest_med, (int, float)) and latest_short > latest_med:
                        signals["ma_bullish"] = {
                            'action': 'buy',
                            'symbol': symbol,
                            'qty': 10,
                            'price': current_price,
                            'strength': 0.8,
                            'reason': f'Bullish MA: 5-day ({latest_short:.2f}) > 10-day ({latest_med:.2f})'
                        }
                    elif isinstance(latest_short, (int, float)) and isinstance(latest_long, (int, float)) and latest_short < latest_long:
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
                    # Get scalar price values to avoid ambiguous truth value errors
                    try:
                        price_5d_ago = float(safe_idx(close, -5) or 0) 
                        current_price_value = float(safe_idx(close, -1) or 0)
                    except (ValueError, TypeError, IndexError) as e:
                        self.logger.warning(f"Error extracting price values: {e}")
                        # Skip momentum calculation if price extraction fails
                        price_5d_ago = 0.0
                    
                    # Only proceed if we have valid price data
                    if price_5d_ago > 0.0 and current_price_value > 0.0:  # Avoid division by zero
                        # Calculate price change as percentage
                        price_change_5d = (current_price_value / price_5d_ago - 1.0) * 100.0
                        
                        # Generate signals based on momentum - using explicit float comparison
                        if price_change_5d > 1.0:  # 1% up in 5 days
                            # Calculate position size (safely)
                            position_size = max(5, int(10 * (price_change_5d / 2) if price_change_5d > 0 else 1))
                            signal_strength = min(0.9, 0.5 + price_change_5d / 10.0)
                            
                            signals["momentum_up"] = {
                                'action': 'buy',
                                'symbol': symbol,
                                'qty': position_size,
                                'price': current_price,
                                'strength': signal_strength,
                                'reason': f'Strong upward momentum: {price_change_5d:.1f}% in 5 days'
                            }
                        elif price_change_5d < -1.0:  # 1% down in 5 days
                            # Calculate position size (safely)
                            position_size = max(5, int(10 * (abs(price_change_5d) / 2) if price_change_5d < 0 else 1))
                            signal_strength = min(0.9, 0.5 + abs(price_change_5d) / 10.0)
                            
                            signals["momentum_down"] = {
                                'action': 'sell',
                                'symbol': symbol,
                                'qty': position_size,
                                'price': current_price,
                                'strength': signal_strength,
                                'reason': f'Strong downward momentum: {price_change_5d:.1f}% in 5 days'
                            }
                except Exception as e:
                    self.logger.warning(f"Error calculating momentum: {str(e)}")
            
            # Simple price pattern: buy on dips, sell on peaks - safer implementation
            if len(close) >= 3:
                try:
                    # Check for short-term price patterns safely
                    prev_price = safe_idx(close, -2)
                    current_price = safe_idx(close, -1)
                    if prev_price and prev_price > 0:  # Avoid division by zero
                        two_day_change = (current_price / prev_price - 1) * 100
                        
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
                current_price = safe_idx(close, -1)
                price_20d_ago = safe_idx(close, -20)
                if current_price and price_20d_ago and current_price > price_20d_ago:  # Price is higher than 20 days ago
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
