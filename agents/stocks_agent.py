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
            
            # Generate signals from each strategy
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
        params = self.strategies["momentum"]["params"]
        
        # RSI Strategy
        try:
            # Convert to pandas Series if it's a numpy array
            if isinstance(close, np.ndarray):
                close = pd.Series(close)
                
            if len(close) >= params["rsi_period"]:
                # Calculate price changes
                delta = close.diff()
                
                # Separate gains and losses
                gain = delta.copy()
                loss = delta.copy()
                gain[gain < 0] = 0
                loss[loss > 0] = 0
                loss = -loss  # Make losses positive
                
                # Calculate average gain and loss over RSI period
                avg_gain = gain.rolling(window=params["rsi_period"]).mean()
                avg_loss = loss.rolling(window=params["rsi_period"]).mean()
                
                # Calculate relative strength
                rs = avg_gain / avg_loss
                
                # Calculate RSI
                rsi_values = 100 - (100 / (1 + rs))
                rsi_values = rsi_values.dropna()
                
                last_rsi = rsi_values.iloc[-1] if not rsi_values.empty else None
            else:
                last_rsi = None
            
            if last_rsi is not None:
                if last_rsi < params["rsi_oversold"]:
                    signals['rsi'] = {
                        'action': 'buy', 
                        'strength': (params["rsi_oversold"] - last_rsi) / params["rsi_oversold"], 
                        'reason': f'RSI oversold ({last_rsi:.2f})',
                        'value': last_rsi
                    }
                elif last_rsi > params["rsi_overbought"]:
                    signals['rsi'] = {
                        'action': 'sell', 
                        'strength': (last_rsi - params["rsi_overbought"]) / (100 - params["rsi_overbought"]), 
                        'reason': f'RSI overbought ({last_rsi:.2f})',
                        'value': last_rsi
                    }
        except Exception as e:
            self.logger.warning(f"RSI calculation failed: {str(e)}")
        
        # MACD Strategy
        try:
            # Convert to pandas Series if it's a numpy array
            if isinstance(close, np.ndarray):
                close = pd.Series(close)
                
            if len(close) >= params["macd_slow"]:
                # Calculate fast EMA
                fast_ema = close.ewm(span=params["macd_fast"], adjust=False).mean()
                
                # Calculate slow EMA
                slow_ema = close.ewm(span=params["macd_slow"], adjust=False).mean()
                
                # Calculate MACD line
                macd_line = fast_ema - slow_ema
                
                # Calculate signal line
                signal_line = macd_line.ewm(span=params["macd_signal"], adjust=False).mean()
                
                # Calculate histogram
                histogram = macd_line - signal_line
                
                # Drop NaN values
                macd_line = macd_line.dropna()
                signal_line = signal_line.dropna()
                histogram = histogram.dropna()
            
            if not macd_line.empty and not signal_line.empty:
                last_macd = macd_line.iloc[-1]
                last_signal = signal_line.iloc[-1]
                prev_macd = macd_line.iloc[-2] if len(macd_line) > 1 else None
                prev_signal = signal_line.iloc[-2] if len(signal_line) > 1 else None
                
                # MACD crossover (bullish)
                if prev_macd and prev_signal and prev_macd < prev_signal and last_macd > last_signal:
                    signals['macd_crossover'] = {
                        'action': 'buy',
                        'strength': 0.8,
                        'reason': 'MACD bullish crossover',
                        'macd': last_macd,
                        'signal': last_signal
                    }
                # MACD crossunder (bearish)
                elif prev_macd and prev_signal and prev_macd > prev_signal and last_macd < last_signal:
                    signals['macd_crossover'] = {
                        'action': 'sell',
                        'strength': 0.8,
                        'reason': 'MACD bearish crossover',
                        'macd': last_macd,
                        'signal': last_signal
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
