"""
Additional trading strategies for the multi-agent trading platform.
These strategies can be combined and customized through the UI.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging

class StrategyLibrary:
    """Library of trading strategies that can be used by agents"""
    
    def __init__(self):
        self.logger = logging.getLogger("StrategyLibrary")
    
    def mean_reversion_strategy(self, close: pd.Series, high: Optional[pd.Series] = None, 
                               low: Optional[pd.Series] = None, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Mean reversion strategy using Bollinger Bands
        
        Parameters:
        - close: Series of closing prices
        - high: Series of high prices (optional)
        - low: Series of low prices (optional)
        - params: Dictionary of parameters including:
            - bollinger_period: Period for Bollinger Bands (default: 20)
            - bollinger_std: Standard deviation multiplier (default: 2.0)
            - oversold_threshold: Threshold for oversold condition (default: 0.05)
            - overbought_threshold: Threshold for overbought condition (default: 0.05)
        
        Returns:
        - Dictionary with signal information
        """
        # Default parameters
        default_params = {
            "bollinger_period": 20,
            "bollinger_std": 2.0,
            "oversold_threshold": 0.05,
            "overbought_threshold": 0.05
        }
        
        # Use provided params or defaults
        if params is None:
            params = default_params
        else:
            for key, value in default_params.items():
                if key not in params:
                    params[key] = value
        
        signals = {}
        
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
        
        return signals
    
    def trend_following_strategy(self, close: pd.Series, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Trend following strategy using moving averages
        
        Parameters:
        - close: Series of closing prices
        - params: Dictionary of parameters including:
            - fast_period: Period for fast moving average (default: 20)
            - slow_period: Period for slow moving average (default: 50)
            - signal_threshold: Threshold for signal strength (default: 0.05)
        
        Returns:
        - Dictionary with signal information
        """
        # Default parameters
        default_params = {
            "fast_period": 20,
            "slow_period": 50,
            "signal_threshold": 0.05
        }
        
        # Use provided params or defaults
        if params is None:
            params = default_params
        else:
            for key, value in default_params.items():
                if key not in params:
                    params[key] = value
        
        signals = {}
        
        # Convert to pandas Series if it's a numpy array
        if isinstance(close, np.ndarray):
            close = pd.Series(close)
            
        if len(close) >= params["slow_period"]:
            # Calculate fast and slow moving averages
            fast_ma = close.rolling(window=params["fast_period"]).mean()
            slow_ma = close.rolling(window=params["slow_period"]).mean()
            
            # Drop NaN values
            fast_ma = fast_ma.dropna()
            slow_ma = slow_ma.dropna()
            
            if not fast_ma.empty and not slow_ma.empty and len(fast_ma) > 1 and len(slow_ma) > 1:
                last_fast = fast_ma.iloc[-1]
                last_slow = slow_ma.iloc[-1]
                prev_fast = fast_ma.iloc[-2]
                prev_slow = slow_ma.iloc[-2]
                
                # Fast MA crosses above slow MA (bullish)
                if prev_fast < prev_slow and last_fast > last_slow:
                    # Calculate crossover strength
                    strength = min(0.8, 0.5 + abs(last_fast - last_slow) / last_slow)
                    
                    signals['ma_crossover_up'] = {
                        'action': 'buy',
                        'strength': strength,
                        'reason': f'Fast MA crossed above slow MA',
                        'fast_ma': last_fast,
                        'slow_ma': last_slow
                    }
                
                # Fast MA crosses below slow MA (bearish)
                elif prev_fast > prev_slow and last_fast < last_slow:
                    # Calculate crossover strength
                    strength = min(0.8, 0.5 + abs(last_fast - last_slow) / last_slow)
                    
                    signals['ma_crossover_down'] = {
                        'action': 'sell',
                        'strength': strength,
                        'reason': f'Fast MA crossed below slow MA',
                        'fast_ma': last_fast,
                        'slow_ma': last_slow
                    }
                
                # Trend strength - distance between MAs
                ma_distance = abs(last_fast - last_slow) / last_slow
                if ma_distance > params["signal_threshold"]:
                    if last_fast > last_slow:
                        signals['trend_strength_up'] = {
                            'action': 'buy',
                            'strength': min(0.7, 0.3 + ma_distance),
                            'reason': f'Strong uptrend (MA distance)',
                            'ma_distance': ma_distance
                        }
                    else:
                        signals['trend_strength_down'] = {
                            'action': 'sell',
                            'strength': min(0.7, 0.3 + ma_distance),
                            'reason': f'Strong downtrend (MA distance)',
                            'ma_distance': ma_distance
                        }
        
        return signals
    
    def breakout_strategy(self, close: pd.Series, high: Optional[pd.Series] = None, 
                         low: Optional[pd.Series] = None, volume: Optional[pd.Series] = None,
                         params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Breakout strategy using price levels and volume
        
        Parameters:
        - close: Series of closing prices
        - high: Series of high prices (optional)
        - low: Series of low prices (optional)
        - volume: Series of volume data (optional)
        - params: Dictionary of parameters including:
            - lookback_period: Period for identifying levels (default: 20)
            - breakout_threshold: Threshold for breakout (default: 0.03)
            - volume_factor: Volume increase factor for confirmation (default: 1.5)
        
        Returns:
        - Dictionary with signal information
        """
        # Default parameters
        default_params = {
            "lookback_period": 20,
            "breakout_threshold": 0.03,
            "volume_factor": 1.5
        }
        
        # Use provided params or defaults
        if params is None:
            params = default_params
        else:
            for key, value in default_params.items():
                if key not in params:
                    params[key] = value
        
        signals = {}
        
        # Convert to pandas Series if it's a numpy array
        if isinstance(close, np.ndarray):
            close = pd.Series(close)
        if high is not None and isinstance(high, np.ndarray):
            high = pd.Series(high, index=close.index)
        if low is not None and isinstance(low, np.ndarray):
            low = pd.Series(low, index=close.index)
        if volume is not None and isinstance(volume, np.ndarray):
            volume = pd.Series(volume, index=close.index)
            
        if len(close) >= params["lookback_period"]:
            # Get historical window
            hist_window = close[-params["lookback_period"]-1:-1]
            
            if not hist_window.empty:
                # Calculate resistance and support levels
                resistance = hist_window.max()
                support = hist_window.min()
                
                last_close = close.iloc[-1]
                
                # Check for resistance breakout
                if last_close > resistance * (1 + params["breakout_threshold"]):
                    # Calculate breakout strength
                    breakout_pct = (last_close - resistance) / resistance
                    strength = min(0.9, 0.5 + breakout_pct * 5)  # Scale strength
                    
                    # Check volume if available
                    volume_confirmed = False
                    if volume is not None and len(volume) > 1:
                        avg_volume = volume[-params["lookback_period"]-1:-1].mean()
                        last_volume = volume.iloc[-1]
                        volume_confirmed = last_volume > avg_volume * params["volume_factor"]
                    
                    signals['resistance_breakout'] = {
                        'action': 'buy',
                        'strength': strength * (1.2 if volume_confirmed else 1.0),  # Boost if volume confirmed
                        'reason': f'Resistance breakout{" with volume confirmation" if volume_confirmed else ""}',
                        'breakout_pct': breakout_pct,
                        'resistance': resistance,
                        'volume_confirmed': volume_confirmed
                    }
                
                # Check for support breakdown
                elif last_close < support * (1 - params["breakout_threshold"]):
                    # Calculate breakdown strength
                    breakdown_pct = (support - last_close) / support
                    strength = min(0.9, 0.5 + breakdown_pct * 5)  # Scale strength
                    
                    # Check volume if available
                    volume_confirmed = False
                    if volume is not None and len(volume) > 1:
                        avg_volume = volume[-params["lookback_period"]-1:-1].mean()
                        last_volume = volume.iloc[-1]
                        volume_confirmed = last_volume > avg_volume * params["volume_factor"]
                    
                    signals['support_breakdown'] = {
                        'action': 'sell',
                        'strength': strength * (1.2 if volume_confirmed else 1.0),  # Boost if volume confirmed
                        'reason': f'Support breakdown{" with volume confirmation" if volume_confirmed else ""}',
                        'breakdown_pct': breakdown_pct,
                        'support': support,
                        'volume_confirmed': volume_confirmed
                    }
        
        return signals
    
    def volatility_strategy(self, close: pd.Series, high: Optional[pd.Series] = None, 
                           low: Optional[pd.Series] = None, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Volatility-based strategy using ATR
        
        Parameters:
        - close: Series of closing prices
        - high: Series of high prices (optional)
        - low: Series of low prices (optional)
        - params: Dictionary of parameters including:
            - atr_period: Period for ATR calculation (default: 14)
            - volatility_threshold: Threshold for high volatility (default: 0.03)
            - low_volatility_threshold: Threshold for low volatility (default: 0.01)
        
        Returns:
        - Dictionary with signal information
        """
        # Default parameters
        default_params = {
            "atr_period": 14,
            "volatility_threshold": 0.03,
            "low_volatility_threshold": 0.01
        }
        
        # Use provided params or defaults
        if params is None:
            params = default_params
        else:
            for key, value in default_params.items():
                if key not in params:
                    params[key] = value
        
        signals = {}
        
        # Need high and low data for ATR
        if high is None or low is None:
            return signals
        
        # Convert to pandas Series if it's a numpy array
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
            atr = tr.rolling(window=params["atr_period"]).mean()
            atr = atr.dropna()
            
            if not atr.empty:
                last_atr = atr.iloc[-1]
                last_close = close.iloc[-1]
                
                # Calculate ATR as percentage of price
                atr_pct = last_atr / last_close
                
                # High volatility signal
                if atr_pct > params["volatility_threshold"]:
                    signals['high_volatility'] = {
                        'action': 'neutral',  # Could be buy or sell depending on other factors
                        'strength': min(0.7, 0.4 + atr_pct * 5),
                        'reason': f'High volatility detected (ATR: {atr_pct:.2%})',
                        'atr': last_atr,
                        'atr_pct': atr_pct
                    }
                
                # Low volatility signal (potential for volatility expansion)
                elif atr_pct < params["low_volatility_threshold"]:
                    signals['low_volatility'] = {
                        'action': 'neutral',  # Could be buy or sell depending on other factors
                        'strength': min(0.6, 0.3 + (params["low_volatility_threshold"] - atr_pct) * 10),
                        'reason': f'Low volatility detected (ATR: {atr_pct:.2%})',
                        'atr': last_atr,
                        'atr_pct': atr_pct
                    }
        
        return signals
    
    def momentum_strategy(self, close: pd.Series, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Momentum strategy using RSI and rate of change
        
        Parameters:
        - close: Series of closing prices
        - params: Dictionary of parameters including:
            - rsi_period: Period for RSI calculation (default: 14)
            - roc_period: Period for rate of change calculation (default: 10)
            - rsi_oversold: Oversold threshold for RSI (default: 30)
            - rsi_overbought: Overbought threshold for RSI (default: 70)
            - roc_threshold: Threshold for significant rate of change (default: 0.05)
        
        Returns:
        - Dictionary with signal information
        """
        # Default parameters
        default_params = {
            "rsi_period": 14,
            "roc_period": 10,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "roc_threshold": 0.05
        }
        
        # Use provided params or defaults
        if params is None:
            params = default_params
        else:
            for key, value in default_params.items():
                if key not in params:
                    params[key] = value
        
        signals = {}
        
        # Convert to pandas Series if it's a numpy array
        if isinstance(close, np.ndarray):
            close = pd.Series(close)
            
        if len(close) >= params["rsi_period"]:
            # Calculate RSI
            delta = close.diff()
            gain = delta.copy()
            loss = delta.copy()
            gain[gain < 0] = 0
            loss[loss > 0] = 0
            loss = -loss  # Make losses positive
            
            avg_gain = gain.rolling(window=params["rsi_period"]).mean()
            avg_loss = loss.rolling(window=params["rsi_period"]).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            rsi = rsi.dropna()
            
            if not rsi.empty:
                last_rsi = rsi.iloc[-1]
                
                # Oversold condition (potential buy)
                if last_rsi < params["rsi_oversold"]:
                    # Calculate signal strength based on how oversold
                    strength = min(0.9, 0.5 + (params["rsi_oversold"] - last_rsi) / 30)
                    
                    signals['rsi_oversold'] = {
                        'action': 'buy',
                        'strength': strength,
                        'reason': f'RSI oversold ({last_rsi:.1f})',
                        'rsi': last_rsi
                    }
                
                # Overbought condition (potential sell)
                elif last_rsi > params["rsi_overbought"]:
                    # Calculate signal strength based on how overbought
                    strength = min(0.9, 0.5 + (last_rsi - params["rsi_overbought"]) / 30)
                    
                    signals['rsi_overbought'] = {
                        'action': 'sell',
                        'strength': strength,
                        'reason': f'RSI overbought ({last_rsi:.1f})',
                        'rsi': last_rsi
                    }
        
        # Calculate Rate of Change (ROC)
        if len(close) >= params["roc_period"]:
            roc = (close / close.shift(params["roc_period"]) - 1) * 100
            roc = roc.dropna()
            
            if not roc.empty:
                last_roc = roc.iloc[-1]
                
                # Strong positive momentum
                if last_roc > params["roc_threshold"] * 100:
                    # Calculate signal strength based on ROC magnitude
                    strength = min(0.8, 0.4 + last_roc / 20)
                    
                    signals['roc_strong_positive'] = {
                        'action': 'buy',
                        'strength': strength,
                        'reason': f'Strong positive momentum (ROC: {last_roc:.1f}%)',
                        'roc': last_roc
                    }
                
                # Strong negative momentum
                elif last_roc < -params["roc_threshold"] * 100:
                    # Calculate signal strength based on ROC magnitude
                    strength = min(0.8, 0.4 + abs(last_roc) / 20)
                    
                    signals['roc_strong_negative'] = {
                        'action': 'sell',
                        'strength': strength,
                        'reason': f'Strong negative momentum (ROC: {last_roc:.1f}%)',
                        'roc': last_roc
                    }
        
        return signals
