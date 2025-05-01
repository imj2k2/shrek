"""
Customizable trading agent that can use different combinations of strategies.
"""

from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
import logging
from datetime import datetime

from agents.strategy_library import StrategyLibrary

class CustomizableAgent:
    """
    A customizable trading agent that can use different combinations of strategies
    and technical indicators based on user preferences.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger("CustomizableAgent")
        self.strategy_library = StrategyLibrary()
        
        # Default configuration
        self.default_config = {
            "strategies": {
                "mean_reversion": {
                    "enabled": True,
                    "weight": 1.0,
                    "params": {
                        "bollinger_period": 20,
                        "bollinger_std": 2.0,
                        "oversold_threshold": 0.05,
                        "overbought_threshold": 0.05
                    }
                },
                "trend_following": {
                    "enabled": True,
                    "weight": 1.0,
                    "params": {
                        "fast_period": 20,
                        "slow_period": 50,
                        "signal_threshold": 0.05
                    }
                },
                "breakout": {
                    "enabled": True,
                    "weight": 1.0,
                    "params": {
                        "lookback_period": 20,
                        "breakout_threshold": 0.03,
                        "volume_factor": 1.5
                    }
                },
                "volatility": {
                    "enabled": True,
                    "weight": 0.8,
                    "params": {
                        "atr_period": 14,
                        "volatility_threshold": 0.03,
                        "low_volatility_threshold": 0.01
                    }
                },
                "momentum": {
                    "enabled": True,
                    "weight": 1.0,
                    "params": {
                        "rsi_period": 14,
                        "roc_period": 10,
                        "rsi_oversold": 30,
                        "rsi_overbought": 70,
                        "roc_threshold": 0.05
                    }
                }
            },
            "position_sizing": {
                "max_position_size": 0.2,  # Maximum position size as fraction of portfolio
                "signal_threshold": 0.5,   # Minimum signal strength to take action
                "scale_by_strength": True  # Scale position size by signal strength
            },
            "risk_management": {
                "stop_loss": 0.05,         # Stop loss as fraction of entry price
                "take_profit": 0.10,       # Take profit as fraction of entry price
                "trailing_stop": 0.03      # Trailing stop as fraction of highest price
            }
        }
        
        # Use provided config or defaults
        self.config = self.default_config.copy()
        if config:
            self.update_config(config)
    
    def update_config(self, config: Dict[str, Any]):
        """Update the agent configuration with new values"""
        # Update strategies
        if "strategies" in config:
            for strategy_name, strategy_config in config["strategies"].items():
                if strategy_name in self.config["strategies"]:
                    # Update existing strategy
                    for key, value in strategy_config.items():
                        if key == "params" and isinstance(value, dict):
                            # Update params
                            self.config["strategies"][strategy_name]["params"].update(value)
                        else:
                            # Update other settings
                            self.config["strategies"][strategy_name][key] = value
                else:
                    # Add new strategy
                    self.config["strategies"][strategy_name] = strategy_config
        
        # Update position sizing
        if "position_sizing" in config:
            self.config["position_sizing"].update(config["position_sizing"])
        
        # Update risk management
        if "risk_management" in config:
            self.config["risk_management"].update(config["risk_management"])
    
    def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market data and generate trading signals based on configured strategies
        
        Parameters:
        - data: Dictionary containing market data including:
            - close: Series of closing prices
            - open: Series of opening prices (optional)
            - high: Series of high prices (optional)
            - low: Series of low prices (optional)
            - volume: Series of volume data (optional)
            - symbol: Symbol being analyzed
            - date: Current date
        
        Returns:
        - Dictionary with trading decision and supporting information
        """
        # Extract data
        symbol = data.get('symbol', 'UNKNOWN')
        date = data.get('date', datetime.now().strftime('%Y-%m-%d'))
        
        # Required data
        close = data.get('close')
        if close is None:
            self.logger.error(f"Missing required close price data for {symbol}")
            return {'action': 'hold', 'confidence': 0, 'reason': 'Missing data'}
        
        # Optional data
        open_prices = data.get('open')
        high = data.get('high')
        low = data.get('low')
        volume = data.get('volume')
        
        # Collect signals from all enabled strategies
        all_signals = {}
        
        # Run each enabled strategy
        for strategy_name, strategy_config in self.config["strategies"].items():
            if strategy_config["enabled"]:
                try:
                    # Call the appropriate strategy method
                    if strategy_name == "mean_reversion":
                        signals = self.strategy_library.mean_reversion_strategy(
                            close, high, low, strategy_config["params"]
                        )
                    elif strategy_name == "trend_following":
                        signals = self.strategy_library.trend_following_strategy(
                            close, strategy_config["params"]
                        )
                    elif strategy_name == "breakout":
                        signals = self.strategy_library.breakout_strategy(
                            close, high, low, volume, strategy_config["params"]
                        )
                    elif strategy_name == "volatility":
                        signals = self.strategy_library.volatility_strategy(
                            close, high, low, strategy_config["params"]
                        )
                    elif strategy_name == "momentum":
                        signals = self.strategy_library.momentum_strategy(
                            close, strategy_config["params"]
                        )
                    else:
                        self.logger.warning(f"Unknown strategy: {strategy_name}")
                        signals = {}
                    
                    # Apply strategy weight to signal strengths
                    for signal_name, signal_data in signals.items():
                        signal_data['strength'] *= strategy_config["weight"]
                        all_signals[f"{strategy_name}_{signal_name}"] = signal_data
                
                except Exception as e:
                    self.logger.error(f"Error in {strategy_name} strategy: {str(e)}")
        
        # Aggregate signals
        return self._aggregate_signals(all_signals, symbol, date)
    
    def _aggregate_signals(self, signals: Dict[str, Dict[str, Any]], symbol: str, date: str) -> Dict[str, Any]:
        """
        Aggregate signals from different strategies into a final decision
        
        Parameters:
        - signals: Dictionary of signals from different strategies
        - symbol: Symbol being analyzed
        - date: Current date
        
        Returns:
        - Dictionary with trading decision and supporting information
        """
        if not signals:
            return {
                'action': 'hold',
                'confidence': 0,
                'reason': 'No signals generated',
                'symbol': symbol,
                'date': date
            }
        
        # Separate buy and sell signals
        buy_signals = {k: v for k, v in signals.items() if v['action'] == 'buy'}
        sell_signals = {k: v for k, v in signals.items() if v['action'] == 'sell'}
        
        # Calculate total buy and sell strength
        buy_strength = sum(s['strength'] for s in buy_signals.values())
        sell_strength = sum(s['strength'] for s in sell_signals.values())
        
        # Determine action based on signal strengths
        threshold = self.config["position_sizing"]["signal_threshold"]
        
        if buy_strength > sell_strength and buy_strength >= threshold:
            action = 'buy'
            confidence = buy_strength
            # Get top reasons
            top_signals = sorted(buy_signals.items(), key=lambda x: x[1]['strength'], reverse=True)[:3]
            reasons = [f"{s[0]}: {s[1]['reason']}" for s in top_signals]
        elif sell_strength > buy_strength and sell_strength >= threshold:
            action = 'sell'
            confidence = sell_strength
            # Get top reasons
            top_signals = sorted(sell_signals.items(), key=lambda x: x[1]['strength'], reverse=True)[:3]
            reasons = [f"{s[0]}: {s[1]['reason']}" for s in top_signals]
        else:
            action = 'hold'
            confidence = max(buy_strength, sell_strength)
            reasons = ['No strong signals']
        
        # Calculate position size
        position_size = 0
        if action != 'hold':
            max_size = self.config["position_sizing"]["max_position_size"]
            if self.config["position_sizing"]["scale_by_strength"]:
                position_size = max_size * min(confidence, 1.0)
            else:
                position_size = max_size
        
        # Prepare result
        result = {
            'action': action,
            'confidence': confidence,
            'position_size': position_size,
            'reason': '; '.join(reasons),
            'symbol': symbol,
            'date': date,
            'signals': signals,
            'buy_strength': buy_strength,
            'sell_strength': sell_strength
        }
        
        return result
    
    def get_available_strategies(self) -> List[str]:
        """Return a list of available strategy names"""
        return list(self.config["strategies"].keys())
    
    def get_strategy_parameters(self, strategy_name: str) -> Dict[str, Any]:
        """Return the parameters for a specific strategy"""
        if strategy_name in self.config["strategies"]:
            return self.config["strategies"][strategy_name]["params"].copy()
        else:
            return {}
    
    def get_config(self) -> Dict[str, Any]:
        """Return the current configuration"""
        return self.config.copy()
    
    def reset(self):
        """Reset the agent to default configuration"""
        self.config = self.default_config.copy()
