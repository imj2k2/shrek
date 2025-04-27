from indicators import options
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from mcp.protocol import MCPMessage
from datetime import datetime
import logging

class OptionsAgent:
    """Agent for options trading strategies including volatility and hedging"""
    
    def __init__(self, mcp=None):
        self.logger = logging.getLogger("OptionsAgent")
        self.mcp = mcp  # Model Context Protocol for communication
        
        # Strategy parameters
        self.strategies = {
            "volatility": {
                "enabled": True,
                "weight": 0.5,  # Strategy weight in overall decision
                "params": {
                    "high_iv_threshold": 0.4,  # IV above this is considered high
                    "low_iv_threshold": 0.2,   # IV below this is considered low
                    "vix_high_threshold": 25,  # VIX above this is considered high volatility
                    "vix_low_threshold": 15    # VIX below this is considered low volatility
                }
            },
            "income": {
                "enabled": True,
                "weight": 0.3,
                "params": {
                    "min_premium_pct": 0.02,  # Minimum premium as % of underlying
                    "days_to_expiration": {
                        "min": 14,
                        "max": 45
                    },
                    "delta_threshold": {
                        "call": 0.3,  # For covered calls
                        "put": 0.3    # For cash-secured puts
                    }
                }
            },
            "hedging": {
                "enabled": True,
                "weight": 0.2,
                "params": {
                    "portfolio_beta_threshold": 1.2,  # Hedge if portfolio beta exceeds this
                    "hedge_ratio": 0.5,              # Hedge this portion of portfolio
                    "put_delta_target": -0.3         # Target delta for protective puts
                }
            }
        }
        
        # Position sizing parameters
        self.position_sizing = {
            "max_position_pct": 0.03,  # Max 3% of portfolio in any single options position
            "max_total_options_pct": 0.20  # Max 20% of portfolio in all options combined
        }
        
        # Tracking variables
        self.last_signals = {}
        self.performance_metrics = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "avg_profit": 0.0
        }
    
    def generate_signals(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate options trading signals based on multiple strategies"""
        # Validate input data
        required_fields = ['S', 'K', 'T', 'r', 'sigma']
        if not data or not all(k in data for k in required_fields):
            self.logger.error("Invalid data format: missing required options data fields")
            return {'error': 'Invalid data format: missing required options data fields'}
        
        try:
            # Extract basic options data
            S = data['S']  # Underlying price
            K = data['K']  # Strike price
            T = data['T']  # Time to expiration (in years)
            r = data['r']  # Risk-free rate
            sigma = data['sigma']  # Volatility
            symbol = data.get('symbol', 'UNKNOWN')
            option_type = data.get('option_type', 'call')
            option_price = data.get('option_price', None)
            vix = data.get('vix', None)
            put_call_ratio = data.get('put_call_ratio', None)
            
            # Calculate Greeks
            greeks = options.greeks(S, K, T, r, sigma)
            
            # Calculate implied volatility if option price is provided
            iv = None
            if option_price:
                iv = options.implied_volatility(option_price, S, K, T, r)
            
            # Generate signals from each strategy
            volatility_signals = self._volatility_strategy(S, K, T, r, sigma, iv, vix, put_call_ratio) \
                if self.strategies["volatility"]["enabled"] else {}
                
            income_signals = self._income_strategy(S, K, T, r, sigma, greeks, option_type) \
                if self.strategies["income"]["enabled"] else {}
                
            hedging_signals = self._hedging_strategy(S, K, T, r, sigma, greeks, data.get('portfolio_beta', 1.0)) \
                if self.strategies["hedging"]["enabled"] else {}
            
            # Combine signals with weights
            combined_signals = self._combine_signals({
                "volatility": volatility_signals,
                "income": income_signals,
                "hedging": hedging_signals
            })
            
            # Add position sizing
            final_signals = self._apply_position_sizing(combined_signals, data)
            
            # Add metadata and Greeks
            final_signals['timestamp'] = datetime.now().isoformat()
            final_signals['symbol'] = symbol
            final_signals['greeks'] = greeks
            if iv is not None:
                final_signals['iv'] = iv
            
            # Store last signals
            self.last_signals = final_signals
            
            # Send to MCP if available
            if self.mcp:
                message = MCPMessage(
                    sender="options_agent",
                    message_type="signal",
                    content=final_signals
                )
                self.mcp.send_message(message)
            
            return final_signals
            
        except Exception as e:
            self.logger.error(f"Error generating options signals: {str(e)}")
            return {'error': f'Signal generation failed: {str(e)}'}
    
    def _volatility_strategy(self, S, K, T, r, sigma, iv, vix, put_call_ratio) -> Dict[str, Any]:
        """Volatility-based options strategies"""
        signals = {}
        params = self.strategies["volatility"]["params"]
        
        # Implied Volatility Strategy
        if iv is not None:
            if iv > params["high_iv_threshold"]:
                # High IV environment - good for selling options
                if K > S:  # OTM call
                    signals['iv_high_call'] = {
                        'action': 'sell',
                        'option_type': 'call',
                        'strategy': 'covered_call',
                        'strength': min(0.9, (iv - params["high_iv_threshold"]) * 2),
                        'reason': f'High IV ({iv:.2f}) - sell covered calls',
                        'iv': iv
                    }
                else:  # OTM put
                    signals['iv_high_put'] = {
                        'action': 'sell',
                        'option_type': 'put',
                        'strategy': 'cash_secured_put',
                        'strength': min(0.9, (iv - params["high_iv_threshold"]) * 2),
                        'reason': f'High IV ({iv:.2f}) - sell cash-secured puts',
                        'iv': iv
                    }
            elif iv < params["low_iv_threshold"]:
                # Low IV environment - good for buying options
                signals['iv_low'] = {
                    'action': 'buy',
                    'option_type': 'call',  # Default to calls in low IV
                    'strategy': 'long_call',
                    'strength': min(0.8, (params["low_iv_threshold"] - iv) * 3),
                    'reason': f'Low IV ({iv:.2f}) - buy calls for potential volatility expansion',
                    'iv': iv
                }
        
        # VIX Strategy
        if vix is not None:
            if vix > params["vix_high_threshold"]:
                # High VIX - market expects high volatility
                signals['vix_high'] = {
                    'action': 'buy',
                    'option_type': 'put',
                    'strategy': 'protective_put',
                    'strength': min(0.9, (vix - params["vix_high_threshold"]) / 10),
                    'reason': f'High VIX ({vix:.1f}) - buy protective puts',
                    'vix': vix
                }
            elif vix < params["vix_low_threshold"]:
                # Low VIX - market expects low volatility
                signals['vix_low'] = {
                    'action': 'sell',
                    'option_type': 'call',
                    'strategy': 'covered_call',
                    'strength': min(0.8, (params["vix_low_threshold"] - vix) / 5),
                    'reason': f'Low VIX ({vix:.1f}) - sell covered calls',
                    'vix': vix
                }
        
        # Put-Call Ratio Strategy
        if put_call_ratio is not None:
            if put_call_ratio > 1.2:  # High put-call ratio - bearish sentiment
                signals['high_put_call'] = {
                    'action': 'buy',
                    'option_type': 'put',
                    'strategy': 'long_put',
                    'strength': min(0.7, (put_call_ratio - 1.2) * 0.5),
                    'reason': f'High put-call ratio ({put_call_ratio:.2f}) - bearish sentiment',
                    'put_call_ratio': put_call_ratio
                }
            elif put_call_ratio < 0.7:  # Low put-call ratio - bullish sentiment
                signals['low_put_call'] = {
                    'action': 'buy',
                    'option_type': 'call',
                    'strategy': 'long_call',
                    'strength': min(0.7, (0.7 - put_call_ratio) * 1.0),
                    'reason': f'Low put-call ratio ({put_call_ratio:.2f}) - bullish sentiment',
                    'put_call_ratio': put_call_ratio
                }
        
        return signals
    
    def _income_strategy(self, S, K, T, r, sigma, greeks, option_type) -> Dict[str, Any]:
        """Income-generating options strategies"""
        signals = {}
        params = self.strategies["income"]["params"]
        
        # Extract Greeks
        delta = greeks.get('delta', 0)
        theta = greeks.get('theta', 0)
        
        # Calculate days to expiration
        dte = int(T * 365)
        
        # Calculate option premium (rough estimate)
        if option_type == 'call':
            premium = options.black_scholes(S, K, T, r, sigma, option_type='call')
        else:  # put
            premium = options.black_scholes(S, K, T, r, sigma, option_type='put')
        
        premium_pct = premium / S
        
        # Covered Call Strategy
        if option_type == 'call' and K > S:  # OTM call
            if (delta < params["delta_threshold"]["call"] and 
                premium_pct >= params["min_premium_pct"] and 
                params["days_to_expiration"]["min"] <= dte <= params["days_to_expiration"]["max"]):
                
                signals['covered_call'] = {
                    'action': 'sell',
                    'option_type': 'call',
                    'strategy': 'covered_call',
                    'strength': 0.8,
                    'reason': f'Optimal covered call: Delta={delta:.2f}, Premium={premium_pct:.2%}, DTE={dte}',
                    'delta': delta,
                    'premium': premium,
                    'premium_pct': premium_pct,
                    'dte': dte
                }
        
        # Cash-Secured Put Strategy
        elif option_type == 'put' and K < S:  # OTM put
            if (abs(delta) < params["delta_threshold"]["put"] and 
                premium_pct >= params["min_premium_pct"] and 
                params["days_to_expiration"]["min"] <= dte <= params["days_to_expiration"]["max"]):
                
                signals['cash_secured_put'] = {
                    'action': 'sell',
                    'option_type': 'put',
                    'strategy': 'cash_secured_put',
                    'strength': 0.8,
                    'reason': f'Optimal cash-secured put: Delta={delta:.2f}, Premium={premium_pct:.2%}, DTE={dte}',
                    'delta': delta,
                    'premium': premium,
                    'premium_pct': premium_pct,
                    'dte': dte
                }
        
        # Iron Condor (if we have high theta)
        if theta > 0 and params["days_to_expiration"]["min"] <= dte <= params["days_to_expiration"]["max"]:
            signals['iron_condor'] = {
                'action': 'sell',
                'option_type': 'iron_condor',
                'strategy': 'iron_condor',
                'strength': min(0.7, theta / 10),  # Higher theta = stronger signal
                'reason': f'Iron condor opportunity: Theta={theta:.2f}, DTE={dte}',
                'theta': theta,
                'dte': dte
            }
        
        return signals
    
    def _hedging_strategy(self, S, K, T, r, sigma, greeks, portfolio_beta) -> Dict[str, Any]:
        """Hedging options strategies"""
        signals = {}
        params = self.strategies["hedging"]["params"]
        
        # Extract Greeks
        delta = greeks.get('delta', 0)
        gamma = greeks.get('gamma', 0)
        
        # Protective Put Strategy
        if portfolio_beta > params["portfolio_beta_threshold"]:
            # Portfolio has high beta, consider protective puts
            if option_type == 'put' and delta < 0:  # Only consider puts
                target_delta = params["put_delta_target"]
                delta_diff = abs(delta - target_delta)
                
                if delta_diff < 0.1:  # Close to our target delta
                    signals['protective_put'] = {
                        'action': 'buy',
                        'option_type': 'put',
                        'strategy': 'protective_put',
                        'strength': 0.9 - delta_diff,  # Higher strength for closer delta match
                        'reason': f'Portfolio beta ({portfolio_beta:.2f}) exceeds threshold, buy protective puts',
                        'delta': delta,
                        'portfolio_beta': portfolio_beta,
                        'hedge_ratio': params["hedge_ratio"]
                    }
        
        # Collar Strategy (for existing positions with high gamma)
        if gamma > 0.05 and option_type == 'call':
            signals['collar'] = {
                'action': 'sell',
                'option_type': 'collar',
                'strategy': 'collar',
                'strength': min(0.7, gamma * 5),
                'reason': f'High gamma ({gamma:.3f}), implement collar to reduce risk',
                'gamma': gamma
            }
        
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
                weighted_signal['strategy_name'] = strategy
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
        
        # Determine best strategy
        best_strategy = None
        if action == 'buy' and buy_signals:
            best_strategy = buy_signals[0]['strategy']
        elif action == 'sell' and sell_signals:
            best_strategy = sell_signals[0]['strategy']
        
        # Combine into final signal
        combined = {
            'action': action,
            'option_type': buy_signals[0]['option_type'] if action == 'buy' and buy_signals else 
                          sell_signals[0]['option_type'] if action == 'sell' and sell_signals else None,
            'strategy': best_strategy,
            'buy_strength': buy_strength,
            'sell_strength': sell_strength,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'net_strength': buy_strength - sell_strength
        }
        
        return combined
    
    def _apply_position_sizing(self, signals: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply position sizing to the signals"""
        if signals['action'] == 'hold' or not signals.get('strategy'):
            return signals
        
        # In a real implementation, this would use portfolio value, current positions, etc.
        # For now, we'll use a simplified approach
        
        # Mock portfolio value (in production, get from broker)
        portfolio_value = 100000  # $100k portfolio
        
        # Calculate max position size based on our risk parameters
        max_position_value = portfolio_value * self.position_sizing["max_position_pct"]
        
        # For options, we'll calculate contracts based on strategy
        strategy = signals['strategy']
        option_type = signals.get('option_type')
        
        # Extract price data
        S = data['S']  # Underlying price
        K = data['K']  # Strike price
        
        # Calculate option price (rough estimate)
        if option_type in ['call', 'put']:
            option_price = data.get('option_price', options.black_scholes(
                S, K, data['T'], data['r'], data['sigma'], option_type=option_type))
        else:
            # For complex strategies like iron condors, just use a placeholder
            option_price = S * 0.03  # Roughly 3% of underlying
        
        # Calculate number of contracts
        # Each contract is for 100 shares
        contract_value = option_price * 100
        max_contracts = int(max_position_value / contract_value)
        
        # Ensure at least 1 contract, but no more than 10 for safety
        num_contracts = max(1, min(max_contracts, 10))
        
        # Add position sizing to signals
        signals['qty'] = num_contracts
        signals['price'] = option_price
        signals['contract_value'] = contract_value
        signals['total_value'] = contract_value * num_contracts
        signals['underlying_price'] = S
        signals['strike_price'] = K
        
        return signals
    
    def update_performance(self, trade_result: Dict[str, Any]):
        """Update agent performance metrics based on trade results"""
        self.performance_metrics["total_trades"] += 1
        
        profit = trade_result.get('profit', 0)
        if profit > 0:
            self.performance_metrics["winning_trades"] += 1
        else:
            self.performance_metrics["losing_trades"] += 1
        
        # Update win rate
        if self.performance_metrics["total_trades"] > 0:
            self.performance_metrics["win_rate"] = self.performance_metrics["winning_trades"] / self.performance_metrics["total_trades"]
        
        # Update average profit
        total_profit = self.performance_metrics.get("avg_profit", 0) * (self.performance_metrics["total_trades"] - 1)
        total_profit += profit
        self.performance_metrics["avg_profit"] = total_profit / self.performance_metrics["total_trades"]
        
        # In a real implementation, we might adjust strategy weights based on performance
        self._adjust_strategies_based_on_performance()
    
    def _adjust_strategies_based_on_performance(self):
        """Dynamically adjust strategy weights based on performance"""
        # This would be implemented in a production system
        # For now, it's a placeholder
        pass
