"""
Portfolio backtester for testing multiple strategies and assets with customizable weights.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
import logging
import json
from datetime import datetime, timedelta
import os
import pickle
from pathlib import Path

from backtest.backtest_engine import Backtester
from agents.customizable_agent import CustomizableAgent
from agents.stocks_agent import StocksAgent
from agents.llm_insights import LLMInsightGenerator

class PortfolioBacktester:
    """
    Backtester for portfolios of multiple assets and strategies with customizable weights.
    """
    
    def __init__(self, data_fetcher=None, broker=None):
        self.logger = logging.getLogger("PortfolioBacktester")
        self.backtester = Backtester(data_fetcher, broker)
        self.llm_insight_generator = LLMInsightGenerator()
        
        # Cache for portfolio backtest results
        self.cache_dir = Path("./cache/portfolio_backtest")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.results_cache = {}
    
    def run_portfolio_backtest(self, 
                              portfolio_config: Dict[str, Any],
                              start_date: str,
                              end_date: str,
                              generate_insights: bool = False) -> Dict[str, Any]:
        """
        Run a backtest on a portfolio of assets and strategies
        
        Parameters:
        - portfolio_config: Dictionary containing:
            - allocations: Dict mapping symbols to allocation weights
            - strategies: Dict mapping strategy names to configurations
        - start_date: Start date in YYYY-MM-DD format
        - end_date: End date in YYYY-MM-DD format
        - generate_insights: Whether to generate LLM insights
        
        Returns:
        - Dictionary with portfolio backtest results
        """
        # Validate inputs
        if not portfolio_config.get("allocations"):
            return {"error": "No allocations provided in portfolio configuration"}
        
        # Check cache
        cache_key = self._get_cache_key(portfolio_config, start_date, end_date)
        cached_result = self._check_cache(cache_key)
        if cached_result:
            self.logger.info(f"Using cached portfolio backtest result for {cache_key}")
            
            # Add insights if requested and not already in cache
            if generate_insights and "insights" not in cached_result:
                try:
                    insights = self.llm_insight_generator.generate_portfolio_insight(
                        portfolio_config, cached_result
                    )
                    cached_result["insights"] = insights
                    self._save_to_cache(cache_key, cached_result)
                except Exception as e:
                    self.logger.error(f"Error generating insights: {str(e)}")
                    cached_result["insights"] = f"Error generating insights: {str(e)}"
            
            return cached_result
        
        # Extract allocations and normalize weights
        allocations = portfolio_config.get("allocations", {})
        total_weight = sum(allocations.values())
        if total_weight <= 0:
            return {"error": "Total allocation weight must be positive"}
        
        # Normalize weights to sum to 1.0
        normalized_allocations = {symbol: weight / total_weight for symbol, weight in allocations.items()}
        
        # Run individual backtests for each symbol and strategy
        symbol_results = {}
        data_sources = {}
        
        for symbol, weight in normalized_allocations.items():
            # Get strategy for this symbol
            strategy_name = portfolio_config.get("symbol_strategies", {}).get(symbol, "customizable")
            
            # Configure agent based on strategy
            if strategy_name == "customizable":
                agent = CustomizableAgent(portfolio_config.get("strategies", {}))
            else:
                # Use default StocksAgent with specified strategy
                agent = StocksAgent(strategy=strategy_name)
            
            # Run backtest for this symbol
            try:
                result = self.backtester.run_backtest(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    agent=agent,
                    initial_capital=10000,  # Use a standard capital for each symbol
                    commission=0.001
                )
                
                if "error" in result:
                    self.logger.warning(f"Error in backtest for {symbol}: {result['error']}")
                    continue
                
                symbol_results[symbol] = result
                
                # Track data source
                if "data_source" in result:
                    data_sources[symbol] = result["data_source"]
                
            except Exception as e:
                self.logger.error(f"Error running backtest for {symbol}: {str(e)}")
                continue
        
        # Combine results into portfolio performance
        portfolio_result = self._combine_results(
            symbol_results, normalized_allocations, start_date, end_date
        )
        
        # Add metadata
        portfolio_result["portfolio_config"] = portfolio_config
        portfolio_result["start_date"] = start_date
        portfolio_result["end_date"] = end_date
        portfolio_result["data_sources"] = data_sources
        portfolio_result["symbols"] = list(normalized_allocations.keys())
        
        # Generate insights if requested
        if generate_insights:
            try:
                insights = self.llm_insight_generator.generate_portfolio_insight(
                    portfolio_config, portfolio_result
                )
                portfolio_result["insights"] = insights
            except Exception as e:
                self.logger.error(f"Error generating insights: {str(e)}")
                portfolio_result["insights"] = f"Error generating insights: {str(e)}"
        
        # Save to cache
        self._save_to_cache(cache_key, portfolio_result)
        
        return portfolio_result
    
    def _combine_results(self, 
                        symbol_results: Dict[str, Dict[str, Any]], 
                        allocations: Dict[str, float],
                        start_date: str,
                        end_date: str) -> Dict[str, Any]:
        """
        Combine individual backtest results into portfolio performance
        
        Parameters:
        - symbol_results: Dictionary mapping symbols to backtest results
        - allocations: Dictionary mapping symbols to allocation weights
        - start_date: Start date in YYYY-MM-DD format
        - end_date: End date in YYYY-MM-DD format
        
        Returns:
        - Dictionary with combined portfolio results
        """
        if not symbol_results:
            return {"error": "No valid backtest results to combine"}
        
        # Create a common date range for all symbols
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        date_range = pd.date_range(start=start, end=end, freq='D')
        
        # Initialize portfolio equity curve
        portfolio_equity = pd.Series(0.0, index=date_range)
        
        # Combine equity curves
        for symbol, result in symbol_results.items():
            if "equity_curve" not in result:
                continue
            
            weight = allocations.get(symbol, 0.0)
            if weight <= 0:
                continue
            
            # Convert equity curve to Series if it's a list
            if isinstance(result["equity_curve"], list):
                # Convert list of [date, value] to Series
                dates = [pd.to_datetime(item[0]) for item in result["equity_curve"]]
                values = [item[1] for item in result["equity_curve"]]
                symbol_equity = pd.Series(values, index=dates)
            elif isinstance(result["equity_curve"], pd.Series):
                symbol_equity = result["equity_curve"]
            else:
                self.logger.warning(f"Unknown equity curve format for {symbol}")
                continue
            
            # Reindex to common date range, forward fill missing values
            symbol_equity = symbol_equity.reindex(date_range, method='ffill')
            
            # Normalize to start at 1.0
            if not symbol_equity.empty and symbol_equity.iloc[0] > 0:
                symbol_equity = symbol_equity / symbol_equity.iloc[0]
            
            # Add weighted contribution to portfolio
            portfolio_equity += symbol_equity * weight
        
        # Calculate portfolio metrics
        metrics = self._calculate_portfolio_metrics(portfolio_equity)
        
        # Convert equity curve to list format for JSON serialization
        equity_curve_list = [[date.strftime('%Y-%m-%d'), float(value)] 
                            for date, value in portfolio_equity.items()]
        
        # Prepare result
        result = {
            "metrics": metrics,
            "equity_curve": equity_curve_list
        }
        
        return result
    
    def _calculate_portfolio_metrics(self, equity_curve: pd.Series) -> Dict[str, float]:
        """
        Calculate performance metrics for a portfolio equity curve
        
        Parameters:
        - equity_curve: Series of portfolio values indexed by date
        
        Returns:
        - Dictionary of performance metrics
        """
        if equity_curve.empty:
            return {
                "total_return": 0.0,
                "annualized_return": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "volatility": 0.0
            }
        
        # Calculate returns
        returns = equity_curve.pct_change().dropna()
        
        # Total return
        if equity_curve.iloc[0] > 0:
            total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        else:
            total_return = 0.0
        
        # Annualized return
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        if days > 0:
            annualized_return = (1 + total_return) ** (365 / days) - 1
        else:
            annualized_return = 0.0
        
        # Volatility (annualized)
        if len(returns) > 1:
            volatility = returns.std() * np.sqrt(252)  # Assuming 252 trading days per year
        else:
            volatility = 0.0
        
        # Sharpe ratio (assuming risk-free rate of 0)
        if volatility > 0:
            sharpe_ratio = annualized_return / volatility
        else:
            sharpe_ratio = 0.0
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max) - 1
        max_drawdown = drawdown.min() if not drawdown.empty else 0.0
        
        # Win rate (if we have trades)
        win_rate = 0.0
        
        return {
            "total_return": float(total_return),
            "annualized_return": float(annualized_return),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown),
            "volatility": float(volatility),
            "win_rate": float(win_rate)
        }
    
    def _get_cache_key(self, portfolio_config: Dict[str, Any], start_date: str, end_date: str) -> str:
        """Generate a cache key for the portfolio backtest"""
        # Create a simplified config for the cache key
        key_config = {
            "allocations": portfolio_config.get("allocations", {}),
            "symbol_strategies": portfolio_config.get("symbol_strategies", {}),
            "strategies": portfolio_config.get("strategies", {})
        }
        
        # Create a hash of the config
        config_str = json.dumps(key_config, sort_keys=True)
        import hashlib
        config_hash = hashlib.md5(config_str.encode()).hexdigest()
        
        return f"{config_hash}_{start_date}_{end_date}"
    
    def _check_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Check if a result exists in cache"""
        # Check memory cache first
        if cache_key in self.results_cache:
            return self.results_cache[cache_key]
        
        # Check file cache
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    result = pickle.load(f)
                # Store in memory cache
                self.results_cache[cache_key] = result
                return result
            except Exception as e:
                self.logger.error(f"Error loading cache file: {str(e)}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, result: Dict[str, Any]):
        """Save a result to cache"""
        # Save to memory cache
        self.results_cache[cache_key] = result
        
        # Save to file cache
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            self.logger.error(f"Error saving to cache file: {str(e)}")
    
    def clear_cache(self):
        """Clear the portfolio backtest cache"""
        self.results_cache = {}
        
        # Clear file cache
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                os.remove(cache_file)
            except Exception as e:
                self.logger.error(f"Error removing cache file {cache_file}: {str(e)}")
                
    def get_available_strategies(self) -> Dict[str, List[str]]:
        """Get available strategies for portfolio backtesting"""
        # Get strategies from CustomizableAgent
        customizable_agent = CustomizableAgent()
        customizable_strategies = customizable_agent.get_available_strategies()
        
        # Get strategies from StocksAgent
        stocks_agent = StocksAgent()
        stocks_strategies = ["momentum", "mean_reversion", "breakout"]
        
        return {
            "customizable": customizable_strategies,
            "stocks": stocks_strategies
        }
