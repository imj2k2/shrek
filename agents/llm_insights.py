"""
LLM-based insights for trading strategies and backtesting results.
"""

import requests
import json
import logging
from typing import Dict, Any, List, Optional
import os

class LLMInsightGenerator:
    """
    Generate insights on trading strategies and backtest results using LLM.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.logger = logging.getLogger("LLMInsightGenerator")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.api_url = "https://api.openai.com/v1/chat/completions"
    
    def generate_strategy_insight(self, 
                                 strategy_config: Dict[str, Any], 
                                 backtest_results: Dict[str, Any]) -> str:
        """
        Generate insights on a trading strategy based on its configuration and backtest results.
        
        Parameters:
        - strategy_config: Configuration of the strategy
        - backtest_results: Results from backtesting the strategy
        
        Returns:
        - String containing insights on the strategy
        """
        if not self.api_key:
            return "API key not provided. Please set OPENAI_API_KEY environment variable or provide it during initialization."
        
        # Prepare the prompt
        prompt = self._prepare_strategy_insight_prompt(strategy_config, backtest_results)
        
        # Call the LLM API
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": "You are a professional trading strategy analyst with expertise in technical analysis, quantitative finance, and algorithmic trading. Provide concise, actionable insights on trading strategies and backtest results."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 1000
            }
            
            response = requests.post(self.api_url, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                insight = result["choices"][0]["message"]["content"]
                return insight
            else:
                self.logger.error(f"API error: {response.status_code} - {response.text}")
                return f"Error generating insights: {response.status_code} - {response.reason}"
                
        except Exception as e:
            self.logger.error(f"Error calling LLM API: {str(e)}")
            return f"Error generating insights: {str(e)}"
    
    def generate_portfolio_insight(self, 
                                  portfolio_config: Dict[str, Any], 
                                  backtest_results: Dict[str, Any]) -> str:
        """
        Generate insights on a portfolio allocation based on its configuration and backtest results.
        
        Parameters:
        - portfolio_config: Configuration of the portfolio
        - backtest_results: Results from backtesting the portfolio
        
        Returns:
        - String containing insights on the portfolio
        """
        if not self.api_key:
            return "API key not provided. Please set OPENAI_API_KEY environment variable or provide it during initialization."
        
        # Prepare the prompt
        prompt = self._prepare_portfolio_insight_prompt(portfolio_config, backtest_results)
        
        # Call the LLM API
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": "You are a professional portfolio manager with expertise in asset allocation, risk management, and portfolio optimization. Provide concise, actionable insights on portfolio performance and allocation strategies."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 1000
            }
            
            response = requests.post(self.api_url, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                insight = result["choices"][0]["message"]["content"]
                return insight
            else:
                self.logger.error(f"API error: {response.status_code} - {response.text}")
                return f"Error generating insights: {response.status_code} - {response.reason}"
                
        except Exception as e:
            self.logger.error(f"Error calling LLM API: {str(e)}")
            return f"Error generating insights: {str(e)}"
    
    def _prepare_strategy_insight_prompt(self, 
                                        strategy_config: Dict[str, Any], 
                                        backtest_results: Dict[str, Any]) -> str:
        """
        Prepare a prompt for strategy insights.
        
        Parameters:
        - strategy_config: Configuration of the strategy
        - backtest_results: Results from backtesting the strategy
        
        Returns:
        - Prompt string for the LLM
        """
        # Extract key metrics
        metrics = backtest_results.get("metrics", {})
        total_return = metrics.get("total_return", 0) * 100  # Convert to percentage
        sharpe_ratio = metrics.get("sharpe_ratio", 0)
        max_drawdown = metrics.get("max_drawdown", 0) * 100  # Convert to percentage
        win_rate = metrics.get("win_rate", 0) * 100  # Convert to percentage
        
        # Extract strategy details
        strategies = strategy_config.get("strategies", {})
        enabled_strategies = [name for name, config in strategies.items() if config.get("enabled", False)]
        
        # Extract symbols and time period
        symbols = backtest_results.get("symbols", [])
        start_date = backtest_results.get("start_date", "")
        end_date = backtest_results.get("end_date", "")
        
        # Extract data sources
        data_sources = backtest_results.get("data_sources", {})
        
        # Build the prompt
        prompt = f"""
        Analyze the following trading strategy and provide insights:
        
        Strategy Overview:
        - Enabled strategies: {', '.join(enabled_strategies)}
        - Symbols: {', '.join(symbols)}
        - Time period: {start_date} to {end_date}
        - Data sources: {json.dumps(data_sources, indent=2)}
        
        Strategy Configuration:
        {json.dumps(strategy_config, indent=2)}
        
        Backtest Results:
        - Total return: {total_return:.2f}%
        - Sharpe ratio: {sharpe_ratio:.2f}
        - Max drawdown: {max_drawdown:.2f}%
        - Win rate: {win_rate:.2f}%
        
        Please provide:
        1. A brief assessment of the strategy's performance
        2. Strengths and weaknesses of the strategy
        3. Potential improvements or optimizations
        4. Risk assessment
        5. Market conditions where this strategy might perform well or poorly
        
        Keep your response concise and actionable.
        """
        
        return prompt
    
    def _prepare_portfolio_insight_prompt(self, 
                                         portfolio_config: Dict[str, Any], 
                                         backtest_results: Dict[str, Any]) -> str:
        """
        Prepare a prompt for portfolio insights.
        
        Parameters:
        - portfolio_config: Configuration of the portfolio
        - backtest_results: Results from backtesting the portfolio
        
        Returns:
        - Prompt string for the LLM
        """
        # Extract key metrics
        metrics = backtest_results.get("metrics", {})
        total_return = metrics.get("total_return", 0) * 100  # Convert to percentage
        sharpe_ratio = metrics.get("sharpe_ratio", 0)
        max_drawdown = metrics.get("max_drawdown", 0) * 100  # Convert to percentage
        volatility = metrics.get("volatility", 0) * 100  # Convert to percentage
        
        # Extract portfolio details
        allocations = portfolio_config.get("allocations", {})
        strategies = portfolio_config.get("strategies", {})
        
        # Extract time period
        start_date = backtest_results.get("start_date", "")
        end_date = backtest_results.get("end_date", "")
        
        # Build the prompt
        prompt = f"""
        Analyze the following portfolio allocation and provide insights:
        
        Portfolio Overview:
        - Time period: {start_date} to {end_date}
        
        Asset Allocations:
        {json.dumps(allocations, indent=2)}
        
        Strategy Allocations:
        {json.dumps(strategies, indent=2)}
        
        Backtest Results:
        - Total return: {total_return:.2f}%
        - Sharpe ratio: {sharpe_ratio:.2f}
        - Max drawdown: {max_drawdown:.2f}%
        - Volatility: {volatility:.2f}%
        
        Please provide:
        1. A brief assessment of the portfolio's performance
        2. Analysis of the asset allocation and diversification
        3. Risk-adjusted performance evaluation
        4. Potential improvements to the allocation
        5. Correlation analysis between assets/strategies (if applicable)
        
        Keep your response concise and actionable.
        """
        
        return prompt
