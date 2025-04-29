import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
import os

from agents.stocks_agent import StocksAgent
from agents.options_agent import OptionsAgent
from agents.crypto_agent import CryptoAgent
from risk.advanced_risk_manager import AdvancedRiskManager
from data.data_fetcher import DataFetcher
from data.storage import Storage

class BacktestResult:
    """Container for backtest results"""
    
    def __init__(self, 
                 strategy_name: str,
                 start_date: datetime,
                 end_date: datetime,
                 initial_capital: float,
                 trades: List[Dict[str, Any]],
                 equity_curve: List[Dict[str, Any]],
                 metrics: Dict[str, Any]):
        self.strategy_name = strategy_name
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.trades = trades
        self.equity_curve = equity_curve
        self.metrics = metrics
    
    def plot_equity_curve(self, save_path: Optional[str] = None):
        """Plot equity curve with drawdowns"""
        df = pd.DataFrame(self.equity_curve)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot equity curve
        ax1.plot(df.index, df['equity'], label='Equity', color='blue')
        ax1.set_title(f'Equity Curve - {self.strategy_name}')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.grid(True)
        ax1.legend()
        
        # Plot drawdowns
        ax2.fill_between(df.index, 0, df['drawdown'] * 100, color='red', alpha=0.3)
        ax2.set_title('Drawdown (%)')
        ax2.set_ylabel('Drawdown %')
        ax2.set_ylim(bottom=0, top=max(df['drawdown'] * 100) * 1.5)
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        return fig, (ax1, ax2)
    
    def plot_monthly_returns(self, save_path: Optional[str] = None):
        """Plot monthly returns heatmap"""
        df = pd.DataFrame(self.equity_curve)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Calculate daily returns
        df['daily_return'] = df['equity'].pct_change()
        
        # Group by year and month
        monthly_returns = df['daily_return'].groupby([df.index.year, df.index.month]).apply(
            lambda x: (1 + x).prod() - 1
        )
        
        # Reshape to have years as rows and months as columns
        monthly_returns = monthly_returns.unstack()
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        cmap = plt.cm.RdYlGn  # Red for negative, green for positive
        
        im = ax.imshow(monthly_returns, cmap=cmap)
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Monthly Return (%)', rotation=-90, va="bottom")
        
        # Set ticks and labels
        ax.set_xticks(np.arange(monthly_returns.shape[1]))
        ax.set_yticks(np.arange(monthly_returns.shape[0]))
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax.set_yticklabels(monthly_returns.index)
        
        # Rotate the tick labels and set alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        for i in range(monthly_returns.shape[0]):
            for j in range(monthly_returns.shape[1]):
                if not np.isnan(monthly_returns.iloc[i, j]):
                    ax.text(j, i, f"{monthly_returns.iloc[i, j]:.2%}",
                           ha="center", va="center", color="black")
        
        ax.set_title(f"Monthly Returns - {self.strategy_name}")
        fig.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        return fig, ax
    
    def plot_trade_distribution(self, save_path: Optional[str] = None):
        """Plot trade profit/loss distribution"""
        if not self.trades:
            return None, None
        
        trade_profits = [trade.get('profit', 0) for trade in self.trades]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(trade_profits, bins=20, alpha=0.7, color='blue')
        ax.axvline(0, color='r', linestyle='--')
        ax.set_title(f'Trade Profit/Loss Distribution - {self.strategy_name}')
        ax.set_xlabel('Profit/Loss ($)')
        ax.set_ylabel('Frequency')
        
        if save_path:
            plt.savefig(save_path)
            
        return fig, ax
    
    def save_results(self, directory: str):
        """Save backtest results to directory"""
        # Create directory if it doesn't exist
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        with open(os.path.join(directory, 'metrics.json'), 'w') as f:
            json.dump(self.metrics, f, indent=4)
        
        # Save trades
        pd.DataFrame(self.trades).to_csv(os.path.join(directory, 'trades.csv'), index=False)
        
        # Save equity curve
        pd.DataFrame(self.equity_curve).to_csv(os.path.join(directory, 'equity_curve.csv'), index=False)
        
        # Save plots
        self.plot_equity_curve(save_path=os.path.join(directory, 'equity_curve.png'))
        self.plot_monthly_returns(save_path=os.path.join(directory, 'monthly_returns.png'))
        self.plot_trade_distribution(save_path=os.path.join(directory, 'trade_distribution.png'))
        
        # Save summary
        with open(os.path.join(directory, 'summary.txt'), 'w') as f:
            f.write(f"Backtest Results - {self.strategy_name}\n")
            f.write(f"Period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}\n")
            f.write(f"Initial Capital: ${self.initial_capital:,.2f}\n\n")
            
            f.write("Performance Metrics:\n")
            for key, value in self.metrics.items():
                if isinstance(value, float):
                    f.write(f"  {key}: {value:.4f}\n")
                else:
                    f.write(f"  {key}: {value}\n")

class Backtester:
    """Backtesting engine for trading strategies"""
    
    def __init__(self, 
                 initial_capital: float = 100000.0,
                 commission: float = 0.001,  # 0.1% commission
                 slippage: float = 0.001,    # 0.1% slippage
                 data_directory: str = 'backtest_data'):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.data_directory = data_directory
        self.logger = logging.getLogger("Backtester")
        
        # Create data directory if it doesn't exist
        Path(data_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize data fetcher and storage
        self.data_fetcher = DataFetcher()
        self.storage = Storage()
    
    def run_backtest(self,
                    agent: Union[StocksAgent, OptionsAgent, CryptoAgent],
                    symbols: List[str],
                    start_date: Union[str, datetime],
                    end_date: Union[str, datetime],
                    timeframe: str = 'day',
                    risk_manager: Optional[AdvancedRiskManager] = None,
                    strategy_name: Optional[str] = None) -> BacktestResult:
        """
        Run backtest for a given agent and symbols
        
        Args:
            agent: Trading agent (stocks, options, crypto)
            symbols: List of symbols to backtest
            start_date: Start date for backtest
            end_date: End date for backtest
            timeframe: Data timeframe ('day', 'hour', '15min', etc.)
            risk_manager: Optional risk manager
            strategy_name: Name of the strategy
            
        Returns:
            BacktestResult object with backtest results
        """
        # Convert dates to datetime if they are strings
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Set strategy name if not provided
        if not strategy_name:
            strategy_name = f"{agent.__class__.__name__}_{timeframe}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        
        self.logger.info(f"Starting backtest for {strategy_name} from {start_date} to {end_date}")
        
        # Fetch historical data for all symbols
        data = {}
        for symbol in symbols:
            self.logger.info(f"Fetching data for {symbol}")
            symbol_data = self._fetch_historical_data(symbol, start_date, end_date, timeframe)
            if symbol_data is not None:
                data[symbol] = symbol_data
        
        if not data:
            self.logger.error("No data available for backtest")
            return None
        
        # Initialize portfolio
        portfolio = {
            'cash': self.initial_capital,
            'positions': {},
            'equity': self.initial_capital,
            'peak_equity': self.initial_capital,
            'drawdown': 0.0
        }
        
        # Store data for later reference
        self.data = data
        
        # Initialize tracking variables
        trades = []
        equity_curve = []
        dates = sorted(data[symbols[0]].index)
        
        # Run backtest for each date
        for date in dates:
            # Skip dates before start_date
            if date.to_pydatetime() < start_date:
                continue
                
            # Skip dates after end_date
            if date.to_pydatetime() > end_date:
                break
            
            # Update portfolio with latest prices
            self._update_portfolio_prices(portfolio, data, date)
            
            # Check if trading is halted by risk manager
            trading_halted = False
            if risk_manager:
                risk_assessment = risk_manager.assess(portfolio)
                trading_halted = risk_assessment.get('trading_halted', False)
            
            if not trading_halted:
                # Generate signals for each symbol
                for symbol in symbols:
                    # Skip if we don't have data for this symbol on this date
                    if date not in data[symbol].index:
                        continue
                    
                    # Prepare data for agent
                    symbol_data = self._prepare_data_for_agent(data[symbol], date, agent)
                    
                    # Generate signal
                    signal = agent.generate_signals(symbol_data)
                    
                    # Execute signal if it's a buy or sell
                    if signal.get('action') in ['buy', 'sell']:
                        trade = self._execute_trade(portfolio, signal, data[symbol].loc[date], date)
                        if trade:
                            trades.append(trade)
            
            # Record equity curve
            equity = self._calculate_equity(portfolio, data, date)
            
            # Update peak equity and drawdown
            if equity > portfolio['peak_equity']:
                portfolio['peak_equity'] = equity
            
            drawdown = 0.0
            if portfolio['peak_equity'] > 0:
                drawdown = (portfolio['peak_equity'] - equity) / portfolio['peak_equity']
            
            equity_curve.append({
                'date': date.strftime('%Y-%m-%d'),
                'equity': equity,
                'cash': portfolio['cash'],
                'holdings': equity - portfolio['cash'],
                'drawdown': drawdown
            })
        
        # Calculate performance metrics
        metrics = self._calculate_metrics(equity_curve, trades)
        
        # Create and return backtest result
        result = BacktestResult(
            strategy_name=strategy_name,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.initial_capital,
            trades=trades,
            equity_curve=equity_curve,
            metrics=metrics
        )
        
        self.logger.info(f"Backtest completed for {strategy_name}")
        self.logger.info(f"Final equity: ${equity_curve[-1]['equity']:,.2f}")
        self.logger.info(f"Total return: {metrics['total_return']:.2%}")
        self.logger.info(f"Sharpe ratio: {metrics['sharpe_ratio']:.4f}")
        
        return result
    
    def _fetch_historical_data(self, symbol: str, start_date: datetime, end_date: datetime, timeframe: str) -> pd.DataFrame:
        """Fetch historical data for a symbol"""
        # Check if we have cached data
        cache_file = os.path.join(self.data_directory, f"{symbol}_{timeframe}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv")
        
        if os.path.exists(cache_file):
            self.logger.info(f"Loading cached data for {symbol} from {cache_file}")
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            return df
        
        # Fetch data using DataFetcher
        try:
            # Adjust dates to ensure we have enough data for indicators
            adjusted_start = start_date - timedelta(days=100)  # Add buffer for indicators
            
            # For stocks
            if timeframe == 'day':
                # Explicitly try Polygon first if API key is available
                if self.data_fetcher.polygon_key:
                    self.logger.info(f"Using Polygon API for {symbol} with key: {self.data_fetcher.polygon_key[:5]}...")
                    data = self.data_fetcher.fetch_stock_data(symbol, start_date=adjusted_start, end_date=end_date, source="polygon")
                else:
                    data = self.data_fetcher.fetch_stock_data(symbol, start_date=adjusted_start, end_date=end_date)
            # For crypto
            elif 'BTC' in symbol or 'ETH' in symbol:
                data = self.data_fetcher.fetch_crypto_data(symbol, start_date=adjusted_start, end_date=end_date, interval=timeframe)
            else:
                self.logger.error(f"Unsupported symbol type or timeframe: {symbol}, {timeframe}")
                return None
            
            # Convert to DataFrame if it's not already
            if not isinstance(data, pd.DataFrame):
                if isinstance(data, dict) and 'results' in data:
                    df = pd.DataFrame(data['results'])
                else:
                    self.logger.error(f"Unexpected data format for {symbol}")
                    return None
            else:
                df = data
            
            # Check if DataFrame is empty
            if df.empty:
                self.logger.error(f"Empty DataFrame for {symbol}")
                return None
                
            # Store data source information
            if hasattr(df, 'data_source'):
                data_source = df.data_source
            else:
                # Try to determine the data source
                if 'Open' in df.columns and 'High' in df.columns:  # Yahoo or mock
                    if len(df) > 0 and df.index[0].year >= 2023:  # Recent data is likely mock
                        data_source = 'mock'
                    else:
                        data_source = 'yahoo'
                elif 'o' in df.columns and 'h' in df.columns:  # Polygon
                    data_source = 'polygon'
                else:
                    data_source = 'unknown'
                
            # Add data source as attribute
            df.data_source = data_source
            self.logger.info(f"Data source for {symbol}: {data_source}")
            
            # Check for required columns with flexibility for different naming conventions
            has_required = False
            
            # Check for Yahoo/Mock style columns (capitalized)
            if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
                has_required = True
            # Check for Polygon style columns (lowercase)
            elif all(col in df.columns for col in ['o', 'h', 'l', 'c']):
                # Rename to standard format
                df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume'}, inplace=True)
                has_required = True
            # Check for generic lowercase columns
            elif all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                # Rename to standard format
                df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}, inplace=True)
                has_required = True
            
            if not has_required:
                self.logger.warning(f"Missing some required columns in data for {symbol}, but will try to adapt")
            
            # Set index to datetime if it's not already
            if not isinstance(df.index, pd.DatetimeIndex):
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                elif 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                else:
                    self.logger.error(f"No date column found in data for {symbol}")
                    return None
            
            # Save to cache
            df.to_csv(cache_file)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def _prepare_data_for_agent(self, df: pd.DataFrame, date: pd.Timestamp, agent) -> Dict[str, Any]:
        """Prepare data for agent based on agent type"""
        # Get historical data up to current date
        hist_data = df[df.index <= date].copy()
        
        # Map column names - handle both lowercase and capitalized column names
        column_mapping = {
            'open': ['open', 'Open'],
            'high': ['high', 'High'],
            'low': ['low', 'Low'],
            'close': ['close', 'Close'],
            'volume': ['volume', 'Volume']
        }
        
        # Create standardized data dictionary
        data = {'symbol': df.name if hasattr(df, 'name') else 'UNKNOWN',
                'date': date.strftime('%Y-%m-%d')}
        
        # Check if we have any data
        if hist_data.empty:
            self.logger.warning(f"No historical data available for date {date}")
            return None
        
        # Add price data with column name flexibility
        for target_col, possible_cols in column_mapping.items():
            found = False
            for col in possible_cols:
                if col in hist_data.columns:
                    data[target_col] = hist_data[col].values
                    found = True
                    break
            if not found:
                if target_col == 'volume':  # Volume can be optional
                    data[target_col] = [1000000] * len(hist_data)  # Default volume
                else:
                    self.logger.warning(f"Missing required column {target_col} in data, using default values")
                    # Use a default value to prevent crashes
                    if target_col == 'open':
                        data[target_col] = hist_data.iloc[:, 0].values  # Use first column
                    elif target_col == 'high':
                        data[target_col] = hist_data.iloc[:, 0].values * 1.01  # 1% higher
                    elif target_col == 'low':
                        data[target_col] = hist_data.iloc[:, 0].values * 0.99  # 1% lower
                    elif target_col == 'close':
                        data[target_col] = hist_data.iloc[:, 0].values  # Use first column
                    else:
                        data[target_col] = [0] * len(hist_data)
        
        # Specific data for options agent
        if isinstance(agent, OptionsAgent):
            # For options, we need to add option-specific data
            # In a real scenario, this would come from options data
            # Here we'll use some mock values
            current_price = hist_data['close'].iloc[-1]
            data.update({
                'S': current_price,  # Underlying price
                'K': current_price * 1.05,  # Strike price (5% OTM)
                'T': 30/365,  # 30 days to expiration
                'r': 0.02,  # 2% risk-free rate
                'sigma': 0.3,  # 30% volatility
                'option_type': 'call',
                'option_price': current_price * 0.05  # Rough estimate
            })
        
        return data
    
    def _update_portfolio_prices(self, portfolio: Dict[str, Any], data: Dict[str, pd.DataFrame], date: pd.Timestamp):
        """Update portfolio with latest prices"""
        for symbol, position in portfolio['positions'].items():
            if symbol in data and date in data[symbol].index:
                current_price = data[symbol].loc[date, 'close']
                position['current_price'] = current_price
                position['market_value'] = position['quantity'] * current_price
                position['profit_loss'] = position['market_value'] - position['cost_basis']
    
    def _execute_trade(self, portfolio: Dict[str, Any], signal: Dict[str, Any], price_data: pd.Series, date: pd.Timestamp) -> Dict[str, Any]:
        """Execute a trade based on a signal"""
        action = signal.get('action')
        symbol = signal.get('symbol', 'UNKNOWN')
        quantity = signal.get('qty', 0)
        
        if quantity <= 0:
            return None
        
        # Get current price with slippage
        current_price = price_data['close']
        if action == 'buy':
            adjusted_price = current_price * (1 + self.slippage)  # Higher price for buys
        else:
            adjusted_price = current_price * (1 - self.slippage)  # Lower price for sells
        
        # Calculate commission
        commission_amount = adjusted_price * quantity * self.commission
        
        # Execute buy
        if action == 'buy':
            # Check if we have enough cash
            total_cost = (adjusted_price * quantity) + commission_amount
            if total_cost > portfolio['cash']:
                # Adjust quantity if we don't have enough cash
                max_quantity = int((portfolio['cash'] - commission_amount) / adjusted_price)
                if max_quantity <= 0:
                    return None
                quantity = max_quantity
                total_cost = (adjusted_price * quantity) + commission_amount
            
            # Update portfolio
            if symbol not in portfolio['positions']:
                portfolio['positions'][symbol] = {
                    'quantity': quantity,
                    'avg_price': adjusted_price,
                    'cost_basis': adjusted_price * quantity,
                    'current_price': adjusted_price,
                    'market_value': adjusted_price * quantity,
                    'profit_loss': 0
                }
            else:
                # Update existing position
                position = portfolio['positions'][symbol]
                new_quantity = position['quantity'] + quantity
                new_cost_basis = position['cost_basis'] + (adjusted_price * quantity)
                position['quantity'] = new_quantity
                position['avg_price'] = new_cost_basis / new_quantity
                position['cost_basis'] = new_cost_basis
                position['current_price'] = adjusted_price
                position['market_value'] = position['current_price'] * new_quantity
                position['profit_loss'] = position['market_value'] - position['cost_basis']
            
            # Deduct cash
            portfolio['cash'] -= total_cost
            
            # Create trade record
            trade = {
                'date': date.strftime('%Y-%m-%d'),
                'symbol': symbol,
                'action': 'buy',
                'quantity': quantity,
                'price': adjusted_price,
                'commission': commission_amount,
                'total_cost': total_cost,
                'strategy': signal.get('strategy', 'unknown')
            }
            
            return trade
        
        # Execute sell
        elif action == 'sell':
            # Check if we have the position
            if symbol not in portfolio['positions'] or portfolio['positions'][symbol]['quantity'] <= 0:
                return None
            
            # Adjust quantity if we don't have enough shares
            position = portfolio['positions'][symbol]
            if quantity > position['quantity']:
                quantity = position['quantity']
            
            # Calculate proceeds
            proceeds = (adjusted_price * quantity) - commission_amount
            
            # Calculate profit/loss
            avg_price = position['avg_price']
            profit_loss = (adjusted_price - avg_price) * quantity
            
            # Update portfolio
            position['quantity'] -= quantity
            if position['quantity'] <= 0:
                # Remove position if no shares left
                del portfolio['positions'][symbol]
            else:
                # Update position metrics
                position['market_value'] = position['current_price'] * position['quantity']
                position['cost_basis'] = position['avg_price'] * position['quantity']
                position['profit_loss'] = position['market_value'] - position['cost_basis']
            
            # Add cash
            portfolio['cash'] += proceeds
            
            # Create trade record
            trade = {
                'date': date.strftime('%Y-%m-%d'),
                'symbol': symbol,
                'action': 'sell',
                'quantity': quantity,
                'price': adjusted_price,
                'commission': commission_amount,
                'total_proceeds': proceeds,
                'profit_loss': profit_loss,
                'strategy': signal.get('strategy', 'unknown')
            }
            
            return trade
        
        return None
    
    def _calculate_equity(self, portfolio: Dict[str, Any], data: Dict[str, pd.DataFrame], date: pd.Timestamp) -> float:
        """Calculate total portfolio equity"""
        equity = portfolio['cash']
        
        for symbol, position in portfolio['positions'].items():
            equity += position['market_value']
        
        return equity
    
    def _calculate_metrics(self, equity_curve: List[Dict[str, Any]], trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate performance metrics"""
        if not equity_curve:
            return {}
        
        # Convert equity curve to DataFrame
        df = pd.DataFrame(equity_curve)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Calculate returns
        df['daily_return'] = df['equity'].pct_change()
        
        # Calculate metrics
        initial_equity = df['equity'].iloc[0]
        final_equity = df['equity'].iloc[-1]
        total_return = (final_equity / initial_equity) - 1
        
        # Annualized return
        days = (df.index[-1] - df.index[0]).days
        if days > 0:
            annual_return = (1 + total_return) ** (365 / days) - 1
        else:
            annual_return = 0
        
        # Volatility and Sharpe ratio
        if len(df) > 1:
            daily_volatility = df['daily_return'].std()
            annual_volatility = daily_volatility * (252 ** 0.5)  # Assuming 252 trading days
            
            # Sharpe ratio (assuming risk-free rate of 0)
            if annual_volatility > 0:
                sharpe_ratio = annual_return / annual_volatility
            else:
                sharpe_ratio = 0
        else:
            daily_volatility = 0
            annual_volatility = 0
            sharpe_ratio = 0
        
        # Maximum drawdown
        max_drawdown = df['drawdown'].max()
        
        # Win rate
        if trades:
            winning_trades = sum(1 for trade in trades if trade.get('action') == 'sell' and trade.get('profit_loss', 0) > 0)
            total_sell_trades = sum(1 for trade in trades if trade.get('action') == 'sell')
            win_rate = winning_trades / total_sell_trades if total_sell_trades > 0 else 0
        else:
            win_rate = 0
        
        # Profit factor
        gross_profit = sum(trade.get('profit_loss', 0) for trade in trades if trade.get('action') == 'sell' and trade.get('profit_loss', 0) > 0)
        gross_loss = abs(sum(trade.get('profit_loss', 0) for trade in trades if trade.get('action') == 'sell' and trade.get('profit_loss', 0) <= 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'daily_volatility': daily_volatility,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(trades),
            'buy_trades': sum(1 for trade in trades if trade.get('action') == 'buy'),
            'sell_trades': sum(1 for trade in trades if trade.get('action') == 'sell')
        }

def run_example_backtest():
    """Run an example backtest"""
    # Initialize backtester
    backtester = Backtester(initial_capital=100000.0)
    
    # Initialize agents
    stocks_agent = StocksAgent()
    
    # Initialize risk manager
    risk_manager = AdvancedRiskManager(max_drawdown=0.2, trailing_stop_pct=0.05)
    
    # Run backtest
    result = backtester.run_backtest(
        agent=stocks_agent,
        symbols=['AAPL', 'MSFT', 'GOOGL'],
        start_date='2022-01-01',
        end_date='2022-12-31',
        timeframe='day',
        risk_manager=risk_manager,
        strategy_name='StockMomentumStrategy'
    )
    
    if result:
        # Save results
        result.save_results('backtest_results/StockMomentumStrategy')
        
        # Display key metrics
        print(f"Total Return: {result.metrics['total_return']:.2%}")
        print(f"Sharpe Ratio: {result.metrics['sharpe_ratio']:.4f}")
        print(f"Max Drawdown: {result.metrics['max_drawdown']:.2%}")
        print(f"Win Rate: {result.metrics['win_rate']:.2%}")
        
        # Plot equity curve
        result.plot_equity_curve()
        plt.show()

if __name__ == "__main__":
    run_example_backtest()
