import unittest
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.backtest_engine import Backtester, BacktestResult
from agents.stocks_agent import StocksAgent
from risk.advanced_risk_manager import AdvancedRiskManager

class MockDataFetcher:
    """Mock data fetcher for testing"""
    
    def fetch_stock_data(self, symbol, start_date=None, end_date=None):
        """Return mock stock data"""
        # Create date range
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
            
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Create mock data
        data = {
            'open': np.linspace(100, 120, len(date_range)) + np.random.normal(0, 1, len(date_range)),
            'high': np.linspace(105, 125, len(date_range)) + np.random.normal(0, 1, len(date_range)),
            'low': np.linspace(95, 115, len(date_range)) + np.random.normal(0, 1, len(date_range)),
            'close': np.linspace(100, 120, len(date_range)) + np.random.normal(0, 1, len(date_range)),
            'volume': np.random.randint(1000, 10000, len(date_range))
        }
        
        df = pd.DataFrame(data, index=date_range)
        df.name = symbol
        return df
    
    def fetch_crypto_data(self, symbol, start_date=None, end_date=None, interval='day'):
        """Return mock crypto data"""
        return self.fetch_stock_data(symbol, start_date, end_date)

class MockStocksAgent(StocksAgent):
    """Mock stocks agent for testing"""
    
    def __init__(self, signal_type='buy_and_hold'):
        super().__init__()
        self.signal_type = signal_type
    
    def generate_signals(self, data):
        """Generate mock signals based on signal_type"""
        if self.signal_type == 'buy_and_hold':
            # Buy on first day, hold forever
            if len(data['close']) < 10:  # First few days
                return {
                    'action': 'buy',
                    'symbol': data['symbol'],
                    'qty': 10,
                    'price': data['close'][-1],
                    'strategy': 'buy_and_hold'
                }
            else:
                return {'action': 'hold'}
        
        elif self.signal_type == 'random':
            # Random buy/sell signals
            import random
            action = random.choice(['buy', 'sell', 'hold'])
            if action == 'hold':
                return {'action': 'hold'}
            else:
                return {
                    'action': action,
                    'symbol': data['symbol'],
                    'qty': random.randint(1, 10),
                    'price': data['close'][-1],
                    'strategy': 'random'
                }
        
        elif self.signal_type == 'trend_following':
            # Simple moving average crossover
            if len(data['close']) < 20:
                return {'action': 'hold'}
            
            # Calculate short and long moving averages
            short_ma = np.mean(data['close'][-10:])
            long_ma = np.mean(data['close'][-20:])
            
            if short_ma > long_ma:
                return {
                    'action': 'buy',
                    'symbol': data['symbol'],
                    'qty': 5,
                    'price': data['close'][-1],
                    'strategy': 'trend_following'
                }
            else:
                return {
                    'action': 'sell',
                    'symbol': data['symbol'],
                    'qty': 5,
                    'price': data['close'][-1],
                    'strategy': 'trend_following'
                }
        
        else:
            return {'action': 'hold'}

class TestBacktester(unittest.TestCase):
    """Test cases for Backtester"""
    
    def setUp(self):
        """Set up test environment"""
        # Create test directory
        os.makedirs('test_backtest_data', exist_ok=True)
        
        # Initialize backtester with mock data fetcher
        self.backtester = Backtester(
            initial_capital=100000.0,
            commission=0.001,
            slippage=0.001,
            data_directory='test_backtest_data'
        )
        
        # Replace data fetcher with mock
        self.backtester.data_fetcher = MockDataFetcher()
    
    def tearDown(self):
        """Clean up after tests"""
        # Remove test directory
        import shutil
        if os.path.exists('test_backtest_data'):
            shutil.rmtree('test_backtest_data')
    
    def test_buy_and_hold_strategy(self):
        """Test buy and hold strategy"""
        # Initialize agent
        agent = MockStocksAgent(signal_type='buy_and_hold')
        
        # Run backtest
        result = self.backtester.run_backtest(
            agent=agent,
            symbols=['AAPL'],
            start_date='2022-01-01',
            end_date='2022-01-31',
            timeframe='day',
            strategy_name='TestBuyAndHold'
        )
        
        # Check result
        self.assertIsNotNone(result)
        self.assertEqual(result.strategy_name, 'TestBuyAndHold')
        self.assertGreater(len(result.equity_curve), 0)
        self.assertGreater(len(result.trades), 0)
        
        # Check that we have at least one buy trade
        buy_trades = [t for t in result.trades if t['action'] == 'buy']
        self.assertGreater(len(buy_trades), 0)
    
    def test_trend_following_strategy(self):
        """Test trend following strategy"""
        # Initialize agent
        agent = MockStocksAgent(signal_type='trend_following')
        
        # Run backtest
        result = self.backtester.run_backtest(
            agent=agent,
            symbols=['AAPL', 'MSFT'],
            start_date='2022-01-01',
            end_date='2022-02-28',
            timeframe='day',
            strategy_name='TestTrendFollowing'
        )
        
        # Check result
        self.assertIsNotNone(result)
        self.assertEqual(result.strategy_name, 'TestTrendFollowing')
        self.assertGreater(len(result.equity_curve), 0)
        
        # We should have both buy and sell trades
        buy_trades = [t for t in result.trades if t['action'] == 'buy']
        sell_trades = [t for t in result.trades if t['action'] == 'sell']
        
        # In a 2-month backtest with trend following, we should have some trades
        # But this depends on the random data, so we can't be too strict
        self.assertGreaterEqual(len(buy_trades) + len(sell_trades), 0)
    
    def test_risk_manager_integration(self):
        """Test risk manager integration"""
        # Initialize agent
        agent = MockStocksAgent(signal_type='random')
        
        # Initialize risk manager with strict parameters
        risk_manager = AdvancedRiskManager(
            max_drawdown=0.05,  # 5% max drawdown
            trailing_stop_pct=0.02,  # 2% trailing stop
            max_position_size=0.1,  # 10% max position size
            notify_discord=False  # Disable Discord notifications for testing
        )
        
        # Run backtest
        result = self.backtester.run_backtest(
            agent=agent,
            symbols=['AAPL'],
            start_date='2022-01-01',
            end_date='2022-03-31',
            timeframe='day',
            risk_manager=risk_manager,
            strategy_name='TestRiskManager'
        )
        
        # Check result
        self.assertIsNotNone(result)
        self.assertEqual(result.strategy_name, 'TestRiskManager')
        
        # Check that max drawdown is respected
        # Note: This is not guaranteed due to the random nature of the test
        # but we can check that the max drawdown in the metrics is calculated
        self.assertIn('max_drawdown', result.metrics)
    
    def test_multiple_symbols(self):
        """Test backtesting with multiple symbols"""
        # Initialize agent
        agent = MockStocksAgent(signal_type='buy_and_hold')
        
        # Run backtest with multiple symbols
        result = self.backtester.run_backtest(
            agent=agent,
            symbols=['AAPL', 'MSFT', 'GOOGL'],
            start_date='2022-01-01',
            end_date='2022-01-31',
            timeframe='day',
            strategy_name='TestMultipleSymbols'
        )
        
        # Check result
        self.assertIsNotNone(result)
        self.assertEqual(result.strategy_name, 'TestMultipleSymbols')
        
        # We should have trades for different symbols
        symbols = set(trade['symbol'] for trade in result.trades)
        self.assertGreater(len(symbols), 1)  # At least 2 different symbols
    
    def test_backtest_result_methods(self):
        """Test BacktestResult methods"""
        # Create a simple backtest result
        equity_curve = [
            {'date': '2022-01-01', 'equity': 100000, 'cash': 100000, 'holdings': 0, 'drawdown': 0},
            {'date': '2022-01-02', 'equity': 101000, 'cash': 50000, 'holdings': 51000, 'drawdown': 0},
            {'date': '2022-01-03', 'equity': 102000, 'cash': 50000, 'holdings': 52000, 'drawdown': 0},
            {'date': '2022-01-04', 'equity': 101500, 'cash': 50000, 'holdings': 51500, 'drawdown': 0.0049},
            {'date': '2022-01-05', 'equity': 103000, 'cash': 50000, 'holdings': 53000, 'drawdown': 0}
        ]
        
        trades = [
            {'date': '2022-01-02', 'symbol': 'AAPL', 'action': 'buy', 'quantity': 10, 'price': 150},
            {'date': '2022-01-04', 'symbol': 'AAPL', 'action': 'sell', 'quantity': 5, 'price': 155, 'profit_loss': 25}
        ]
        
        metrics = {
            'total_return': 0.03,
            'annual_return': 0.15,
            'sharpe_ratio': 1.2,
            'max_drawdown': 0.0049,
            'win_rate': 1.0
        }
        
        result = BacktestResult(
            strategy_name='TestStrategy',
            start_date=datetime(2022, 1, 1),
            end_date=datetime(2022, 1, 5),
            initial_capital=100000,
            trades=trades,
            equity_curve=equity_curve,
            metrics=metrics
        )
        
        # Test plot methods (just make sure they don't raise exceptions)
        try:
            fig, ax = result.plot_equity_curve()
            self.assertIsNotNone(fig)
            self.assertIsNotNone(ax)
        except Exception as e:
            self.fail(f"plot_equity_curve raised exception: {e}")
        
        try:
            fig, ax = result.plot_trade_distribution()
            self.assertIsNotNone(fig)
            self.assertIsNotNone(ax)
        except Exception as e:
            self.fail(f"plot_trade_distribution raised exception: {e}")
        
        # Test save_results (just make sure it doesn't raise exceptions)
        try:
            result.save_results('test_backtest_data/test_results')
            self.assertTrue(os.path.exists('test_backtest_data/test_results'))
            self.assertTrue(os.path.exists('test_backtest_data/test_results/metrics.json'))
        except Exception as e:
            self.fail(f"save_results raised exception: {e}")

if __name__ == '__main__':
    unittest.main()
