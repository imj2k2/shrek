import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.backtest_engine import Backtester, BacktestResult
from agents.stocks_agent import StocksAgent
from agents.options_agent import OptionsAgent
from agents.crypto_agent import CryptoAgent
from risk.advanced_risk_manager import AdvancedRiskManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('backtest.log')
    ]
)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run backtest for trading strategies')
    
    # Agent type
    parser.add_argument('--agent', type=str, required=True, choices=['stocks', 'options', 'crypto'],
                        help='Type of agent to use for backtesting')
    
    # Symbols
    parser.add_argument('--symbols', type=str, required=True, 
                        help='Comma-separated list of symbols to backtest')
    
    # Date range
    parser.add_argument('--start-date', type=str, required=True,
                        help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, required=True,
                        help='End date for backtest (YYYY-MM-DD)')
    
    # Timeframe
    parser.add_argument('--timeframe', type=str, default='day',
                        choices=['day', 'hour', '15min', '5min', '1min'],
                        help='Data timeframe')
    
    # Initial capital
    parser.add_argument('--capital', type=float, default=100000.0,
                        help='Initial capital for backtest')
    
    # Risk parameters
    parser.add_argument('--max-drawdown', type=float, default=0.2,
                        help='Maximum drawdown allowed (0.2 = 20%)')
    parser.add_argument('--trailing-stop', type=float, default=0.05,
                        help='Trailing stop percentage (0.05 = 5%)')
    parser.add_argument('--max-position-size', type=float, default=0.2,
                        help='Maximum position size as percentage of portfolio (0.2 = 20%)')
    
    # Output directory
    parser.add_argument('--output-dir', type=str, default='backtest_results',
                        help='Directory to save backtest results')
    
    # Strategy name
    parser.add_argument('--strategy-name', type=str, default=None,
                        help='Name of the strategy (default: auto-generated)')
    
    return parser.parse_args()

def main():
    """Main function to run backtest"""
    args = parse_args()
    
    # Parse symbols
    symbols = [s.strip() for s in args.symbols.split(',')]
    
    # Initialize backtester
    backtester = Backtester(initial_capital=args.capital)
    
    # Initialize agent based on type
    if args.agent == 'stocks':
        agent = StocksAgent()
    elif args.agent == 'options':
        agent = OptionsAgent()
    elif args.agent == 'crypto':
        agent = CryptoAgent()
    else:
        raise ValueError(f"Unsupported agent type: {args.agent}")
    
    # Initialize risk manager
    risk_manager = AdvancedRiskManager(
        max_drawdown=args.max_drawdown,
        trailing_stop_pct=args.trailing_stop,
        max_position_size=args.max_position_size
    )
    
    # Generate strategy name if not provided
    strategy_name = args.strategy_name
    if not strategy_name:
        strategy_name = f"{args.agent.capitalize()}Strategy_{args.timeframe}_{datetime.now().strftime('%Y%m%d')}"
    
    # Run backtest
    logging.info(f"Starting backtest for {strategy_name} with symbols: {symbols}")
    
    result = backtester.run_backtest(
        agent=agent,
        symbols=symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        timeframe=args.timeframe,
        risk_manager=risk_manager,
        strategy_name=strategy_name
    )
    
    if result:
        # Create output directory
        output_dir = os.path.join(args.output_dir, strategy_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results
        result.save_results(output_dir)
        
        # Display key metrics
        print("\n" + "="*50)
        print(f"Backtest Results - {strategy_name}")
        print("="*50)
        print(f"Period: {args.start_date} to {args.end_date}")
        print(f"Symbols: {', '.join(symbols)}")
        print(f"Initial Capital: ${args.capital:,.2f}")
        print(f"Final Equity: ${result.equity_curve[-1]['equity']:,.2f}")
        print("-"*50)
        print(f"Total Return: {result.metrics['total_return']:.2%}")
        print(f"Annual Return: {result.metrics['annual_return']:.2%}")
        print(f"Sharpe Ratio: {result.metrics['sharpe_ratio']:.4f}")
        print(f"Max Drawdown: {result.metrics['max_drawdown']:.2%}")
        print(f"Win Rate: {result.metrics['win_rate']:.2%}")
        print(f"Profit Factor: {result.metrics['profit_factor']:.4f}")
        print(f"Total Trades: {result.metrics['total_trades']}")
        print("-"*50)
        print(f"Results saved to: {output_dir}")
        
        # Show plots
        plt.figure(figsize=(12, 8))
        fig, (ax1, ax2) = result.plot_equity_curve()
        plt.tight_layout()
        plt.show()
    else:
        logging.error("Backtest failed to run")

if __name__ == "__main__":
    main()
