import gradio as gr
import logging
import os
import json
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get API URL from environment or use default
API_BASE_URL = os.environ.get("API_BASE_URL", "http://backend:8000")

# Global variable to store screened symbols
screened_symbols_for_backtest = ""

def create_ui():
    """Create the Gradio UI with a simplified tab structure"""
    with gr.Blocks(title="Shrek Trading Platform", theme=gr.themes.Default()) as demo:
        # Create tabs for different functionality
        with gr.Tabs():
            # Portfolio Tab
            with gr.Tab("Portfolio"):
                gr.Markdown("### Portfolio Summary")
                portfolio_json = gr.JSON(label="Portfolio Data")
                gr.Markdown("### Positions")
                positions_table = gr.DataFrame(label="Current Positions")
                refresh_btn = gr.Button("Refresh Portfolio")
            
            # Backtest Tab
            with gr.Tab("Backtest"):
                gr.Markdown("### Backtest Configuration")
                agent_type = gr.Dropdown(
                    label="Agent Type",
                    choices=["stocks_agent", "options_agent", "crypto_agent", "customizable_agent"],
                    value="customizable_agent"
                )
                
                # Strategy toggles
                with gr.Accordion("Strategy Selection", open=True):
                    momentum_enabled = gr.Checkbox(label="Enable Momentum", value=True)
                    mean_reversion_enabled = gr.Checkbox(label="Enable Mean Reversion", value=True)
                    breakout_enabled = gr.Checkbox(label="Enable Breakout", value=True)
                    volatility_enabled = gr.Checkbox(label="Enable Volatility", value=True)
                
                # Strategy weights
                with gr.Accordion("Strategy Weights", open=True):
                    momentum_weight = gr.Slider(minimum=0, maximum=2, value=1.0, step=0.1, label="Momentum Weight")
                    mean_reversion_weight = gr.Slider(minimum=0, maximum=2, value=1.0, step=0.1, label="Mean Reversion Weight")
                    breakout_weight = gr.Slider(minimum=0, maximum=2, value=1.0, step=0.1, label="Breakout Weight")
                    volatility_weight = gr.Slider(minimum=0, maximum=2, value=0.8, step=0.1, label="Volatility Weight")
                
                # Risk management
                with gr.Accordion("Risk Management", open=True):
                    max_position_size = gr.Slider(minimum=0.01, maximum=0.5, value=0.1, step=0.01, label="Max Position Size")
                    stop_loss = gr.Slider(minimum=0.01, maximum=0.2, value=0.05, step=0.01, label="Stop Loss (%)")
                    take_profit = gr.Slider(minimum=0.01, maximum=0.5, value=0.15, step=0.01, label="Take Profit (%)")
                    trailing_stop = gr.Checkbox(label="Use Trailing Stop", value=True)
                    allow_short = gr.Checkbox(label="Allow Short Positions", value=False)
                
                # Symbols and dates
                symbols = gr.Textbox(label="Symbols (comma-separated)", value="AAPL,MSFT,GOOG")
                load_from_screener = gr.Button("Load from Screener")
                start_date = gr.Textbox(label="Start Date (YYYY-MM-DD)", value="2023-01-01")
                end_date = gr.Textbox(label="End Date (YYYY-MM-DD)", value="2023-12-31")
                initial_capital = gr.Number(label="Initial Capital", value=100000)
                use_mock_data = gr.Checkbox(label="Use Mock Data", value=False)
                
                # Run button
                backtest_btn = gr.Button("Run Backtest", variant="primary")
                
                # Results display
                with gr.Tabs():
                    with gr.Tab("Results"):
                        metrics_table = gr.DataFrame(label="Performance Metrics")
                        equity_plot = gr.Plot(label="Equity Curve")
                        trades_table = gr.DataFrame(label="Trades")
                
                # Error output
                error_output = gr.Textbox(label="Errors", visible=False)
            
            # Stock Screener Tab
            with gr.Tab("Stock Screener"):
                gr.Markdown("### Stock Screening Criteria")
                
                # Universe selection
                universe_input = gr.Textbox(
                    label="Stock Universe", 
                    placeholder="Leave blank for S&P 500 or enter comma-separated symbols",
                    info="Leave blank to screen S&P 500 stocks or enter specific symbols"
                )
                
                # Price criteria
                with gr.Accordion("Price Criteria", open=True):
                    min_price = gr.Number(label="Minimum Price ($)", value=5)
                    max_price = gr.Number(label="Maximum Price ($)", value=1000)
                    min_volume = gr.Number(label="Minimum Volume", value=100000)
                
                # Technical indicators
                with gr.Accordion("Technical Indicators", open=True):
                    min_volatility = gr.Number(label="Minimum Volatility (%)", value=1)
                    max_volatility = gr.Number(label="Maximum Volatility (%)", value=5)
                    min_rsi = gr.Number(label="Minimum RSI", value=30)
                    max_rsi = gr.Number(label="Maximum RSI", value=70)
                    
                    price_above_sma50 = gr.Checkbox(label="Price Above 50-day SMA", value=False)
                    price_below_sma50 = gr.Checkbox(label="Price Below 50-day SMA", value=False)
                    price_above_sma200 = gr.Checkbox(label="Price Above 200-day SMA", value=False)
                    price_below_sma200 = gr.Checkbox(label="Price Below 200-day SMA", value=False)
                    
                    macd_positive = gr.Checkbox(label="MACD Positive", value=False)
                    macd_negative = gr.Checkbox(label="MACD Negative", value=False)
                
                # Sorting options
                with gr.Accordion("Sorting Options", open=True):
                    sort_by = gr.Dropdown(
                        label="Sort By",
                        choices=["Symbol", "Price", "Volume", "Volatility", "RSI"],
                        value="Volume"
                    )
                    sort_ascending = gr.Checkbox(label="Sort Ascending", value=False)
                    max_results = gr.Number(label="Maximum Results", value=50)
                
                # Position type
                allow_short_positions = gr.Checkbox(label="Allow Short Positions", value=False)
                
                # Run button
                run_screener_btn = gr.Button("Run Stock Screener", variant="primary")
                
                # Results
                screener_results = gr.DataFrame(label="Screener Results")
                
                # Add to backtest selection
                add_to_backtest_btn = gr.Button("Add Selected to Backtest")
                selected_stocks = gr.DataFrame(label="Selected Stocks for Backtesting")
                
                # Transfer to backtest tab
                transfer_to_backtest_btn = gr.Button("Transfer to Backtest Tab", variant="secondary")
                transfer_status = gr.Textbox(label="Transfer Status")
            
            # Saved Strategies Tab
            with gr.Tab("Saved Strategies"):
                gr.Markdown("### Saved Trading Strategies")
                refresh_strategies_btn = gr.Button("Refresh Strategies List")
                strategies_table = gr.DataFrame(
                    headers=["ID", "Name", "Agent Type", "Symbols", "Created", "Total Return", "Sharpe", "Win Rate"],
                    label="Saved Strategies"
                )
                
                # Strategy details
                with gr.Row():
                    with gr.Column(scale=1):
                        selected_strategy_id = gr.Textbox(label="Selected Strategy ID")
                        view_strategy_btn = gr.Button("View Strategy Details")
                    with gr.Column(scale=1):
                        broker_dropdown = gr.Dropdown(
                            label="Broker", 
                            choices=["Alpaca", "Interactive Brokers", "TD Ameritrade"],
                            value="Alpaca"
                        )
                        deploy_strategy_btn = gr.Button("Deploy Strategy")
                
                strategy_details = gr.JSON(label="Strategy Details")
                deployment_result = gr.Textbox(label="Deployment Result")
            
            # Portfolio Backtest Tab
            with gr.Tab("Portfolio Backtest"):
                gr.Markdown("### Portfolio Configuration")
                portfolio_allocations = gr.Textbox(
                    label="Asset Allocations (symbol: weight)", 
                    value="AAPL: 0.4\nMSFT: 0.3\nGOOG: 0.3",
                    lines=10
                )
                portfolio_strategies = gr.Textbox(
                    label="Symbol Strategies (symbol: strategy_name)", 
                    value="AAPL: customizable\nMSFT: momentum\nGOOG: mean_reversion",
                    lines=10
                )
                portfolio_start = gr.Textbox(label="Start Date (YYYY-MM-DD)", value="2023-01-01")
                portfolio_end = gr.Textbox(label="End Date (YYYY-MM-DD)", value="2023-12-31")
                generate_insights = gr.Checkbox(label="Generate LLM Insights", value=False)
                run_portfolio_btn = gr.Button("Run Portfolio Backtest")
                
                # Results
                portfolio_metrics = gr.DataFrame(
                    headers=["Metric", "Value"],
                    label="Portfolio Performance Metrics"
                )
                portfolio_plot = gr.Plot(label="Portfolio Equity Curve")
                portfolio_results = gr.JSON(label="Portfolio Results")
                portfolio_insights = gr.Textbox(label="LLM Insights", lines=20)
            
            # Alpaca Trading Tab
            with gr.Tab("Alpaca Trading"):
                gr.Markdown("### Alpaca Paper Trading Configuration")
                
                # API Keys
                with gr.Accordion("API Keys", open=True):
                    alpaca_api_key = gr.Textbox(label="Alpaca API Key", type="password")
                    alpaca_api_secret = gr.Textbox(label="Alpaca API Secret", type="password")
                    alpaca_paper = gr.Checkbox(label="Use Paper Trading", value=True)
                
                # Strategy Selection
                with gr.Accordion("Strategy Selection", open=True):
                    # Dropdown for saved strategies
                    strategy_dropdown = gr.Dropdown(
                        label="Select Strategy", 
                        choices=["Custom Strategy", "Mean Reversion", "Momentum", "Breakout"],
                        value="Custom Strategy"
                    )
                    
                    # Custom strategy parameters
                    with gr.Group(visible=True) as custom_strategy_params:
                        gr.Markdown("### Custom Strategy Parameters")
                        # Strategy toggles
                        momentum_enabled_alpaca = gr.Checkbox(label="Enable Momentum", value=True)
                        mean_reversion_enabled_alpaca = gr.Checkbox(label="Enable Mean Reversion", value=True)
                        breakout_enabled_alpaca = gr.Checkbox(label="Enable Breakout", value=True)
                        volatility_enabled_alpaca = gr.Checkbox(label="Enable Volatility", value=True)
                        
                        # Strategy weights
                        momentum_weight_alpaca = gr.Slider(minimum=0, maximum=2, value=1.0, step=0.1, label="Momentum Weight")
                        mean_reversion_weight_alpaca = gr.Slider(minimum=0, maximum=2, value=1.0, step=0.1, label="Mean Reversion Weight")
                        breakout_weight_alpaca = gr.Slider(minimum=0, maximum=2, value=1.0, step=0.1, label="Breakout Weight")
                        volatility_weight_alpaca = gr.Slider(minimum=0, maximum=2, value=0.8, step=0.1, label="Volatility Weight")
                
                # Portfolio Allocation
                with gr.Accordion("Portfolio Allocation", open=True):
                    max_allocation = gr.Slider(minimum=0.01, maximum=1.0, value=0.5, step=0.01, 
                                             label="Maximum Portfolio Allocation", 
                                             info="Maximum percentage of portfolio to allocate to this strategy")
                    max_position_size_alpaca = gr.Slider(minimum=0.01, maximum=0.5, value=0.1, step=0.01, 
                                                label="Maximum Position Size", 
                                                info="Maximum percentage of portfolio per position")
                
                # Risk Management
                with gr.Accordion("Risk Management", open=True):
                    stop_loss_alpaca = gr.Slider(minimum=0.01, maximum=0.2, value=0.05, step=0.01, 
                                        label="Stop Loss (%)", 
                                        info="Percentage below entry price to exit position")
                    take_profit_alpaca = gr.Slider(minimum=0.01, maximum=0.5, value=0.15, step=0.01, 
                                          label="Take Profit (%)", 
                                          info="Percentage above entry price to exit position")
                    trailing_stop_alpaca = gr.Checkbox(label="Use Trailing Stop", value=True)
                    allow_short_alpaca = gr.Checkbox(label="Allow Short Positions", value=False)
                
                # Symbols to trade
                symbols_input = gr.Textbox(
                    label="Symbols to Trade (comma-separated)", 
                    value="AAPL,MSFT,GOOG,AMZN",
                    info="Enter the symbols you want to trade"
                )
                
                # Load from backtest results
                load_from_backtest = gr.Checkbox(label="Load from Backtest Results", value=False)
                
                # Deploy button
                deploy_btn = gr.Button("Deploy to Alpaca", variant="primary")
                
                # Status and logs
                deployment_status = gr.Textbox(label="Deployment Status", lines=3)
                deployment_logs = gr.Textbox(label="Deployment Logs", lines=15)

        return demo

def launch_gradio():
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

if __name__ == "__main__":
    launch_gradio()
