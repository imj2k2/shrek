import gradio as gr
import pandas as pd
import logging
import os
import sys

# Add parent directory to path to import from data module
sys.path.append('/app')

# Import screener functions
from ui.screener_functions import (
    run_stock_screener, 
    add_stocks_to_backtest, 
    transfer_to_backtest_tab, 
    prepare_for_backtest, 
    load_screened_symbols, 
    deploy_to_alpaca
)

# Import backtest functions
from ui.backtest_functions import run_backtest, run_mock_backtest

# Import database functions
from ui.database_functions import refresh_database, get_database_status

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to store screened symbols
screened_symbols_for_backtest = ""

def create_ui():
    """Create the Gradio UI"""
    with gr.Blocks(title="Shrek Trading Platform") as demo:
        gr.Markdown("# Shrek Trading Platform")
        
        with gr.Tabs(selected=0) as tabs:
            # Stock Screener Tab
            with gr.Tab("Stock Screener"):
                with gr.Row():
                    with gr.Column(scale=2):
                        # Screener inputs
                        universe_input = gr.Textbox(label="Universe (comma-separated symbols)", value="AAPL,MSFT,GOOG,AMZN,META")
                        
                        # Price criteria
                        with gr.Row():
                            min_price = gr.Number(label="Min Price ($)", value=10)
                            max_price = gr.Number(label="Max Price ($)", value=1000)
                        
                        # Volume criteria
                        min_volume = gr.Number(label="Min Daily Volume", value=100000)
                        
                        # Volatility criteria
                        with gr.Row():
                            min_volatility = gr.Number(label="Min Volatility (%)", value=0)
                            max_volatility = gr.Number(label="Max Volatility (%)", value=100)
                        
                        # RSI criteria
                        with gr.Row():
                            min_rsi = gr.Number(label="Min RSI", value=0, minimum=0, maximum=100)
                            max_rsi = gr.Number(label="Max RSI", value=100, minimum=0, maximum=100)
                        
                        # Moving average criteria
                        price_above_sma50 = gr.Checkbox(label="Price Above 50-day MA", value=False)
                        price_below_sma50 = gr.Checkbox(label="Price Below 50-day MA", value=False)
                        price_above_sma200 = gr.Checkbox(label="Price Above 200-day MA", value=False)
                        price_below_sma200 = gr.Checkbox(label="Price Below 200-day MA", value=False)
                        
                        # MACD criteria
                        macd_positive = gr.Checkbox(label="MACD Positive", value=False)
                        macd_negative = gr.Checkbox(label="MACD Negative", value=False)
                        
                        # Sorting and limits
                        with gr.Row():
                            sort_by = gr.Dropdown(
                                label="Sort By", 
                                choices=["Symbol", "Price", "Volume", "Volatility", "RSI", "Daily_Change_%"], 
                                value="Volume"
                            )
                            sort_ascending = gr.Checkbox(label="Sort Ascending", value=False)
                        
                        max_results = gr.Slider(minimum=5, maximum=100, value=20, step=5, label="Max Results")
                        
                        # Allow short positions
                        allow_short_positions = gr.Checkbox(label="Allow Short Positions", value=False)
                        
                        # Run button
                        run_screener_btn = gr.Button("Run Stock Screener", variant="primary")
                    
                    with gr.Column(scale=3):
                        # Results section
                        screener_results = gr.DataFrame(label="Screener Results")
                        
                        # Add to backtest selection
                        add_to_backtest_btn = gr.Button("Add Selected to Backtest")
                        selected_stocks = gr.DataFrame(label="Selected Stocks for Backtesting")
                        
                        # Transfer to backtest tab
                        transfer_to_backtest_btn = gr.Button("Transfer to Backtest Tab", variant="secondary")
                        transfer_status = gr.Textbox(label="Transfer Status")
                
                # Wire up the screener button
                run_screener_btn.click(
                    fn=run_stock_screener,
                    inputs=[
                        universe_input, min_price, max_price, min_volume,
                        min_volatility, max_volatility, min_rsi, max_rsi,
                        price_above_sma50, price_below_sma50,
                        price_above_sma200, price_below_sma200,
                        macd_positive, macd_negative,
                        sort_by, sort_ascending, max_results
                    ],
                    outputs=[screener_results]
                )
                
                # Wire up the add to backtest button
                add_to_backtest_btn.click(
                    fn=add_stocks_to_backtest,
                    inputs=[screener_results, allow_short_positions],
                    outputs=[selected_stocks]
                )
                
                # Wire up the transfer button
                transfer_to_backtest_btn.click(
                    fn=transfer_to_backtest_tab,
                    inputs=[selected_stocks],
                    outputs=[transfer_status]
                )
            
            # Backtest Tab
            with gr.Tab("Backtest"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # Backtest inputs
                        backtest_symbols = gr.Textbox(label="Symbols (comma-separated)", value="AAPL,MSFT,GOOG", info="e.g. AAPL,MSFT,GOOG")
                        load_screened_btn = gr.Button("Load from Screener")
                        load_status = gr.Textbox(label="Load Status")
                        
                        # Date range
                        with gr.Row():
                            start_date = gr.Textbox(label="Start Date", value="2023-01-01", info="YYYY-MM-DD")
                            end_date = gr.Textbox(label="End Date", value="2023-12-31", info="YYYY-MM-DD")
                        
                        # Initial capital
                        initial_capital = gr.Number(label="Initial Capital ($)", value=10000)
                        
                        # Strategy type
                        strategy_type = gr.Dropdown(
                            label="Strategy Type", 
                            choices=["MeanReversion", "Momentum", "Breakout", "MultiStrategy"], 
                            value="MultiStrategy"
                        )
                        
                        # Run button
                        run_backtest_btn = gr.Button("Run Backtest", variant="primary")
                    
                    with gr.Column(scale=2):
                        # Results section
                        backtest_results = gr.Textbox(label="Backtest Results", lines=10)
                        backtest_chart = gr.Plot(label="Equity Curve")
                
                # Wire up the load screened symbols button
                load_screened_btn.click(
                    fn=load_screened_symbols,
                    inputs=[],
                    outputs=[load_status, backtest_symbols]
                )
                
                # Wire up the run backtest button
                run_backtest_btn.click(
                    fn=run_backtest,  # Use real backtest function
                    inputs=[backtest_symbols, start_date, end_date, initial_capital, strategy_type],
                    outputs=[backtest_results, backtest_chart]
                )
            
            # Database Management Tab
            with gr.Tab("Database Management"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # Database refresh options
                        gr.Markdown("### Refresh Market Data")
                        data_source = gr.Dropdown(
                            label="Data Source", 
                            choices=["polygon_s3", "polygon_api", "yahoo"], 
                            value="polygon_s3",
                            info="polygon_s3: Polygon.io S3 (fastest), polygon_api: Polygon.io API, yahoo: Yahoo Finance"
                        )
                        symbols_input = gr.Textbox(
                            label="Symbols (comma-separated, leave empty for default universe)", 
                            value="", 
                            info="e.g. AAPL,MSFT,GOOG"
                        )
                        days_input = gr.Slider(
                            minimum=1, 
                            maximum=30, 
                            value=7, 
                            step=1, 
                            label="Days of Historical Data"
                        )
                        reset_db = gr.Checkbox(
                            label="Reset Database", 
                            value=False,
                            info="WARNING: This will delete all existing data"
                        )
                        refresh_btn = gr.Button("Refresh Database", variant="primary")
                    
                    with gr.Column(scale=2):
                        # Status and logs
                        refresh_status = gr.Textbox(label="Refresh Status", lines=3)
                        db_status = gr.TextArea(label="Database Status", lines=15, interactive=False)
                        refresh_logs = gr.JSON(label="Refresh Details")
                
                # Wire up the refresh button
                refresh_btn.click(
                    fn=refresh_database,
                    inputs=[data_source, symbols_input, days_input, reset_db],
                    outputs=[refresh_status, refresh_logs]
                )
                
                # Add a button to get database status
                status_btn = gr.Button("Get Database Status")
                status_btn.click(
                    fn=get_database_status,
                    inputs=[],
                    outputs=[db_status]
                )
            
            # Deploy Tab
            with gr.Tab("Deploy to Alpaca"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # API credentials
                        api_key = gr.Textbox(label="Alpaca API Key", type="password")
                        api_secret = gr.Textbox(label="Alpaca API Secret", type="password")
                        paper_trading = gr.Checkbox(label="Paper Trading", value=True)
                        
                        # Strategy type
                        strategy_type = gr.Dropdown(
                            label="Strategy Type", 
                            choices=["MeanReversion", "Momentum", "Breakout", "MultiStrategy"], 
                            value="MultiStrategy"
                        )
                        
                        # Strategy parameters
                        gr.Markdown("### Strategy Parameters")
                        with gr.Row():
                            momentum_enabled = gr.Checkbox(label="Momentum", value=True)
                            mean_reversion_enabled = gr.Checkbox(label="Mean Reversion", value=True)
                            breakout_enabled = gr.Checkbox(label="Breakout", value=True)
                            volatility_enabled = gr.Checkbox(label="Volatility", value=True)
                        
                        with gr.Row():
                            momentum_weight = gr.Slider(minimum=0, maximum=1, value=0.25, step=0.05, label="Momentum Weight")
                            mean_reversion_weight = gr.Slider(minimum=0, maximum=1, value=0.25, step=0.05, label="Mean Reversion Weight")
                            breakout_weight = gr.Slider(minimum=0, maximum=1, value=0.25, step=0.05, label="Breakout Weight")
                            volatility_weight = gr.Slider(minimum=0, maximum=1, value=0.25, step=0.05, label="Volatility Weight")
                        
                        # Risk management
                        gr.Markdown("### Risk Management")
                        with gr.Row():
                            max_allocation = gr.Slider(minimum=0.1, maximum=1, value=0.9, step=0.05, label="Max Allocation")
                            max_position_size = gr.Slider(minimum=0.01, maximum=0.5, value=0.1, step=0.01, label="Max Position Size")
                        
                        with gr.Row():
                            stop_loss = gr.Slider(minimum=0.01, maximum=0.2, value=0.05, step=0.01, label="Stop Loss")
                            take_profit = gr.Slider(minimum=0.01, maximum=0.5, value=0.1, step=0.01, label="Take Profit")
                            trailing_stop = gr.Slider(minimum=0, maximum=0.2, value=0.02, step=0.01, label="Trailing Stop")
                        
                        allow_short = gr.Checkbox(label="Allow Short Positions", value=False)
                        
                        # Symbols
                        symbols = gr.Textbox(label="Symbols (comma-separated)", value="AAPL,MSFT,GOOG", info="e.g. AAPL,MSFT,GOOG")
                        load_from_backtest = gr.Checkbox(label="Load Symbols from Backtest", value=False)
                        
                        # Deploy button
                        deploy_btn = gr.Button("Deploy to Alpaca", variant="primary")
                    
                    with gr.Column(scale=1):
                        # Status and logs
                        deployment_status = gr.Textbox(label="Deployment Status", lines=3)
                        deployment_logs = gr.Textbox(label="Deployment Logs", lines=15)
                
                # Wire up the deploy button
                deploy_btn.click(
                    fn=deploy_to_alpaca,
                    inputs=[
                        api_key, api_secret, paper_trading, strategy_type,
                        momentum_enabled, mean_reversion_enabled, breakout_enabled, volatility_enabled,
                        momentum_weight, mean_reversion_weight, breakout_weight, volatility_weight,
                        max_allocation, max_position_size, stop_loss, take_profit, trailing_stop, allow_short,
                        symbols, load_from_backtest
                    ],
                    outputs=[deployment_status, deployment_logs]
                )
    
    return demo

def launch_gradio():
    """Launch the Gradio UI"""
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    launch_gradio()
