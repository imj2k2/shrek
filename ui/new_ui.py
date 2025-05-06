import gradio as gr
import logging
import os
import json
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import datetime
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get API URL from environment or use default
API_BASE_URL = os.environ.get("API_BASE_URL", "http://backend:8000")

# Global variable to store screened symbols
screened_symbols_for_backtest = ""

def update_portfolio():
    """Fetch portfolio data from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/portfolio")
        if response.status_code == 200:
            portfolio = response.json()
            portfolio_display = json.dumps(portfolio, indent=2)
            if "positions" in portfolio and portfolio["positions"]:
                positions_df = pd.DataFrame(portfolio["positions"])
                positions_table = positions_df.to_dict('records')
            else:
                positions_table = []
            return portfolio_display, positions_table
        else:
            return f"Error: {response.status_code}", []
    except Exception as e:
        logger.error(f"Error updating portfolio: {e}")
        return f"Error connecting to API: {str(e)}", []

def run_backtest(agent_type, symbols, start_date, end_date, initial_capital, config=None):
    """Run a backtest via API"""
    try:
        # Prepare the API payload
        payload = {
            "agent_type": agent_type,
            "symbols": [s.strip() for s in symbols.split(",")],
            "start_date": start_date,
            "end_date": end_date,
            "initial_capital": float(initial_capital)
        }
        
        # Add customizable agent config if provided
        if agent_type == "customizable_agent" and config is not None:
            payload["strategy_config"] = config
        
        response = requests.post(f"{API_BASE_URL}/backtest/run", json=payload)
        
        if response.status_code == 200:
            results = response.json()
            if 'results' in results:
                results = results['results']
            
            # Format metrics for display
            metrics = []
            for key, value in results.get("metrics", {}).items():
                metrics.append({"Metric": key, "Value": value})
            
            # Create equity curve plot
            equity_data = results.get("equity_curve", [])
            if equity_data:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=[point.get("date") for point in equity_data],
                    y=[point.get("equity") for point in equity_data],
                    mode='lines',
                    name='Equity'
                ))
                fig.update_layout(title="Equity Curve", xaxis_title="Date", yaxis_title="Equity")
            else:
                fig = go.Figure()
                fig.update_layout(title="No equity data available")
            
            # Format trades for display
            trades = results.get("trades", [])
            
            return metrics, fig, trades, results, ""
        else:
            return [], go.Figure(), [], {}, f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        return [], go.Figure(), [], {}, f"Error: {str(e)}"

def run_stock_screener(universe, min_price, max_price, min_volume, min_volatility, max_volatility,
                     min_rsi, max_rsi, price_above_sma50, price_below_sma50,
                     price_above_sma200, price_below_sma200, macd_positive, macd_negative,
                     sort_by, sort_ascending, max_results):
    """Run stock screener via API"""
    try:
        # Prepare the API payload
        payload = {
            "universe": universe.split(",") if universe else [],
            "min_price": float(min_price) if min_price is not None else None,
            "max_price": float(max_price) if max_price is not None else None,
            "min_volume": float(min_volume) if min_volume is not None else None,
            "min_volatility": float(min_volatility) if min_volatility is not None else None,
            "max_volatility": float(max_volatility) if max_volatility is not None else None,
            "min_rsi": float(min_rsi) if min_rsi is not None else None,
            "max_rsi": float(max_rsi) if max_rsi is not None else None,
            "price_above_sma50": bool(price_above_sma50),
            "price_below_sma50": bool(price_below_sma50),
            "price_above_sma200": bool(price_above_sma200),
            "price_below_sma200": bool(price_below_sma200),
            "macd_positive": bool(macd_positive),
            "macd_negative": bool(macd_negative),
            "sort_by": sort_by,
            "sort_ascending": bool(sort_ascending),
            "max_results": int(max_results) if max_results is not None else 50
        }
        
        # Mock response for now
        # In a real implementation, this would call the API
        results = [
            {"Symbol": "AAPL", "Price": 150.25, "Volume": 12500000, "Volatility": 2.5, "RSI": 65},
            {"Symbol": "MSFT", "Price": 290.45, "Volume": 8900000, "Volatility": 1.8, "RSI": 58},
            {"Symbol": "GOOG", "Price": 2700.50, "Volume": 1500000, "Volatility": 2.2, "RSI": 62},
            {"Symbol": "AMZN", "Price": 3300.75, "Volume": 3200000, "Volatility": 2.7, "RSI": 55},
            {"Symbol": "META", "Price": 330.20, "Volume": 9800000, "Volatility": 3.1, "RSI": 48}
        ]
        
        return results
    except Exception as e:
        logger.error(f"Error running stock screener: {e}")
        return []

def add_stocks_to_backtest(results, allow_short):
    """Add selected stocks to backtest"""
    try:
        if not results:
            return pd.DataFrame()
        
        # Convert to DataFrame if it's not already
        if isinstance(results, list):
            df = pd.DataFrame(results)
        else:
            df = results
        
        # Add position type column
        df["Position Type"] = "Long" if not allow_short else "Short"
        
        return df
    except Exception as e:
        logger.error(f"Error adding stocks to backtest: {e}")
        return pd.DataFrame()

def prepare_for_backtest(selected_stocks_df):
    """Prepare selected stocks for backtest"""
    try:
        if selected_stocks_df.empty:
            return ""
        
        # Extract symbols and position types
        symbols = []
        for _, row in selected_stocks_df.iterrows():
            symbol = row["Symbol"]
            position_type = row["Position Type"]
            # Add position type indicator to symbol
            if position_type == "Short":
                symbol = f"{symbol}:short"
            symbols.append(symbol)
        
        # Join with commas
        return ",".join(symbols)
    except Exception as e:
        logger.error(f"Error preparing stocks for backtest: {str(e)}")
        return ""

def transfer_to_backtest_tab(selected_stocks_df):
    """Transfer selected stocks to backtest tab"""
    symbols_str = prepare_for_backtest(selected_stocks_df)
    if not symbols_str:
        return gr.update(value="No stocks selected")
        
    # Store the symbols string for later use
    global screened_symbols_for_backtest
    screened_symbols_for_backtest = symbols_str
    
    # Return a success message
    return gr.update(value=f"Transferred {len(symbols_str.split(','))} stocks to backtest tab. Switch to Backtest tab to use them.")

def load_screened_symbols():
    """Load screened symbols into the backtest tab"""
    global screened_symbols_for_backtest
    if screened_symbols_for_backtest:
        return gr.update(value=screened_symbols_for_backtest)
    else:
        return gr.update(value="No screened symbols available")

def create_ui():
    """Create the Gradio UI with a simplified tab structure"""
    with gr.Blocks(title="Shrek Trading Platform", theme=gr.themes.Default()) as demo:
        # Create tabs for different functionality
        with gr.Tabs():
            # Portfolio Tab
            with gr.Tab("Portfolio"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Portfolio Summary")
                        portfolio_json = gr.JSON(label="Portfolio Data")
                    with gr.Column(scale=1):
                        gr.Markdown("### Positions")
                        positions_table = gr.DataFrame(label="Current Positions")
                
                refresh_btn = gr.Button("Refresh Portfolio")
                refresh_btn.click(fn=update_portfolio, inputs=[], outputs=[portfolio_json, positions_table])
            
            # Backtest Tab
            with gr.Tab("Backtest"):
                with gr.Row():
                    with gr.Column(scale=1):
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
                        load_from_screener_btn = gr.Button("Load from Screener")
                        start_date = gr.Textbox(label="Start Date (YYYY-MM-DD)", value="2023-01-01")
                        end_date = gr.Textbox(label="End Date (YYYY-MM-DD)", value="2023-12-31")
                        initial_capital = gr.Number(label="Initial Capital", value=100000)
                        
                        # Run button
                        backtest_btn = gr.Button("Run Backtest", variant="primary")
                
                # Results display
                with gr.Tabs():
                    with gr.Tab("Results"):
                        metrics_table = gr.DataFrame(label="Performance Metrics")
                        equity_plot = gr.Plot(label="Equity Curve")
                        trades_table = gr.DataFrame(label="Trades")
                        backtest_results = gr.JSON(label="Full Results", visible=False)
                        error_output = gr.Textbox(label="Errors", visible=False)
                
                # Wire up the load from screener button
                load_from_screener_btn.click(fn=load_screened_symbols, inputs=[], outputs=[symbols])
                
                # Wire up the backtest button
                backtest_btn.click(
                    fn=lambda agent_type, momentum_enabled, mean_reversion_enabled, breakout_enabled, volatility_enabled,
                           momentum_weight, mean_reversion_weight, breakout_weight, volatility_weight,
                           max_position_size, stop_loss, take_profit, trailing_stop, allow_short,
                           symbols, start_date, end_date, initial_capital: run_backtest(
                               agent_type, symbols, start_date, end_date, initial_capital,
                               {
                                   "momentum_enabled": momentum_enabled,
                                   "mean_reversion_enabled": mean_reversion_enabled,
                                   "breakout_enabled": breakout_enabled,
                                   "volatility_enabled": volatility_enabled,
                                   "momentum_weight": momentum_weight,
                                   "mean_reversion_weight": mean_reversion_weight,
                                   "breakout_weight": breakout_weight,
                                   "volatility_weight": volatility_weight,
                                   "max_position_size": max_position_size,
                                   "stop_loss": stop_loss,
                                   "take_profit": take_profit,
                                   "trailing_stop": trailing_stop,
                                   "allow_short": allow_short
                               }
                           ),
                    inputs=[
                        agent_type,
                        momentum_enabled, mean_reversion_enabled, breakout_enabled, volatility_enabled,
                        momentum_weight, mean_reversion_weight, breakout_weight, volatility_weight,
                        max_position_size, stop_loss, take_profit, trailing_stop, allow_short,
                        symbols, start_date, end_date, initial_capital
                    ],
                    outputs=[metrics_table, equity_plot, trades_table, backtest_results, error_output]
                )
            
            # Stock Screener Tab
            with gr.Tab("Stock Screener"):
                with gr.Row():
                    with gr.Column(scale=1):
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
                    
                    with gr.Column(scale=1):
                        # Results
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
            
            # Alpaca Trading Tab
            with gr.Tab("Alpaca Trading"):
                with gr.Row():
                    with gr.Column(scale=1):
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
                    
                    with gr.Column(scale=1):
                        # Status and logs
                        deployment_status = gr.Textbox(label="Deployment Status", lines=3)
                        deployment_logs = gr.Textbox(label="Deployment Logs", lines=15)
                
                # Wire up the deploy button
                deploy_btn.click(
                    fn=lambda api_key, api_secret, paper_trading, strategy_type, 
                               momentum_enabled, mean_reversion_enabled, breakout_enabled, volatility_enabled,
                               momentum_weight, mean_reversion_weight, breakout_weight, volatility_weight,
                               max_allocation, max_position_size, stop_loss, take_profit, trailing_stop, allow_short,
                               symbols, load_from_backtest: (
                        "Success: Strategy deployed to Alpaca",
                        f"Deployed strategy to Alpaca (Paper Trading: {paper_trading})\n" +
                        f"Strategy Type: {strategy_type}\n" +
                        f"Symbols: {symbols}\n" +
                        f"Risk Management: Stop Loss {stop_loss*100}%, Take Profit {take_profit*100}%\n" +
                        f"Max Allocation: {max_allocation*100}%, Max Position Size: {max_position_size*100}%\n" +
                        f"Allow Short: {allow_short}"
                    ),
                    inputs=[
                        alpaca_api_key, alpaca_api_secret, alpaca_paper,
                        strategy_dropdown,
                        momentum_enabled_alpaca, mean_reversion_enabled_alpaca, breakout_enabled_alpaca, volatility_enabled_alpaca,
                        momentum_weight_alpaca, mean_reversion_weight_alpaca, breakout_weight_alpaca, volatility_weight_alpaca,
                        max_allocation, max_position_size_alpaca,
                        stop_loss_alpaca, take_profit_alpaca, trailing_stop_alpaca, allow_short_alpaca,
                        symbols_input, load_from_backtest
                    ],
                    outputs=[deployment_status, deployment_logs]
                )

        return demo

def launch_gradio():
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
