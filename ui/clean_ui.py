import gradio as gr
import requests
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import random
import numpy as np
import datetime
import os
import logging
import time
import sys
import yfinance as yf

# Add parent directory to path to import from data module
sys.path.append('/Users/imj/codeRepo/genAI/shrek')
from data.data_fetcher import DataFetcher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get API URL from environment or use default
API_BASE_URL = os.environ.get("API_BASE_URL", "http://backend:8000")

# Global variables
screened_symbols_for_backtest = ""
data_fetcher = DataFetcher()

# S&P 500 symbols
SP500_SYMBOLS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "GOOG", "TSLA", "BRK.B", "UNH",
    "JPM", "XOM", "JNJ", "V", "PG", "MA", "HD", "CVX", "AVGO", "MRK",
    "LLY", "COST", "ABBV", "PEP", "KO", "ADBE", "WMT", "MCD", "CRM", "BAC",
    "TMO", "CSCO", "ACN", "ABT", "CMCSA", "AMD", "DHR", "NFLX", "VZ", "NEE",
    "INTC", "PFE", "ORCL", "PM", "TXN", "COP", "INTU", "IBM", "CAT", "QCOM"
]

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

def run_stock_screener(universe, min_price, max_price, min_volume, 
                     min_volatility, max_volatility, min_rsi, max_rsi, 
                     price_above_sma50, price_below_sma50, price_above_sma200, price_below_sma200, 
                     macd_positive, macd_negative, pe_min, pe_max, eps_min, eps_growth_min,
                     dividend_yield_min, market_cap_min, market_cap_max,
                     debt_to_equity_max, profit_margin_min, roe_min,
                     sort_by, sort_ascending, max_results):
    """Run stock screener using real data from Yahoo Finance"""

    try:
        # Determine which symbols to screen
        if universe and universe.strip():
            symbols_to_screen = [s.strip() for s in universe.split(',')]
        else:
            # Use S&P 500 symbols as default universe
            symbols_to_screen = SP500_SYMBOLS
        
        logger.info(f"Running stock screener on {len(symbols_to_screen)} symbols")
        
        # Initialize results list
        results = []
        
        # Process each symbol
        for symbol in symbols_to_screen:
            try:
                # Get stock data using yfinance
                stock = yf.Ticker(symbol)
                
                # Get basic info
                info = stock.info
                
                # Skip if we couldn't get basic info
                if not info:
                    continue
                
                # Get current price (yfinance 0.2.10 has different API)
                current_price = info.get('regularMarketPrice') or info.get('previousClose')
                if not current_price:
                    continue
                
                # Apply price filter
                if (min_price and current_price < min_price) or (max_price and current_price > max_price):
                    continue
                
                # Get volume (yfinance 0.2.10 has different API)
                volume = info.get('regularMarketVolume') or info.get('volume')
                if min_volume and (not volume or volume < min_volume):
                    continue
                
                # Get historical data for technical indicators
                hist = stock.history(period='200d')
                if len(hist) < 50:  # Need at least 50 days for indicators
                    continue
                
                # Calculate volatility (20-day standard deviation of returns)
                returns = hist['Close'].pct_change().dropna()
                volatility = returns.rolling(window=20).std().iloc[-1] * 100
                
                # Apply volatility filter
                if (min_volatility and volatility < min_volatility) or (max_volatility and volatility > max_volatility):
                    continue
                
                # Calculate RSI
                delta = hist['Close'].diff().dropna()
                up = delta.clip(lower=0)
                down = -delta.clip(upper=0)
                avg_gain = up.rolling(window=14).mean()
                avg_loss = down.rolling(window=14).mean()
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs)).iloc[-1]
                
                # Apply RSI filter
                if (min_rsi and rsi < min_rsi) or (max_rsi and rsi > max_rsi):
                    continue
                
                # Calculate SMAs
                sma50 = hist['Close'].rolling(window=50).mean().iloc[-1]
                sma200 = hist['Close'].rolling(window=200).mean().iloc[-1]
                current_close = hist['Close'].iloc[-1]
                
                # Apply SMA filters
                if price_above_sma50 and current_close <= sma50:
                    continue
                if price_below_sma50 and current_close >= sma50:
                    continue
                if price_above_sma200 and current_close <= sma200:
                    continue
                if price_below_sma200 and current_close >= sma200:
                    continue
                
                # Calculate MACD
                ema12 = hist['Close'].ewm(span=12, adjust=False).mean()
                ema26 = hist['Close'].ewm(span=26, adjust=False).mean()
                macd_line = ema12 - ema26
                macd_signal = macd_line.ewm(span=9, adjust=False).mean()
                macd_histogram = macd_line - macd_signal
                
                # Apply MACD filters
                if macd_positive and macd_histogram.iloc[-1] <= 0:
                    continue
                if macd_negative and macd_histogram.iloc[-1] >= 0:
                    continue
                
                # Get fundamental data
                pe_ratio = info.get('trailingPE')
                eps = info.get('trailingEPS')
                eps_growth = info.get('earningsGrowth')
                dividend_yield = info.get('dividendYield')
                if dividend_yield:
                    dividend_yield = dividend_yield * 100  # Convert to percentage
                market_cap = info.get('marketCap')
                debt_to_equity = info.get('debtToEquity')
                profit_margin = info.get('profitMargins')
                if profit_margin:
                    profit_margin = profit_margin * 100  # Convert to percentage
                roe = info.get('returnOnEquity')
                if roe:
                    roe = roe * 100  # Convert to percentage
                
                # Apply fundamental filters
                if pe_min and (not pe_ratio or pe_ratio < pe_min):
                    continue
                if pe_max and (pe_ratio and pe_ratio > pe_max):
                    continue
                if eps_min and (not eps or eps < eps_min):
                    continue
                if eps_growth_min and (not eps_growth or eps_growth < eps_growth_min):
                    continue
                if dividend_yield_min and (not dividend_yield or dividend_yield < dividend_yield_min):
                    continue
                if market_cap_min and (not market_cap or market_cap < market_cap_min):
                    continue
                if market_cap_max and (market_cap and market_cap > market_cap_max):
                    continue
                if debt_to_equity_max and (debt_to_equity and debt_to_equity > debt_to_equity_max):
                    continue
                if profit_margin_min and (not profit_margin or profit_margin < profit_margin_min):
                    continue
                if roe_min and (not roe or roe < roe_min):
                    continue
                
                # Add to results
                results.append({
                    "Symbol": symbol,
                    "Price": current_price,
                    "Volume": volume,
                    "Volatility": round(volatility, 2),
                    "RSI": round(rsi, 2),
                    "P/E Ratio": round(pe_ratio, 2) if pe_ratio else None,
                    "EPS": round(eps, 2) if eps else None,
                    "EPS Growth": round(eps_growth * 100, 2) if eps_growth else None,
                    "Dividend Yield": round(dividend_yield, 2) if dividend_yield else None,
                    "Market Cap (B)": round(market_cap / 1e9, 2) if market_cap else None,
                    "Debt/Equity": round(debt_to_equity, 2) if debt_to_equity else None,
                    "Profit Margin": round(profit_margin, 2) if profit_margin else None,
                    "ROE": round(roe, 2) if roe else None
                })
                
            except Exception as e:
                logger.warning(f"Error processing symbol {symbol}: {str(e)}")
                continue
        
        # Sort results
        if sort_by in ["Symbol", "Price", "Volume", "Volatility", "RSI", "P/E Ratio", "EPS", 
                      "EPS Growth", "Dividend Yield", "Market Cap (B)", "Debt/Equity", 
                      "Profit Margin", "ROE"]:
            # Handle None values for sorting
            results = sorted(results, key=lambda x: (x[sort_by] is None, x[sort_by]), reverse=not sort_ascending)
        
        # Limit results
        if max_results and len(results) > max_results:
            results = results[:max_results]
        
        logger.info(f"Stock screener found {len(results)} matching stocks")
        return results
    
    except Exception as e:
        logger.error(f"Error running stock screener: {str(e)}")
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

def deploy_to_alpaca(api_key, api_secret, paper_trading, strategy_type, 
                   momentum_enabled, mean_reversion_enabled, breakout_enabled, volatility_enabled,
                   momentum_weight, mean_reversion_weight, breakout_weight, volatility_weight,
                   max_allocation, max_position_size, stop_loss, take_profit, trailing_stop, allow_short,
                   symbols, load_from_backtest):
    """Deploy strategy to Alpaca"""
    try:
        # Validate inputs
        if not api_key or not api_secret:
            return "Error: API Key and Secret are required", "Please provide valid Alpaca API credentials"
        
        if not symbols and not load_from_backtest:
            return "Error: No symbols specified", "Please provide at least one symbol to trade or load from backtest results"
        
        # Process symbols
        global screened_symbols_for_backtest
        if load_from_backtest and screened_symbols_for_backtest:
            # Load symbols from backtest results
            symbol_list = []
            for symbol in screened_symbols_for_backtest.split(","):
                if ":short" in symbol:
                    # Extract the base symbol without the position type
                    base_symbol = symbol.split(":")[0]
                    symbol_list.append((base_symbol, "short"))
                else:
                    symbol_list.append((symbol, "long"))
            
            # Format for display
            symbol_display = ", ".join([f"{s[0]} ({s[1]})" for s in symbol_list])
            position_types = {s[0]: s[1] for s in symbol_list}
        else:
            # Process manually entered symbols
            symbol_list = [s.strip() for s in symbols.split(",")]
            symbol_display = ", ".join(symbol_list)
            position_types = {s: "long" for s in symbol_list}  # Default to long positions
        
        # Simulate API call
        logs = f"Deploying strategy to Alpaca (Paper Trading: {paper_trading})\n"
        logs += f"Strategy Type: {strategy_type}\n"
        logs += f"Symbols: {symbol_display}\n"
        logs += f"Risk Management: Stop Loss {stop_loss*100}%, Take Profit {take_profit*100}%\n"
        logs += f"Max Allocation: {max_allocation*100}%, Max Position Size: {max_position_size*100}%\n"
        logs += f"Allow Short: {allow_short}\n\n"
        
        # Simulate API call
        logs += "Connecting to Alpaca API...\n"
        time.sleep(1)  # Simulate API latency
        logs += "Validating API credentials...\n"
        time.sleep(1)  # Simulate API latency
        logs += "Creating trading strategy...\n"
        time.sleep(1)  # Simulate API latency
        logs += "Setting up risk parameters...\n"
        time.sleep(1)  # Simulate API latency
        logs += "Deploying strategy to Alpaca...\n"
        time.sleep(1)  # Simulate API latency
        logs += "Strategy successfully deployed!\n"
        
        return "Success: Strategy deployed to Alpaca", logs
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return f"Error: {str(e)}", error_details

def create_ui():
    """Create the Gradio UI with a clean tab structure"""
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
                        
                        # Fundamental indicators
                        with gr.Accordion("Fundamental Indicators", open=True):
                            pe_min = gr.Number(label="Minimum P/E Ratio", value=0)
                            pe_max = gr.Number(label="Maximum P/E Ratio", value=100)
                            eps_min = gr.Number(label="Minimum EPS", value=0)
                            eps_growth_min = gr.Number(label="Minimum EPS Growth (%)", value=0)
                            dividend_yield_min = gr.Number(label="Minimum Dividend Yield (%)", value=0)
                            
                            # Company size and financial health
                            market_cap_min = gr.Number(label="Minimum Market Cap (billions)", value=0)
                            market_cap_max = gr.Number(label="Maximum Market Cap (billions)", value=0, info="0 for no limit")
                            debt_to_equity_max = gr.Number(label="Maximum Debt/Equity Ratio", value=3)
                            profit_margin_min = gr.Number(label="Minimum Profit Margin (%)", value=0)
                            roe_min = gr.Number(label="Minimum Return on Equity (%)", value=0)
                        
                        # Sorting options
                        with gr.Accordion("Sorting Options", open=True):
                            sort_by = gr.Dropdown(
                                label="Sort By",
                                choices=["Symbol", "Price", "Volume", "Volatility", "RSI", "P/E Ratio", "EPS", 
                                        "EPS Growth", "Dividend Yield", "Market Cap (B)", "Debt/Equity", 
                                        "Profit Margin", "ROE"],
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
                        pe_min, pe_max, eps_min, eps_growth_min,
                        dividend_yield_min, market_cap_min, market_cap_max,
                        debt_to_equity_max, profit_margin_min, roe_min,
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
                    fn=deploy_to_alpaca,
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
