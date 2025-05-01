import gradio as gr
import requests
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import datetime
import os
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get API URL from environment or use default
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

def update_portfolio():
    """Fetch portfolio data from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/portfolio")
        if response.status_code == 200:
            portfolio = response.json()
            
            # Create a formatted version for display
            portfolio_display = json.dumps(portfolio, indent=2)
            
            # Create a table of positions
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

def run_backtest(agent_type, symbols, start_date, end_date, initial_capital, config=None, 
                strategy_name=None, strategy_description=None, save_strategy=False):
    """Run a backtest via API"""
    try:
        # Prepare the API payload
        payload = {
            "agent_type": agent_type,
            "symbols": [s.strip() for s in symbols.split(",")],
            "start_date": start_date,
            "end_date": end_date,
            "initial_capital": float(initial_capital),
            "save_strategy": save_strategy
        }
        
        # Add strategy saving information if provided
        if save_strategy:
            if strategy_name:
                payload["strategy_name"] = strategy_name
            if strategy_description:
                payload["description"] = strategy_description
        
        # Add customizable agent config if provided
        if agent_type == "customizable_agent" and config is not None:
            payload["strategy_config"] = config
            logger.info(f"Using customizable agent with config: {config}")
        
        logger.info(f"Sending backtest request to {API_BASE_URL}/backtest/run with payload: {payload}")
        response = requests.post(f"{API_BASE_URL}/backtest/run", json=payload)
        
        if response.status_code == 200:
            try:
                response_data = response.json()
                logger.info(f"Received backtest response: {response_data}")
                results = response_data.get('results', {})
                
                # Get data source information
                data_sources = results.get("data_sources", {})
                
                # Add data source information to results for display
                if data_sources:
                    data_source_info = "\n\nData Sources:\n"
                    for symbol, source in data_sources.items():
                        data_source_info += f"  {symbol}: {source.upper()}\n"
                    
                    # Add to the beginning of the results JSON
                    results_with_sources = {"data_source_summary": data_source_info.strip()}
                    results_with_sources.update(results)
                else:
                    results_with_sources = results
                
                # Create metrics table
                metrics = results.get("metrics", {})
                
                # Add data source information to metrics table
                metrics_table = [
                    {"Metric": k, "Value": v}
                    for k, v in metrics.items()
                ]
                
                # Add data sources to metrics table
                for symbol, source in data_sources.items():
                    metrics_table.append({"Metric": f"Data Source ({symbol})", "Value": source.upper()})
                
                # Create plot
                plot = plot_backtest_results(results)
                
                # Return in order: metrics table, plot, results JSON
                return metrics_table, plot, json.dumps(results_with_sources, indent=2)
            except json.JSONDecodeError as je:
                logger.error(f"JSON decode error: {je}, Response content: {response.text}")
                return [], None, f"Error decoding response: {response.text}"
        else:
            error_msg = f"Error: {response.status_code}"
            try:
                error_detail = response.json().get('detail', '')
                logger.error(f"Backtest API error: {error_detail}")
                error_msg += f" - {error_detail}"
            except:
                pass
            return [], None, error_msg
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        return [], None, f"Error: {str(e)}"

def plot_backtest_results(results):
    """Create a plotly figure from backtest results with benchmark comparison"""
    try:
        # Extract equity curve and other data
        equity_curve = results.get("equity_curve", [])
        metrics = results.get("metrics", {})
        trades = results.get("trades", [])
        
        if not equity_curve:
            return go.Figure().update_layout(title="No equity curve data available")
        
        # Create DataFrame from equity curve
        equity_df = pd.DataFrame(equity_curve)
        equity_df['date'] = pd.to_datetime(equity_df['date'])
        
        # Check if we have benchmark comparison metrics
        has_benchmark = any(key in metrics for key in ['benchmark_total_return', 'alpha', 'beta'])
        
        # Create figure with subplots - add a third subplot if we have benchmark data
        if has_benchmark:
            fig = make_subplots(rows=3, cols=1, 
                             shared_xaxes=True,
                             vertical_spacing=0.05,
                             row_heights=[0.6, 0.2, 0.2],
                             subplot_titles=("Portfolio Value", "Drawdown", "Strategy vs Benchmark"))
        else:
            fig = make_subplots(rows=2, cols=1, 
                             shared_xaxes=True,
                             vertical_spacing=0.05,
                             row_heights=[0.7, 0.3],
                             subplot_titles=("Portfolio Value", "Drawdown"))
        
        # Add equity curve to first subplot
        fig.add_trace(
            go.Scatter(
                x=equity_df['date'],
                y=equity_df['equity'],
                mode='lines',
                name='Strategy',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Check if benchmark data is available in the results
        benchmark_data = results.get('benchmark_data', None)
        if benchmark_data is not None and isinstance(benchmark_data, dict) and 'close' in benchmark_data:
            # Get benchmark data dates and prices
            benchmark_dates = benchmark_data.get('date', [])
            benchmark_prices = benchmark_data.get('close', [])
            
            if benchmark_dates and benchmark_prices and len(benchmark_dates) == len(benchmark_prices):
                # Convert to pandas Series
                benchmark_df = pd.DataFrame({
                    'date': pd.to_datetime(benchmark_dates),
                    'close': benchmark_prices
                })
                benchmark_df.set_index('date', inplace=True)
                
                # Filter to match equity curve dates
                first_date = equity_df['date'].min()
                last_date = equity_df['date'].max()
                benchmark_df = benchmark_df[(benchmark_df.index >= first_date) & (benchmark_df.index <= last_date)]
                
                if not benchmark_df.empty:
                    # Normalize benchmark to start at the same value as portfolio
                    initial_capital = equity_df['equity'].iloc[0]
                    initial_price = benchmark_df['close'].iloc[0]
                    normalized_benchmark = benchmark_df['close'] / initial_price * initial_capital
                    
                    # Add benchmark to first subplot
                    fig.add_trace(
                        go.Scatter(
                            x=benchmark_df.index,
                            y=normalized_benchmark,
                            mode='lines',
                            name='SPY Benchmark',
                            line=dict(color='green', width=2, dash='dash')
                        ),
                        row=1, col=1
                    )
                    
                    # Calculate and add relative performance to third subplot
                    if has_benchmark:
                        # Calculate daily returns
                        equity_df.set_index('date', inplace=True)
                        strategy_returns = equity_df['equity'].pct_change().fillna(0)
                        benchmark_returns = benchmark_df['close'].pct_change().fillna(0)
                        
                        # Align the return series
                        common_dates = strategy_returns.index.intersection(benchmark_returns.index)
                        if len(common_dates) > 1:
                            # Calculate cumulative excess return
                            strategy_aligned = strategy_returns.loc[common_dates]
                            benchmark_aligned = benchmark_returns.loc[common_dates]
                            excess_returns = strategy_aligned - benchmark_aligned
                            cumulative_excess = (1 + excess_returns).cumprod() - 1
                            
                            # Add to third subplot
                            fig.add_trace(
                                go.Scatter(
                                    x=common_dates,
                                    y=cumulative_excess * 100,  # Convert to percentage
                                    mode='lines',
                                    name='Excess Return',
                                    line=dict(color='purple', width=2)
                                ),
                                row=3, col=1
                            )
                            
                            # Add zero line reference
                            fig.add_shape(
                                type="line",
                                x0=common_dates[0],
                                y0=0,
                                x1=common_dates[-1],
                                y1=0,
                                line=dict(color="gray", width=1, dash="dash"),
                                row=3, col=1
                            )
        
        # Add drawdown to second subplot
        fig.add_trace(
            go.Scatter(
                x=equity_df['date'] if 'date' in equity_df.columns else equity_df.index,
                y=equity_df['drawdown'] * 100,  # Convert to percentage
                fill='tozeroy',
                mode='lines',
                name='Drawdown',
                line=dict(color='red', width=2)
            ),
            row=2, col=1
        )
        
        # Add buy and sell trade markers
        for trade in trades:
            if trade.get('action') == 'buy':
                fig.add_trace(
                    go.Scatter(
                        x=[pd.to_datetime(trade.get('date'))],
                        y=[trade.get('price')],
                        mode='markers',
                        name='Buy',
                        marker=dict(symbol='triangle-up', size=10, color='green'),
                        showlegend=False
                    ),
                    row=1, col=1
                )
            elif trade.get('action') == 'sell':
                fig.add_trace(
                    go.Scatter(
                        x=[pd.to_datetime(trade.get('date'))],
                        y=[trade.get('price')],
                        mode='markers',
                        name='Sell',
                        marker=dict(symbol='triangle-down', size=10, color='red'),
                        showlegend=False
                    ),
                    row=1, col=1
                )
        
        # Update layout
        fig.update_layout(
            height=800 if has_benchmark else 600,
            title="Backtest Results",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        # Update axis labels
        fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        if has_benchmark:
            fig.update_yaxes(title_text="Excess Return vs SPY (%)", row=3, col=1)
        fig.update_xaxes(title_text="Date", row=2 if not has_benchmark else 3, col=1)
        
        return fig
        
    except Exception as e:
        logger.error(f"Error plotting backtest results: {e}")
        import traceback
        logger.error(traceback.format_exc())
        # Return an empty figure with error message
        fig = go.Figure()
        fig.add_annotation(text=f"Error plotting results: {str(e)}", 
                         xref="paper", yref="paper",
                         x=0.5, y=0.5, showarrow=False)
        return fig

def run_portfolio_backtest(allocations, symbol_strategies, start_date, end_date, generate_insights):
    """Run a portfolio backtest via API"""
    try:
        # Parse allocations from text input
        allocation_dict = {}
        for line in allocations.strip().split('\n'):
            if ':' in line:
                symbol, weight = line.split(':', 1)
                try:
                    allocation_dict[symbol.strip()] = float(weight.strip())
                except ValueError:
                    logger.error(f"Invalid allocation format: {line}")
        
        if not allocation_dict:
            return [], None, "Error: No valid allocations provided"
        
        # Parse symbol strategies from text input
        strategy_dict = {}
        if symbol_strategies:
            for line in symbol_strategies.strip().split('\n'):
                if ':' in line:
                    symbol, strategy = line.split(':', 1)
                    strategy_dict[symbol.strip()] = strategy.strip()
        
        # Prepare payload
        payload = {
            "allocations": allocation_dict,
            "symbol_strategies": strategy_dict,
            "start_date": start_date,
            "end_date": end_date,
            "generate_insights": generate_insights
        }
        
        logger.info(f"Sending portfolio backtest request to {API_BASE_URL}/portfolio/backtest with payload: {payload}")
        response = requests.post(f"{API_BASE_URL}/portfolio/backtest", json=payload)
        
        if response.status_code == 200:
            try:
                response_data = response.json()
                logger.info(f"Received portfolio backtest response")
                results = response_data.get('results', {})
                
                # Get data source information
                data_sources = results.get("data_sources", {})
                
                # Create metrics table
                metrics = results.get("metrics", {})
                
                # Format metrics table
                metrics_table = [
                    {"Metric": k, "Value": f"{v:.2%}" if "return" in k.lower() or "drawdown" in k.lower() or "volatility" in k.lower() else f"{v:.4f}"}
                    for k, v in metrics.items()
                ]
                
                # Add data sources to metrics table
                for symbol, source in data_sources.items():
                    metrics_table.append({"Metric": f"Data Source ({symbol})", "Value": source.upper()})
                
                # Create plot
                plot = plot_portfolio_results(results)
                
                # Add insights if available
                insights = results.get("insights", "")
                
                # Return in order: metrics table, plot, results JSON, insights
                return metrics_table, plot, json.dumps(results, indent=2), insights
            except json.JSONDecodeError as je:
                logger.error(f"JSON decode error: {je}, Response content: {response.text}")
                return [], None, f"Error decoding response: {response.text}", ""
        else:
            error_msg = f"Error: {response.status_code}"
            try:
                error_detail = response.json().get("detail", "")
                error_msg += f" - {error_detail}"
            except:
                error_msg += f" - {response.text}"
            
            logger.error(error_msg)
            return [], None, error_msg, ""
    except Exception as e:
        logger.error(f"Error running portfolio backtest: {e}")
        return [], None, f"Error: {str(e)}", ""

def plot_portfolio_results(results):
    """Create a plotly figure from portfolio backtest results"""
    try:
        # Extract equity curve
        equity_curve = results.get("equity_curve", [])
        
        if not equity_curve:
            fig = go.Figure()
            fig.update_layout(title="No equity curve data available")
            return fig
        
        # Convert to DataFrame
        dates = [item[0] for item in equity_curve]
        values = [item[1] for item in equity_curve]
        
        # Create figure
        fig = go.Figure()
        
        # Add equity curve
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=values,
                mode="lines",
                name="Portfolio Value",
                line=dict(color="blue", width=2)
            )
        )
        
        # Update layout
        fig.update_layout(
            title="Portfolio Backtest Results",
            xaxis_title="Date",
            yaxis_title="Portfolio Value",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=20, r=20, t=40, b=20),
            height=500
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error creating portfolio plot: {e}")
        fig = go.Figure()
        fig.update_layout(title=f"Error creating plot: {str(e)}")
        return fig

def create_ui():
    """Create the Gradio UI"""
    with gr.Blocks(title="AI Trading Platform", theme=gr.themes.Default()) as demo:
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
                        
                with gr.Row():
                    refresh_portfolio_btn = gr.Button("Refresh Portfolio")
                    
                # Wire up refresh button
                refresh_portfolio_btn.click(
                    fn=update_portfolio,
                    inputs=[],
                    outputs=[portfolio_json, positions_table]
                )
            
            # Backtest Tab
            with gr.Tab("Backtest"):
                with gr.Row():
                    with gr.Column(scale=1):
                        backtest_agent = gr.Dropdown(
                            ["value_agent", "trend_agent", "sentiment_agent", "ensemble_agent", "customizable_agent"],
                            label="Agent",
                            value="value_agent"
                        )
                        backtest_symbols = gr.Textbox(label="Symbols (comma-separated)", value="AAPL,MSFT,GOOG")
                        backtest_start = gr.Textbox(label="Start Date (YYYY-MM-DD)", value="2023-01-01")
                        backtest_end = gr.Textbox(label="End Date (YYYY-MM-DD)", value="2023-12-31")
                        backtest_capital = gr.Number(label="Initial Capital", value=10000)
                        
                        # Strategy saving options
                        with gr.Accordion("Strategy Management", open=False):
                            strategy_name = gr.Textbox(label="Strategy Name", value="", placeholder="Enter a name for this strategy")
                            strategy_description = gr.Textbox(label="Description", value="", placeholder="Enter a description for this strategy", lines=2)
                            save_strategy = gr.Checkbox(label="Save Strategy for Live Trading", value=False)
                
                # Create a column for customizable agent settings that's initially hidden
                with gr.Column(scale=1, visible=False) as customizable_settings:
                    gr.Markdown("### Customizable Agent Settings")
                    
                    with gr.Accordion("Strategy Selection", open=True):
                        # Strategy toggles with descriptions
                        momentum_enabled = gr.Checkbox(label="Momentum Strategy", value=True, 
                                                      info="Uses RSI, MACD and price momentum to identify trending markets")
                        mean_reversion_enabled = gr.Checkbox(label="Mean Reversion Strategy", value=True,
                                                         info="Uses Bollinger Bands to buy oversold and sell overbought conditions")
                        breakout_enabled = gr.Checkbox(label="Breakout Strategy", value=True,
                                                    info="Detects price breakouts from ranges with volume confirmation")
                        volatility_enabled = gr.Checkbox(label="Volatility Strategy", value=True,
                                                     info="Uses ATR (Average True Range) to identify high volatility opportunities")
                    
                    with gr.Accordion("Strategy Parameters", open=True):
                        # Technical indicator parameters
                        gr.Markdown("#### Technical Indicators")
                        # RSI parameters
                        rsi_period = gr.Slider(minimum=5, maximum=30, value=14, step=1, label="RSI Period")
                        rsi_overbought = gr.Slider(minimum=60, maximum=90, value=70, step=1, label="RSI Overbought Level")
                        rsi_oversold = gr.Slider(minimum=10, maximum=40, value=30, step=1, label="RSI Oversold Level")
                        
                        # Moving Average parameters
                        ma_fast_period = gr.Slider(minimum=5, maximum=50, value=12, step=1, label="Fast MA Period")
                        ma_slow_period = gr.Slider(minimum=20, maximum=200, value=26, step=1, label="Slow MA Period")
                        
                        # Bollinger Band parameters
                        bb_period = gr.Slider(minimum=10, maximum=50, value=20, step=1, label="Bollinger Band Period")
                        bb_std = gr.Slider(minimum=1.0, maximum=3.0, value=2.0, step=0.1, label="Bollinger Band Standard Deviation")
                    
                    with gr.Accordion("Strategy Weights", open=True):
                        # Strategy weights
                        gr.Markdown("#### Strategy Weights")
                        momentum_weight = gr.Slider(minimum=0, maximum=2, value=1.0, step=0.1, label="Momentum Weight")
                        mean_reversion_weight = gr.Slider(minimum=0, maximum=2, value=1.0, step=0.1, label="Mean Reversion Weight")
                        breakout_weight = gr.Slider(minimum=0, maximum=2, value=1.0, step=0.1, label="Breakout Weight")
                        volatility_weight = gr.Slider(minimum=0, maximum=2, value=0.8, step=0.1, label="Volatility Weight")
                    
                    with gr.Accordion("Risk Management", open=True):
                        # Risk parameters
                        gr.Markdown("#### Risk Parameters")
                        max_position_size = gr.Slider(minimum=0.01, maximum=1.0, value=0.2, step=0.01, 
                                                      label="Max Position Size", info="Maximum % of portfolio in one position")
                        stop_loss = gr.Slider(minimum=0.01, maximum=0.2, value=0.05, step=0.01, 
                                              label="Stop Loss (%)", info="Exit position if losses exceed this percentage")
                        take_profit = gr.Slider(minimum=0.01, maximum=0.4, value=0.1, step=0.01, 
                                               label="Take Profit (%)", info="Exit position when gains reach this percentage")
                        trailing_stop = gr.Checkbox(label="Use Trailing Stop", value=True, 
                                                 info="Adjusts stop loss as price moves in favorable direction")
                
                # Run button at the bottom
                run_backtest_btn = gr.Button("Run Backtest")
            
            with gr.Row():
                backtest_metrics = gr.DataFrame(
                    headers=["Metric", "Value"],
                    label="Performance Metrics"
                )
            
            with gr.Row():
                backtest_plot = gr.Plot(label="Equity Curve")
            
            with gr.Row():
                backtest_results = gr.JSON(label="Backtest Results")
            
            # Add JavaScript to show/hide customizable agent settings
            backtest_agent.change(
                fn=lambda agent: {"visible": agent == "customizable_agent"},
                inputs=[backtest_agent],
                outputs=[customizable_settings]
            )
            
            # Function to prepare customizable agent config
            def prepare_customizable_config(agent_type, momentum_enabled, mean_reversion_enabled, 
                                          breakout_enabled, volatility_enabled, 
                                          rsi_period, rsi_overbought, rsi_oversold,
                                          ma_fast_period, ma_slow_period, bb_period, bb_std,
                                          momentum_weight, mean_reversion_weight, breakout_weight, volatility_weight,
                                          max_position_size, stop_loss, take_profit, trailing_stop):
                # Only prepare config if using customizable agent
                if agent_type != "customizable_agent":
                    return agent_type
                
                # Create JSON config as a string
                config = {
                    "strategies": {
                        "momentum": {
                            "enabled": momentum_enabled, 
                            "weight": momentum_weight,
                            "params": {
                                "rsi_period": rsi_period,
                                "rsi_overbought": rsi_overbought,
                                "rsi_oversold": rsi_oversold,
                                "macd_fast": ma_fast_period,
                                "macd_slow": ma_slow_period,
                                "macd_signal": 9
                            }
                        },
                        "mean_reversion": {
                            "enabled": mean_reversion_enabled, 
                            "weight": mean_reversion_weight,
                            "params": {
                                "bollinger_period": bb_period,
                                "bollinger_std": bb_std,
                                "vwap_period": 14
                            }
                        },
                        "breakout": {
                            "enabled": breakout_enabled, 
                            "weight": breakout_weight,
                            "params": {
                                "atr_period": 14,
                                "atr_multiplier": 2.0,
                                "ma_periods": [20, 50, 200]
                            }
                        },
                        "volatility": {
                            "enabled": volatility_enabled, 
                            "weight": volatility_weight,
                            "params": {
                                "atr_period": 14,
                                "atr_multiplier": 1.5
                            }
                        }
                    },
                    "position_sizing": {
                        "max_position_size": max_position_size,
                        "signal_threshold": 0.5,
                        "scale_by_strength": True
                    },
                    "risk_management": {
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "trailing_stop": trailing_stop
                    }
                }
                
                # Convert to JSON string to pass to backend
                import json
                return f"customizable_agent:{json.dumps(config)}"
            
            # Let's simplify the approach and move the function into the main run_backtest
            # Define a modified version of run_backtest that processes the agent settings
            def process_and_run_backtest(agent_type, 
                                       # Strategy toggles
                                       momentum_enabled, mean_reversion_enabled, breakout_enabled, volatility_enabled,
                                       # Technical indicator parameters
                                       rsi_period, rsi_overbought, rsi_oversold,
                                       ma_fast_period, ma_slow_period, bb_period, bb_std,
                                       # Strategy weights
                                       momentum_weight, mean_reversion_weight, breakout_weight, volatility_weight,
                                       # Risk parameters
                                       max_position_size, stop_loss, take_profit, trailing_stop,
                                       # Core backtest parameters
                                       symbols, start_date, end_date, initial_capital,
                                       # Strategy saving parameters
                                       strategy_name, strategy_description, save_strategy):
                
                # Handle customizable agent
                if agent_type == "customizable_agent":
                    # Create comprehensive config structure with strategy params
                    config = {
                        "strategies": {
                            "momentum": {
                                "enabled": momentum_enabled, 
                                "weight": momentum_weight,
                                "params": {
                                    "rsi_period": int(rsi_period),
                                    "rsi_overbought": int(rsi_overbought),
                                    "rsi_oversold": int(rsi_oversold),
                                    "macd_fast": int(ma_fast_period),
                                    "macd_slow": int(ma_slow_period),
                                    "macd_signal": 9
                                }
                            },
                            "mean_reversion": {
                                "enabled": mean_reversion_enabled, 
                                "weight": mean_reversion_weight,
                                "params": {
                                    "bollinger_period": int(bb_period),
                                    "bollinger_std": float(bb_std),
                                    "vwap_period": 14
                                }
                            },
                            "breakout": {
                                "enabled": breakout_enabled, 
                                "weight": breakout_weight,
                                "params": {
                                    "atr_period": 14,
                                    "atr_multiplier": 2.0,
                                    "ma_periods": [20, 50, 200]
                                }
                            },
                            "volatility": {
                                "enabled": volatility_enabled, 
                                "weight": volatility_weight,
                                "params": {
                                    "atr_period": 14,
                                    "atr_multiplier": 1.5
                                }
                            }
                        },
                        "position_sizing": {
                            "max_position_size": float(max_position_size),
                            "signal_threshold": 0.5,
                            "scale_by_strength": True
                        },
                        "risk_management": {
                            "stop_loss": float(stop_loss),
                            "take_profit": float(take_profit),
                            "trailing_stop": trailing_stop
                        }
                    }
                    # Call the standard backtest function with the processed config and saving options
                    return run_backtest(
                        agent_type, symbols, start_date, end_date, initial_capital, 
                        config=config, 
                        strategy_name=strategy_name, 
                        strategy_description=strategy_description,
                        save_strategy=save_strategy
                    )
                else:
                    # Just pass through to the standard function with strategy saving options
                    return run_backtest(
                        agent_type, symbols, start_date, end_date, initial_capital,
                        strategy_name=strategy_name,
                        strategy_description=strategy_description,
                        save_strategy=save_strategy
                    )
            
            # Show/hide customizable settings based on agent selection
            backtest_agent.change(fn=lambda x: {customizable_settings: gr.update(visible=(x == "customizable_agent"))}, 
                                inputs=[backtest_agent], outputs=[customizable_settings])
            
            # Set up backtest button with all parameters
            run_backtest_btn.click(
                process_and_run_backtest,
                inputs=[
                    # Agent selection
                    backtest_agent, 
                    # Strategy toggles
                    momentum_enabled, mean_reversion_enabled, breakout_enabled, volatility_enabled,
                    # Technical indicator parameters
                    rsi_period, rsi_overbought, rsi_oversold,
                    ma_fast_period, ma_slow_period, bb_period, bb_std,
                    # Strategy weights
                    momentum_weight, mean_reversion_weight, breakout_weight, volatility_weight,
                    # Risk parameters
                    max_position_size, stop_loss, take_profit, trailing_stop,
                    # Core backtest parameters
                    backtest_symbols, backtest_start, backtest_end, backtest_capital,
                    # Strategy saving parameters
                    strategy_name, strategy_description, save_strategy
                ],
                outputs=[backtest_metrics, backtest_plot, backtest_results]
            )
            
        # Saved Strategies Tab
        with gr.Tab("Saved Strategies"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Saved Trading Strategies")
                    refresh_strategies_btn = gr.Button("Refresh Strategies List")
                    strategies_table = gr.DataFrame(
                        headers=["ID", "Name", "Agent Type", "Symbols", "Created", "Total Return", "Sharpe", "Win Rate"],
                        label="Saved Strategies"
                    )
            
            with gr.Row():
                with gr.Column(scale=1):
                    selected_strategy_id = gr.Textbox(label="Selected Strategy ID", placeholder="Enter strategy ID to view or deploy")
                    view_strategy_btn = gr.Button("View Strategy Details")
                    deploy_strategy_btn = gr.Button("Deploy for Live Trading")
                    broker_dropdown = gr.Dropdown(
                        ["paper", "alpaca", "robinhood"],
                        label="Trading Broker",
                        value="paper"
                    )
                
            with gr.Row():
                strategy_details = gr.JSON(label="Strategy Details")
                deployment_result = gr.JSON(label="Deployment Result")
            
            # Strategy list refresh function
            def fetch_strategies_list():
                try:
                    response = requests.get(f"{API_BASE_URL}/strategies")
                    if response.status_code == 200:
                        strategies = response.json().get("strategies", [])
                        # Format for display in table
                        table_data = []
                        for s in strategies:
                            performance = s.get("performance", {})
                            total_return = performance.get("total_return", 0)
                            sharpe = performance.get("sharpe_ratio", 0)
                            win_rate = performance.get("win_rate", 0)
                            
                            # Format as percentages where appropriate
                            if isinstance(total_return, (int, float)):
                                total_return = f"{total_return*100:.2f}%"
                            if isinstance(win_rate, (int, float)):
                                win_rate = f"{win_rate*100:.2f}%"
                            
                            table_data.append([
                                s.get("id", ""),
                                s.get("name", ""),
                                s.get("agent_type", ""),
                                ", ".join(s.get("symbols", [])),
                                s.get("created_at", "").split("T")[0] if s.get("created_at") else "",
                                total_return,
                                f"{sharpe:.2f}" if isinstance(sharpe, (int, float)) else sharpe,
                                win_rate
                            ])
                        return table_data
                    else:
                        return [[f"Error: {response.status_code}", "", "", "", "", "", "", ""]]
                except Exception as e:
                    logger.error(f"Error fetching strategies list: {e}")
                    return [[f"Error: {str(e)}", "", "", "", "", "", "", ""]]
            
            # View strategy details function
            def view_strategy(strategy_id):
                if not strategy_id:
                    return {"error": "Please enter a strategy ID"}
                try:
                    response = requests.get(f"{API_BASE_URL}/strategies/{strategy_id}")
                    if response.status_code == 200:
                        return response.json()
                    else:
                        return {"error": f"Error: {response.status_code}"}
                except Exception as e:
                    logger.error(f"Error viewing strategy: {e}")
                    return {"error": f"Error: {str(e)}"}
            
            # Deploy strategy function
            def deploy_strategy(strategy_id, broker):
                if not strategy_id:
                    return {"error": "Please enter a strategy ID"}
                try:
                    payload = {"strategy_id": strategy_id, "broker": broker}
                    response = requests.post(f"{API_BASE_URL}/strategies/deploy", json=payload)
                    if response.status_code == 200:
                        return response.json()
                    else:
                        return {"error": f"Error: {response.status_code}"}
                except Exception as e:
                    logger.error(f"Error deploying strategy: {e}")
                    return {"error": f"Error: {str(e)}"}
            
            # Wire up the buttons
            refresh_strategies_btn.click(
                fn=fetch_strategies_list,
                inputs=[],
                outputs=[strategies_table]
            )
            
            view_strategy_btn.click(
                fn=view_strategy,
                inputs=[selected_strategy_id],
                outputs=[strategy_details]
            )
            
            deploy_strategy_btn.click(
                fn=deploy_strategy,
                inputs=[selected_strategy_id, broker_dropdown],
                outputs=[deployment_result]
            )
            
            # Load strategies on startup
            demo.load(
                fn=fetch_strategies_list,
                inputs=[],
                outputs=[strategies_table]
            )
        
        # Portfolio Backtest Tab
        with gr.Tab("Portfolio Backtest"):
            with gr.Row():
                with gr.Column(scale=1):
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
                
            with gr.Row():
                portfolio_metrics = gr.DataFrame(
                    headers=["Metric", "Value"],
                    label="Portfolio Performance Metrics"
                )
            
            with gr.Row():
                portfolio_plot = gr.Plot(label="Portfolio Equity Curve")
            
            with gr.Row():
                with gr.Column(scale=1):
                    portfolio_results = gr.JSON(label="Portfolio Results")
                with gr.Column(scale=1):
                    portfolio_insights = gr.Textbox(label="LLM Insights", lines=20)
            
            # Set up portfolio backtest button
            run_portfolio_btn.click(
                run_portfolio_backtest,
                inputs=[
                    portfolio_allocations, 
                    portfolio_strategies, 
                    portfolio_start, 
                    portfolio_end, 
                    generate_insights
                ],
                outputs=[
                    portfolio_metrics, 
                    portfolio_plot, 
                    portfolio_results,
                    portfolio_insights
                ]
            )
        
        # Initialize with default data
        try:
            demo.load(
                fn=lambda: update_portfolio(),
                outputs=[portfolio_json, positions_table]
            )
        except Exception as e:
            logger.error(f"Error loading initial data: {e}")
        
        # Return the demo for launch
        return demo

if __name__ == "__main__":
    # Create and launch the UI
    demo = create_ui()
    # Launch with host set to 0.0.0.0 to be accessible from outside the container
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
else:
    # For module import, provide a function to launch the UI
    def launch_gradio():
        demo = create_ui()
        demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
