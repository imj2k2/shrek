import gradio as gr
import pandas as pd
import logging
import sys
import os
import json
import requests
import plotly.graph_objects as go
from datetime import datetime

# Add parent directory to path to import from data module
sys.path.append('/app')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API base URL
API_BASE_URL = os.environ.get("API_BASE_URL", "http://backend:8000")

def run_backtest(symbols, start_date, end_date, initial_capital, strategy_type):
    """
    Run a backtest using the backend API
    
    Args:
        symbols: Comma-separated string of symbols
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        initial_capital: Initial capital for the backtest
        strategy_type: Strategy type (MeanReversion, Momentum, etc.)
        
    Returns:
        Tuple of (backtest_results_text, equity_curve_plot)
    """
    try:
        # Parse symbols
        symbol_list = [s.strip() for s in symbols.split(',')]
        
        # Validate dates
        try:
            datetime.strptime(start_date, '%Y-%m-%d')
            datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError:
            return "Error: Invalid date format. Please use YYYY-MM-DD format.", None
        
        # Validate initial capital
        if not initial_capital or initial_capital <= 0:
            initial_capital = 10000  # Default value
        
        # Prepare request payload
        payload = {
            "agent_type": strategy_type,
            "symbols": symbol_list,
            "start_date": start_date,
            "end_date": end_date,
            "initial_capital": float(initial_capital),
            "timeframe": "day",
            "strategy_name": f"{strategy_type}_{datetime.now().strftime('%Y%m%d')}",
            "description": f"Backtest of {strategy_type} strategy on {', '.join(symbol_list)}",
            "save_strategy": True
        }
        
        logger.info(f"Running backtest with payload: {payload}")
        
        # Make API request to backend
        response = requests.post(
            f"{API_BASE_URL}/backtest/run",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        # Check if request was successful
        if response.status_code == 200:
            result = response.json()
            
            # Format results as text
            results_text = f"Backtest Results for {strategy_type} Strategy\n"
            results_text += f"Symbols: {', '.join(symbol_list)}\n"
            results_text += f"Period: {start_date} to {end_date}\n"
            results_text += f"Initial Capital: ${initial_capital:,.2f}\n\n"
            
            # Add performance metrics
            if "results" in result and "metrics" in result["results"]:
                perf = result["results"]["metrics"]
                results_text += f"Final Portfolio Value: ${perf.get('final_value', perf.get('total_value', 0)):,.2f}\n"
                results_text += f"Total Return: {perf.get('total_return', 0) * 100:.2f}%\n"
                results_text += f"Annual Return: {perf.get('annual_return', 0) * 100:.2f}%\n"
                results_text += f"Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}\n"
                results_text += f"Max Drawdown: {perf.get('max_drawdown', 0) * 100:.2f}%\n"
                results_text += f"Win Rate: {perf.get('win_rate', 0) * 100:.2f}%\n"
                # Add additional metrics if available
                if 'total_trades' in perf:
                    results_text += f"Total Trades: {perf.get('total_trades', 0)}\n"
                if 'profit_factor' in perf:
                    results_text += f"Profit Factor: {perf.get('profit_factor', 0):.2f}\n"
            elif "performance" in result:
                perf = result["performance"]
                results_text += f"Final Portfolio Value: ${perf.get('final_value', 0):,.2f}\n"
                results_text += f"Total Return: {perf.get('total_return', 0) * 100:.2f}%\n"
                results_text += f"Annual Return: {perf.get('annual_return', 0) * 100:.2f}%\n"
                results_text += f"Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}\n"
                results_text += f"Max Drawdown: {perf.get('max_drawdown', 0) * 100:.2f}%\n"
                results_text += f"Win Rate: {perf.get('win_rate', 0) * 100:.2f}%\n"
            else:
                results_text += "No performance metrics available\n"
            
            # Create equity curve plot if data is available
            equity_curve = None
            if "results" in result and "equity_curve" in result["results"]:
                equity_data = result["results"]["equity_curve"]
                # If we have raw equity curve data, process it
                if isinstance(equity_data, list) and equity_data:
                    try:
                        # Convert to dataframe for easier plotting
                        df = pd.DataFrame(equity_data)
                        if 'date' in df.columns and 'equity' in df.columns:
                            # Convert date strings to datetime objects
                            df['date'] = pd.to_datetime(df['date'])
                            
                            # Create Plotly figure
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=df['date'],
                                y=df['equity'],
                                mode='lines',
                                name='Portfolio Value'
                            ))
                            
                            # Add benchmark if available
                            if 'benchmark' in df.columns:
                                fig.add_trace(go.Scatter(
                                    x=df['date'],
                                    y=df['benchmark'],
                                    mode='lines',
                                    name='Benchmark (SPY)',
                                    line=dict(dash='dash')
                                ))
                            
                            # Customize layout
                            fig.update_layout(
                                title=f"{strategy_type} Strategy Backtest Results",
                                xaxis_title="Date",
                                yaxis_title="Portfolio Value ($)",
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="right",
                                    x=1
                                )
                            )
                            
                            equity_curve = fig
                    except Exception as e:
                        logger.error(f"Error creating equity curve plot: {e}")
                        # Fall back to default plotting logic
                        equity_curve = None
            elif "equity_curve" in result:
                try:
                    # Convert equity curve data to DataFrame
                    equity_data = result["equity_curve"]
                    dates = [datetime.fromisoformat(d.replace('Z', '+00:00')) for d in equity_data["dates"]]
                    values = equity_data["values"]
                    
                    # Create Plotly figure
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=dates,
                        y=values,
                        mode='lines',
                        name='Portfolio Value'
                    ))
                    
                    # Add benchmark if available
                    if "benchmark" in equity_data:
                        benchmark_values = equity_data["benchmark"]
                        fig.add_trace(go.Scatter(
                            x=dates,
                            y=benchmark_values,
                            mode='lines',
                            name='Benchmark (SPY)',
                            line=dict(dash='dash')
                        ))
                    
                    # Customize layout
                    fig.update_layout(
                        title=f"{strategy_type} Strategy Backtest Results",
                        xaxis_title="Date",
                        yaxis_title="Portfolio Value ($)",
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    equity_curve = fig
                except Exception as e:
                    logger.error(f"Error creating equity curve plot: {str(e)}")
            
            return results_text, equity_curve
        else:
            error_msg = f"Error: API returned status code {response.status_code}"
            try:
                error_details = response.json()
                error_msg += f"\nDetails: {json.dumps(error_details, indent=2)}"
            except:
                error_msg += f"\nResponse: {response.text}"
            
            logger.error(error_msg)
            return error_msg, None
    
    except requests.exceptions.RequestException as e:
        error_msg = f"Error connecting to backend API: {str(e)}"
        logger.error(error_msg)
        return error_msg, None
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Error running backtest: {error_details}")
        return f"Error running backtest: {str(e)}", None

def run_mock_backtest(symbols, start_date, end_date, initial_capital, strategy_type):
    """
    Run a mock backtest for testing purposes when the backend API is not available
    
    Args:
        symbols: Comma-separated string of symbols
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        initial_capital: Initial capital for the backtest
        strategy_type: Strategy type (MeanReversion, Momentum, etc.)
        
    Returns:
        Tuple of (backtest_results_text, equity_curve_plot)
    """
    import numpy as np
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Parse symbols
    symbol_list = [s.strip() for s in symbols.split(',')]
    
    # Generate mock results
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Create date range
    date_range = []
    current = start
    while current <= end:
        if current.weekday() < 5:  # Only weekdays
            date_range.append(current)
        current += timedelta(days=1)
    
    # Generate random equity curve with upward trend
    np.random.seed(42)  # For reproducibility
    
    # Start with initial capital
    equity = [float(initial_capital)]
    
    # Generate random daily returns with a slight positive bias
    for i in range(1, len(date_range)):
        daily_return = np.random.normal(0.0005, 0.01)  # Mean slightly positive, std=1%
        new_value = equity[-1] * (1 + daily_return)
        equity.append(new_value)
    
    # Create benchmark (slightly worse performance)
    benchmark = [float(initial_capital)]
    for i in range(1, len(date_range)):
        daily_return = np.random.normal(0.0003, 0.01)  # Lower mean
        new_value = benchmark[-1] * (1 + daily_return)
        benchmark.append(new_value)
    
    # Calculate performance metrics
    total_return = (equity[-1] / equity[0]) - 1
    days = (end - start).days
    annual_return = ((1 + total_return) ** (365 / days)) - 1
    
    # Calculate drawdowns
    drawdowns = []
    peak = equity[0]
    for value in equity:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        drawdowns.append(drawdown)
    max_drawdown = max(drawdowns)
    
    # Create Plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=date_range,
        y=equity,
        mode='lines',
        name='Portfolio Value'
    ))
    
    fig.add_trace(go.Scatter(
        x=date_range,
        y=benchmark,
        mode='lines',
        name='Benchmark (SPY)',
        line=dict(dash='dash')
    ))
    
    # Customize layout
    fig.update_layout(
        title=f"{strategy_type} Strategy Backtest Results (Mock Data)",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Format results as text
    results_text = f"Backtest Results for {strategy_type} Strategy (MOCK DATA)\n"
    results_text += f"Symbols: {', '.join(symbol_list)}\n"
    results_text += f"Period: {start_date} to {end_date}\n"
    results_text += f"Initial Capital: ${initial_capital:,.2f}\n\n"
    results_text += f"Final Portfolio Value: ${equity[-1]:,.2f}\n"
    results_text += f"Total Return: {total_return * 100:.2f}%\n"
    results_text += f"Annual Return: {annual_return * 100:.2f}%\n"
    results_text += f"Sharpe Ratio: {np.random.uniform(0.8, 2.5):.2f}\n"
    results_text += f"Max Drawdown: {max_drawdown * 100:.2f}%\n"
    results_text += f"Win Rate: {np.random.uniform(48, 65):.2f}%\n"
    results_text += "\nNOTE: This is mock data for testing purposes only."
    
    return results_text, fig
