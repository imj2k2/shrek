import gradio as gr
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import datetime
import os
import sys

# Add project root to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backtest.backtest_engine import Backtester

# API endpoint configuration - use environment variable or default to localhost
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

def get_portfolio():
    """Fetch portfolio data from API"""
    try:
        resp = requests.get(f"{API_BASE_URL}/portfolio")
        if resp.ok:
            return resp.json().get('portfolio', {})
        else:
            return {"error": f"Failed to fetch portfolio: {resp.status_code}"}
    except Exception as e:
        return {"error": f"Error connecting to API: {str(e)}"}

def get_signals(market_data):
    """Get trading signals based on market data"""
    try:
        resp = requests.post(f"{API_BASE_URL}/agents/signal", json={"data": market_data})
        if resp.ok:
            return resp.json().get('signals', {})
        else:
            return {"error": f"Failed to get signals: {resp.status_code}"}
    except Exception as e:
        return {"error": f"Error connecting to API: {str(e)}"}

def execute_trade(signal):
    """Execute a trade based on signal"""
    try:
        resp = requests.post(f"{API_BASE_URL}/agents/execute", json=signal)
        if resp.ok:
            return resp.json().get('result', {})
        else:
            return {"error": f"Failed to execute trade: {resp.status_code}"}
    except Exception as e:
        return {"error": f"Error connecting to API: {str(e)}"}

def get_risk_metrics():
    """Fetch risk metrics from API"""
    try:
        resp = requests.get(f"{API_BASE_URL}/risk/metrics")
        if resp.ok:
            return resp.json().get('metrics', {})
        else:
            return {"error": f"Failed to fetch risk metrics: {resp.status_code}"}
    except Exception as e:
        return {"error": f"Error connecting to API: {str(e)}"}

def plot_portfolio_performance(history):
    """Create portfolio performance chart with equity curve and drawdown"""
    if not history or len(history) < 2:
        fig = go.Figure()
        fig.update_layout(title="No portfolio history available")
        return fig
    
    # Convert to DataFrame
    df = pd.DataFrame(history)
    
    # Create figure with secondary y-axis
    fig = go.Figure()
    
    # Add equity curve
    fig.add_trace(
        go.Scatter(
            x=df['date'], 
            y=df['portfolio_value'], 
            mode='lines',
            name='Portfolio Value',
            line=dict(color='blue', width=2)
        )
    )
    
    # Add drawdown if available
    if 'drawdown' in df.columns:
        # Create secondary Y axis for drawdown
        fig.add_trace(
            go.Scatter(
                x=df['date'], 
                y=df['drawdown'] * 100,  # Convert to percentage
                mode='lines',
                name='Drawdown %',
                line=dict(color='red', width=1.5),
                yaxis="y2"
            )
        )
        
        # Update layout with secondary y-axis
        fig.update_layout(
            yaxis2=dict(
                title="Drawdown %",
                titlefont=dict(color="red"),
                tickfont=dict(color="red"),
                anchor="x",
                overlaying="y",
                side="right",
                range=[0, max(df['drawdown'] * 100) * 1.5]  # Scale for better visibility
            )
        )
    
    # Update layout
    fig.update_layout(
        title='Portfolio Performance',
        xaxis_title='Date',
        yaxis_title='Portfolio Value ($)',
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

def create_positions_table(portfolio):
    """Create a formatted positions table from portfolio data"""
    if not portfolio or 'positions' not in portfolio or not portfolio['positions']:
        return pd.DataFrame(columns=['Symbol', 'Quantity', 'Entry Price', 'Current Price', 'P&L', 'P&L %'])
    
    positions = portfolio['positions']
    
    # Convert to DataFrame
    data = []
    for symbol, pos in positions.items():
        entry_price = pos.get('avg_price', 0)
        current_price = pos.get('current_price', entry_price)
        quantity = pos.get('quantity', 0)
        
        # Calculate P&L
        pnl = (current_price - entry_price) * quantity
        pnl_pct = ((current_price / entry_price) - 1) * 100 if entry_price > 0 else 0
        
        data.append({
            'Symbol': symbol,
            'Quantity': quantity,
            'Entry Price': f"${entry_price:.2f}",
            'Current Price': f"${current_price:.2f}",
            'P&L': f"${pnl:.2f}",
            'P&L %': f"{pnl_pct:.2f}%"
        })
    
    return pd.DataFrame(data)

def run_backtest(agent_type, symbols, start_date, end_date, initial_capital):
    """Run a backtest and return results"""
    try:
        # Convert inputs
        symbols_list = [s.strip() for s in symbols.split(',')]
        capital = float(initial_capital)
        
        # Call backtest API
        payload = {
            "agent_type": agent_type,
            "symbols": symbols_list,
            "start_date": start_date,
            "end_date": end_date,
            "initial_capital": capital
        }
        
        resp = requests.post(f"{API_BASE_URL}/backtest/run", json=payload)
        
        if resp.ok:
            return resp.json().get('results', {})
        else:
            return {"error": f"Backtest failed: {resp.status_code}", "details": resp.text}
    except Exception as e:
        return {"error": f"Error running backtest: {str(e)}"}

def plot_backtest_results(results):
    """Plot backtest results"""
    if not results or 'equity_curve' not in results or not results['equity_curve']:
        fig = go.Figure()
        fig.update_layout(title="No backtest results available")
        return fig
    
    # Convert to DataFrame
    df = pd.DataFrame(results['equity_curve'])
    
    # Create figure with secondary y-axis
    fig = go.Figure()
    
    # Add equity curve
    fig.add_trace(
        go.Scatter(
            x=df['date'], 
            y=df['equity'], 
            mode='lines',
            name='Equity',
            line=dict(color='green', width=2)
        )
    )
    
    # Add drawdown
    if 'drawdown' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['date'], 
                y=df['drawdown'] * 100,
                mode='lines',
                name='Drawdown %',
                line=dict(color='red', width=1.5),
                yaxis="y2"
            )
        )
        
        # Update layout with secondary y-axis
        fig.update_layout(
            yaxis2=dict(
                title="Drawdown %",
                titlefont=dict(color="red"),
                tickfont=dict(color="red"),
                anchor="x",
                overlaying="y",
                side="right"
            )
        )
    
    # Update layout
    fig.update_layout(
        title=f"Backtest Results: {results.get('strategy_name', 'Strategy')}",
        xaxis_title='Date',
        yaxis_title='Equity ($)',
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

def launch_gradio():
    """Launch the Gradio UI"""
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# AI-driven Multi-Agent Trading Platform")
        
        # Portfolio Tab
        with gr.Tab("Portfolio"):
            with gr.Row():
                refresh_portfolio_btn = gr.Button("Refresh Portfolio")
            
            with gr.Row():
                with gr.Column(scale=2):
                    portfolio_json = gr.JSON(label="Portfolio Details")
                with gr.Column(scale=1):
                    portfolio_summary = gr.Dataframe(
                        headers=["Metric", "Value"],
                        label="Summary"
                    )
            
            with gr.Row():
                perf_plot = gr.Plot(label="Portfolio Performance")
            
            with gr.Row():
                positions_table = gr.Dataframe(label="Current Positions")
            
            # Update portfolio data when refresh button is clicked
            def update_portfolio():
                portfolio = get_portfolio()
                
                # Create summary
                summary = []
                if isinstance(portfolio, dict) and 'error' not in portfolio:
                    summary = [
                        ["Total Equity", f"${portfolio.get('equity', 0):.2f}"],
                        ["Cash", f"${portfolio.get('cash', 0):.2f}"],
                        ["Invested", f"${portfolio.get('invested', 0):.2f}"],
                        ["Day P&L", f"${portfolio.get('day_pnl', 0):.2f}"],
                        ["Total P&L", f"${portfolio.get('total_pnl', 0):.2f}"],
                        ["Total P&L %", f"{portfolio.get('total_pnl_pct', 0):.2f}%"]
                    ]
                
                # Create positions table
                positions_df = create_positions_table(portfolio)
                
                # Create performance plot
                plot = plot_portfolio_performance(portfolio.get('history', []))
                
                return portfolio, summary, plot, positions_df
            
            refresh_portfolio_btn.click(
                update_portfolio,
                outputs=[portfolio_json, portfolio_summary, perf_plot, positions_table]
            )
        
        # Trading Signals Tab
        with gr.Tab("Trading Signals"):
            with gr.Row():
                with gr.Column():
                    symbol_input = gr.Textbox(label="Symbol", value="AAPL")
                    lookback_input = gr.Slider(minimum=1, maximum=30, value=10, step=1, label="Lookback Days")
                    get_signals_btn = gr.Button("Get Trading Signals")
            
            with gr.Row():
                signals_output = gr.JSON(label="Trading Signals")
            
            with gr.Row():
                signal_summary = gr.Dataframe(
                    headers=["Agent", "Signal", "Confidence", "Target Price"],
                    label="Signal Summary"
                )
            
            # Get signals when button is clicked
            def fetch_signals(symbol, lookback):
                market_data = {
                    "symbol": symbol,
                    "lookback": int(lookback)
                }
                
                signals = get_signals(market_data)
                
                # Create summary table
                summary = []
                if isinstance(signals, dict) and 'error' not in signals:
                    for agent, signal in signals.items():
                        if isinstance(signal, dict):
                            summary.append([
                                agent,
                                signal.get('action', 'unknown'),
                                f"{signal.get('confidence', 0):.2f}",
                                f"${signal.get('target_price', 0):.2f}"
                            ])
                
                return signals, summary
            
            get_signals_btn.click(
                fetch_signals,
                inputs=[symbol_input, lookback_input],
                outputs=[signals_output, signal_summary]
            )
        
        # Trade Execution Tab
        with gr.Tab("Trade Execution"):
            with gr.Row():
                with gr.Column():
                    symbol_exec = gr.Textbox(label="Symbol", value="AAPL")
                    action_exec = gr.Dropdown(
                        choices=["buy", "sell", "hold"],
                        label="Action",
                        value="buy"
                    )
                    quantity_exec = gr.Number(label="Quantity", value=10)
                    price_exec = gr.Number(label="Price", value=0)
                    execute_btn = gr.Button("Execute Trade")
            
            with gr.Row():
                trade_result = gr.JSON(label="Execution Result")
                trade_status = gr.Textbox(label="Status")
            
            # Execute trade when button is clicked
            def do_execute_trade(symbol, action, quantity, price):
                signal = {
                    "symbol": symbol,
                    "action": action,
                    "quantity": float(quantity)
                }
                
                if price > 0:
                    signal["price"] = float(price)
                
                result = execute_trade(signal)
                
                status = "Trade executed successfully!"
                if isinstance(result, dict) and 'error' in result:
                    status = f"Error: {result['error']}"
                
                return result, status
            
            execute_btn.click(
                do_execute_trade,
                inputs=[symbol_exec, action_exec, quantity_exec, price_exec],
                outputs=[trade_result, trade_status]
            )
        
        # Risk Management Tab
        with gr.Tab("Risk Management"):
            with gr.Row():
                refresh_risk_btn = gr.Button("Refresh Risk Metrics")
            
            with gr.Row():
                with gr.Column():
                    risk_metrics = gr.JSON(label="Risk Metrics")
                with gr.Column():
                    risk_summary = gr.Dataframe(
                        headers=["Metric", "Value", "Status"],
                        label="Risk Summary"
                    )
            
            # Update risk metrics when refresh button is clicked
            def update_risk_metrics():
                metrics = get_risk_metrics()
                
                # Create summary table
                summary = []
                if isinstance(metrics, dict) and 'error' not in metrics:
                    for key, value in metrics.items():
                        if key == 'alerts':
                            continue
                            
                        status = "✅ OK"
                        if key == 'drawdown' and value > 0.1:
                            status = "⚠️ Warning"
                        if key == 'drawdown' and value > 0.2:
                            status = "🛑 Critical"
                            
                        # Format value based on type
                        formatted_value = value
                        if isinstance(value, float):
                            if key.endswith('_pct') or key in ['drawdown']:
                                formatted_value = f"{value*100:.2f}%"
                            else:
                                formatted_value = f"{value:.2f}"
                                
                        summary.append([key, formatted_value, status])
                
                return metrics, summary
            
            refresh_risk_btn.click(
                update_risk_metrics,
                outputs=[risk_metrics, risk_summary]
            )
        
        # Backtesting Tab
        with gr.Tab("Backtesting"):
            with gr.Row():
                with gr.Column():
                    backtest_agent = gr.Dropdown(
                        choices=["stocks", "options", "crypto"],
                        label="Agent Type",
                        value="stocks"
                    )
                    backtest_symbols = gr.Textbox(label="Symbols (comma-separated)", value="AAPL,MSFT")
                    backtest_start = gr.Textbox(
                        label="Start Date (YYYY-MM-DD)",
                        value=(datetime.datetime.now() - datetime.timedelta(days=365)).strftime('%Y-%m-%d')
                    )
                    backtest_end = gr.Textbox(
                        label="End Date (YYYY-MM-DD)",
                        value=datetime.datetime.now().strftime('%Y-%m-%d')
                    )
                    backtest_capital = gr.Number(label="Initial Capital", value=100000)
                    run_backtest_btn = gr.Button("Run Backtest")
            
            with gr.Row():
                backtest_results = gr.JSON(label="Backtest Results")
            
            with gr.Row():
                backtest_plot = gr.Plot(label="Equity Curve")
            
            with gr.Row():
                backtest_metrics = gr.Dataframe(
                    headers=["Metric", "Value"],
                    label="Performance Metrics"
                )
            
            # Run backtest when button is clicked
            def do_run_backtest(agent_type, symbols, start_date, end_date, capital):
                results = run_backtest(agent_type, symbols, start_date, end_date, capital)
                
                # Create metrics table
                metrics_table = []
                if isinstance(results, dict) and 'error' not in results and 'metrics' in results:
                    for key, value in results['metrics'].items():
                        formatted_value = value
                        if isinstance(value, float):
                            if key in ['total_return', 'annual_return', 'max_drawdown', 'win_rate']:
                                formatted_value = f"{value*100:.2f}%"
                            else:
                                formatted_value = f"{value:.4f}"
                                
                        metrics_table.append([key, formatted_value])
                
                # Create plot
                plot = plot_backtest_results(results)
                
                return results, plot, metrics_table
            
            run_backtest_btn.click(
                do_run_backtest,
                inputs=[backtest_agent, backtest_symbols, backtest_start, backtest_end, backtest_capital],
                outputs=[backtest_results, backtest_plot, backtest_metrics]
            )
    
    # Initialize with default data
    demo.load(
        fn=lambda: update_portfolio()[0],
        outputs=[portfolio_json]
    )
    
    demo.launch(server_port=7860, share=False)
