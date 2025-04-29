import gradio as gr
import requests
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import datetime
import os
import logging

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

def run_backtest(agent, symbols, start_date, end_date, initial_capital):
    """Run a backtest via API"""
    try:
        payload = {
            "agent_type": agent,
            "symbols": [s.strip() for s in symbols.split(",")],
            "start_date": start_date,
            "end_date": end_date,
            "initial_capital": float(initial_capital)
        }
        
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
    """Create a plotly figure from backtest results"""
    try:
        # Extract equity curve
        equity_curve = results.get("equity_curve", [])
        if not equity_curve:
            return go.Figure().update_layout(title="No equity curve data available")
        
        dates = [item.get("date") for item in equity_curve]
        equity = [item.get("equity") for item in equity_curve]
        
        # Create figure
        fig = go.Figure()
        
        # Add equity curve
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=equity,
                mode="lines",
                name="Portfolio Value",
                line=dict(color="blue", width=2)
            )
        )
        
        # Add buy/sell markers if available
        trades = results.get("trades", [])
        if trades:
            buy_dates = [trade.get("entry_date") for trade in trades if trade.get("side") == "buy"]
            buy_prices = [trade.get("entry_price") for trade in trades if trade.get("side") == "buy"]
            
            sell_dates = [trade.get("exit_date") for trade in trades if trade.get("exit_date")]
            sell_prices = [trade.get("exit_price") for trade in trades if trade.get("exit_date")]
            
            fig.add_trace(
                go.Scatter(
                    x=buy_dates,
                    y=buy_prices,
                    mode="markers",
                    name="Buy",
                    marker=dict(color="green", size=10, symbol="triangle-up")
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=sell_dates,
                    y=sell_prices,
                    mode="markers",
                    name="Sell",
                    marker=dict(color="red", size=10, symbol="triangle-down")
                )
            )
        
        # Update layout
        fig.update_layout(
            title="Backtest Results",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
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
        logger.error(f"Error creating plot: {e}")
        fig = go.Figure()
        fig.update_layout(title=f"Error creating plot: {str(e)}")
        return fig

def create_ui():
    """Create the Gradio UI"""
    with gr.Blocks(title="AI Trading Platform", theme=gr.themes.Default()) as demo:
        gr.Markdown("# AI-driven Multi-Agent Trading Platform")
        
        # Portfolio Tab
        with gr.Tab("Portfolio"):
            with gr.Row():
                refresh_btn = gr.Button("Refresh Portfolio")
            
            with gr.Row():
                portfolio_json = gr.JSON(label="Portfolio Data")
                positions_table = gr.DataFrame(label="Positions")
            
            # Set up refresh button
            refresh_btn.click(
                update_portfolio,
                inputs=[],
                outputs=[portfolio_json, positions_table]
            )
        
        # Backtest Tab
        with gr.Tab("Backtest"):
            with gr.Row():
                with gr.Column():
                    backtest_agent = gr.Dropdown(
                        ["value_agent", "trend_agent", "sentiment_agent", "ensemble_agent"],
                        label="Agent",
                        value="value_agent"
                    )
                    backtest_symbols = gr.Textbox(label="Symbols (comma-separated)", value="AAPL,MSFT,GOOG")
                    backtest_start = gr.Textbox(label="Start Date (YYYY-MM-DD)", value="2023-01-01")
                    backtest_end = gr.Textbox(label="End Date (YYYY-MM-DD)", value="2023-12-31")
                    backtest_capital = gr.Number(label="Initial Capital", value=10000)
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
            
            # Set up backtest button
            run_backtest_btn.click(
                run_backtest,
                inputs=[backtest_agent, backtest_symbols, backtest_start, backtest_end, backtest_capital],
                outputs=[backtest_metrics, backtest_plot, backtest_results]
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
