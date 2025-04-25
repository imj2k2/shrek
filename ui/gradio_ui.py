import gradio as gr
import requests
import pandas as pd
import plotly.graph_objects as go

def get_portfolio():
    resp = requests.get("http://localhost:8000/portfolio")
    return resp.json().get('portfolio', {}) if resp.ok else {}

def get_signals(market_data):
    resp = requests.post("http://localhost:8000/agents/signal", json={"data": market_data})
    return resp.json().get('signals', {}) if resp.ok else {}

def execute_trade(signal):
    resp = requests.post("http://localhost:8000/agents/execute", json=signal)
    return resp.json().get('result', {}) if resp.ok else {}

def plot_performance(history):
    if not history:
        return go.Figure()
    df = pd.DataFrame(history)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=df['portfolio_value'], mode='lines', name='Portfolio Value'))
    fig.update_layout(title='Portfolio Performance', xaxis_title='Date', yaxis_title='Value')
    return fig

def launch_gradio():
    with gr.Blocks() as demo:
        gr.Markdown("# AI-driven Multi-Agent Trading Platform")
        with gr.Tab("Portfolio"):
            portfolio = get_portfolio()
            gr.JSON(portfolio)
            perf_plot = gr.Plot(plot_performance(portfolio.get('history', [])))
        with gr.Tab("Signals"):
            market_data_input = gr.Textbox(label="Market Data (JSON)")
            signals_output = gr.JSON(label="Signals")
            get_signals_btn = gr.Button("Get Signals")
            get_signals_btn.click(
                lambda x: get_signals(eval(x)),
                inputs=market_data_input,
                outputs=signals_output
            )
        with gr.Tab("Trade Execution"):
            signal_input = gr.Textbox(label="Signal (JSON)")
            trade_result = gr.JSON(label="Execution Result")
            execute_btn = gr.Button("Execute Trade")
            execute_btn.click(
                lambda x: execute_trade(eval(x)),
                inputs=signal_input,
                outputs=trade_result
            )
    demo.launch(server_port=7860, share=False)
