from mcp.protocol import MCPProtocol
from agents import CoordinatorAgent, StocksAgent, OptionsAgent, CryptoAgent, RiskAgent, TradingExecutor
from trading.lumibot_wrapper import LumibotBroker
from backtest.backtest_engine import Backtester
from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import datetime
import logging

# Instantiate agents and MCP
mcp = MCPProtocol()
stocks_agent = StocksAgent()
options_agent = OptionsAgent()
crypto_agent = CryptoAgent()
risk_agent = RiskAgent()
broker = LumibotBroker()
executor = TradingExecutor(broker)
coordinator = CoordinatorAgent({
    'stocks': stocks_agent,
    'options': options_agent,
    'crypto': crypto_agent,
    'risk': risk_agent,
    'executor': executor
})

# Initialize backtester
backtester = Backtester(initial_capital=100000.0)

# FastAPI endpoints
router = APIRouter()

class MarketData(BaseModel):
    data: Dict[str, Any]

class BacktestRequest(BaseModel):
    agent_type: str = Field(..., description="Type of agent to use for backtesting (stocks, options, crypto)")
    symbols: List[str] = Field(..., description="List of symbols to backtest")
    start_date: str = Field(..., description="Start date for backtest (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date for backtest (YYYY-MM-DD)")
    initial_capital: float = Field(100000.0, description="Initial capital for backtest")
    timeframe: str = Field("day", description="Data timeframe (day, hour, 15min, 5min, 1min)")
    max_drawdown: Optional[float] = Field(0.2, description="Maximum drawdown allowed (0.2 = 20%)")
    trailing_stop: Optional[float] = Field(0.05, description="Trailing stop percentage (0.05 = 5%)")
    max_position_size: Optional[float] = Field(0.2, description="Maximum position size as percentage of portfolio")
    strategy_name: Optional[str] = Field(None, description="Name of the strategy")

@router.post('/agents/signal')
def agent_signal(market_data: MarketData):
    signals = coordinator.coordinate(market_data.data)
    return {'signals': signals}

@router.post('/agents/execute')
def execute_trade(signal: Dict[str, Any]):
    result = executor.execute(signal)
    return {'result': result}

@router.post('/backtest/run')
@router.post('/backtest')  # Add an alias for backward compatibility
def run_backtest(request: BacktestRequest):
    """Run a backtest with the specified parameters"""
    try:
        # Log the request for debugging
        logging.info(f"Received backtest request: {request}")
        
        # Handle both agent and agent_type parameters for flexibility
        agent_type = request.agent_type
        
        # Select the appropriate agent based on agent_type
        if agent_type == "stocks" or agent_type == "value_agent":
            agent = StocksAgent()
        elif agent_type == "options":
            agent = OptionsAgent()
        elif agent_type == "crypto" or agent_type == "trend_agent":
            agent = CryptoAgent()
        elif agent_type == "sentiment_agent":
            agent = StocksAgent()  # Use StocksAgent as a fallback for sentiment
        elif agent_type == "ensemble_agent":
            agent = StocksAgent()  # Use StocksAgent as a fallback for ensemble
        else:
            raise HTTPException(status_code=400, detail=f"Invalid agent type: {agent_type}")
        
        # Initialize risk manager if parameters are provided
        from risk.advanced_risk_manager import AdvancedRiskManager
        risk_manager = AdvancedRiskManager(
            max_drawdown=request.max_drawdown,
            trailing_stop_pct=request.trailing_stop,
            max_position_size=request.max_position_size,
            notify_discord=False  # Disable Discord notifications for backtesting
        )
        
        # Generate strategy name if not provided
        strategy_name = request.strategy_name
        if not strategy_name:
            strategy_name = f"{request.agent_type.capitalize()}Strategy_{request.timeframe}_{datetime.datetime.now().strftime('%Y%m%d')}"
        
        # Run backtest
        result = backtester.run_backtest(
            agent=agent,
            symbols=request.symbols,
            start_date=request.start_date,
            end_date=request.end_date,
            timeframe=request.timeframe,
            risk_manager=risk_manager,
            strategy_name=strategy_name
        )
        
        if result:
            # Convert result to serializable format
            serialized_result = {
                'strategy_name': result.strategy_name,
                'start_date': result.start_date.strftime('%Y-%m-%d'),
                'end_date': result.end_date.strftime('%Y-%m-%d'),
                'initial_capital': result.initial_capital,
                'metrics': result.metrics,
                'equity_curve': result.equity_curve,
                'trades': result.trades[:100]  # Limit to first 100 trades for performance
            }
            return {'results': serialized_result}
        else:
            raise HTTPException(status_code=500, detail="Backtest failed to run")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running backtest: {str(e)}")

@router.get('/risk/metrics')
def get_risk_metrics():
    """Get current risk metrics from the risk manager"""
    try:
        # Get portfolio from broker
        portfolio = broker.get_portfolio()
        
        # Get risk assessment
        risk_assessment = risk_agent.assess_risk(portfolio)
        
        return {'metrics': risk_assessment}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching risk metrics: {str(e)}")

# For UI
@router.get('/portfolio')
def get_portfolio():
    # Fetch from broker
    return {'portfolio': broker.get_portfolio()}
