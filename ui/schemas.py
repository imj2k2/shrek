from mcp.protocol import MCPProtocol
from agents import CoordinatorAgent, StocksAgent, OptionsAgent, CryptoAgent, RiskAgent, TradingExecutor
from trading.lumibot_wrapper import LumibotBroker
from backtest.backtest_engine import Backtester
from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import datetime
import logging
import math
import os
import json
import uuid
from pathlib import Path

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

# Define path for strategy repository
STRATEGY_REPO_PATH = Path("strategies")

# Create strategy directory if it doesn't exist
STRATEGY_REPO_PATH.mkdir(exist_ok=True)

def save_strategy_to_repository(strategy_metadata: Dict[str, Any]) -> str:
    """Save a strategy to the repository"""
    # Ensure we have a strategy ID
    if not strategy_metadata.get("id"):
        strategy_metadata["id"] = f"{strategy_metadata['name'].lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}"
    
    # Ensure the strategy directory exists
    strategy_path = STRATEGY_REPO_PATH / f"{strategy_metadata['id']}.json"
    
    # Save strategy metadata as JSON
    with open(strategy_path, "w") as f:
        json.dump(strategy_metadata, f, indent=2)
    
    return strategy_metadata["id"]

def load_strategy_from_repository(strategy_id: str) -> Dict[str, Any]:
    """Load a strategy from the repository"""
    strategy_path = STRATEGY_REPO_PATH / f"{strategy_id}.json"
    
    if not strategy_path.exists():
        raise ValueError(f"Strategy with ID {strategy_id} not found")
    
    with open(strategy_path, "r") as f:
        return json.load(f)

def list_strategies_in_repository() -> List[Dict[str, Any]]:
    """List all strategies in the repository"""
    strategies = []
    
    for strategy_file in STRATEGY_REPO_PATH.glob("*.json"):
        try:
            with open(strategy_file, "r") as f:
                strategy = json.load(f)
                # Add a minimal version of the strategy for listing
                strategies.append({
                    "id": strategy.get("id"),
                    "name": strategy.get("name"),
                    "agent_type": strategy.get("agent_type"),
                    "symbols": strategy.get("symbols"),
                    "created_at": strategy.get("created_at"),
                    "description": strategy.get("description"),
                    "performance": {
                        "total_return": strategy.get("metrics", {}).get("total_return"),
                        "sharpe_ratio": strategy.get("metrics", {}).get("sharpe_ratio"),
                        "win_rate": strategy.get("metrics", {}).get("win_rate"),
                    }
                })
        except Exception as e:
            logging.error(f"Error loading strategy {strategy_file}: {str(e)}")
    
    # Sort by creation date (newest first)
    strategies.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return strategies

# Define models for strategy management
class StrategyDeployRequest(BaseModel):
    strategy_id: str
    broker: str = "paper"  # paper, alpaca, robinhood
    initial_capital: Optional[float] = None
    description: Optional[str] = None

class StrategyListResponse(BaseModel):
    strategies: List[Dict[str, Any]]

# FastAPI endpoints
router = APIRouter()

class MarketData(BaseModel):
    data: Dict[str, Any]

class BacktestRequest(BaseModel):
    agent_type: str
    symbols: List[str]
    start_date: str
    end_date: str
    timeframe: str = "day"
    strategy_name: Optional[str] = None
    strategy_id: Optional[str] = None  # For saving/loading strategies
    description: Optional[str] = None  # Strategy description
    initial_capital: float = 100000.0
    strategy_config: Optional[Dict[str, Any]] = None
    save_strategy: bool = False  # Whether to save this strategy for live trading
    # Risk management parameters
    max_drawdown: float = 0.1  # Maximum drawdown (10%)
    trailing_stop: float = 0.05  # Trailing stop percentage (5%)
    max_position_size: float = 0.2  # Maximum position size as percentage of portfolio (20%)

@router.post('/agents/signal')
def agent_signal(market_data: MarketData):
    signals = coordinator.coordinate(market_data.data)
    return {'signals': signals}

@router.post('/agents/execute')
def execute_trade(signal: Dict[str, Any]):
    result = executor.execute(signal)
    return {'result': result}

@router.get('/strategies', response_model=StrategyListResponse)
def list_strategies():
    """List all saved strategies in the repository"""
    try:
        strategies = list_strategies_in_repository()
        return {"strategies": strategies}
    except Exception as e:
        logging.error(f"Error listing strategies: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing strategies: {str(e)}")

@router.get('/strategies/{strategy_id}')
def get_strategy(strategy_id: str):
    """Get a specific strategy from the repository"""
    try:
        strategy = load_strategy_from_repository(strategy_id)
        return strategy
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logging.error(f"Error getting strategy {strategy_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting strategy: {str(e)}")

@router.post('/strategies/deploy')
def deploy_strategy(request: StrategyDeployRequest):
    """Deploy a saved strategy for live trading"""
    try:
        # Load the strategy from repository
        strategy = load_strategy_from_repository(request.strategy_id)
        
        # TODO: Implement actual deployment logic here
        # This would involve:
        # 1. Creating the appropriate agent based on strategy.agent_type
        # 2. Configuring it using strategy.config
        # 3. Setting up the broker connection
        # 4. Starting the trading loop
        
        # For now, we'll return a success message
        return {
            "status": "success",
            "message": f"Strategy {strategy['name']} (ID: {request.strategy_id}) deployed for live trading",
            "strategy": strategy,
            "broker": request.broker,
            "deployed_at": datetime.datetime.now().isoformat()
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logging.error(f"Error deploying strategy {request.strategy_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deploying strategy: {str(e)}")

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
        elif agent_type == "customizable_agent":
            # Initialize the customizable agent with provided or default configuration
            from agents.customizable_agent import CustomizableAgent
            if request.strategy_config:
                agent = CustomizableAgent(request.strategy_config)
                logging.info(f"Using CustomizableAgent with custom configuration: {request.strategy_config}")
            else:
                agent = CustomizableAgent()
                logging.info("Using CustomizableAgent with default configuration")
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
        
        try:
            # Run backtest
            result = backtester.run_backtest(
                agent=agent,
                symbols=request.symbols,
                start_date=request.start_date,
                end_date=request.end_date,
                timeframe=request.timeframe,
                risk_manager=risk_manager,
                strategy_name=strategy_name,
                initial_capital=request.initial_capital
            )
            
            if not result:
                logging.error("Backtest returned no results")
                return {"error": "Backtest returned no results", "status": "failed"}
        except Exception as e:
            logging.error(f"Backtest execution error: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return {"error": f"Backtest execution error: {str(e)}", "status": "failed"}
        
        # Get data source information
        data_sources = {}
        for symbol in request.symbols:
            # Try to get data source from backtester's internal data
            if hasattr(backtester, 'data') and symbol in backtester.data and hasattr(backtester.data[symbol], 'data_source'):
                data_sources[symbol] = backtester.data[symbol].data_source
            else:
                data_sources[symbol] = 'unknown'
        
        # Save successful strategy for live trading if requested
        if request.save_strategy:
            # Generate a unique strategy ID if not provided
            strategy_id = request.strategy_id or f"{strategy_name.lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}"
            
            # Prepare strategy metadata
            strategy_metadata = {
                "id": strategy_id,
                "name": strategy_name,
                "agent_type": request.agent_type,
                "symbols": request.symbols,
                "timeframe": request.timeframe,
                "created_at": datetime.datetime.now().isoformat(),
                "config": request.strategy_config if request.strategy_config else {},
                "metrics": result.metrics,
                "description": request.description or f"Strategy created from backtest on {datetime.datetime.now().strftime('%Y-%m-%d')}"
            }
            
            # Save strategy to the repository
            try:
                save_strategy_to_repository(strategy_metadata)
                logging.info(f"Saved strategy {strategy_id} to repository")
            except Exception as e:
                logging.error(f"Error saving strategy: {str(e)}")
                # Continue execution even if strategy saving fails
        
        # Process metrics to handle NaN and Inf values for JSON serialization
        sanitized_metrics = {}
        if hasattr(result, 'metrics') and result.metrics is not None:
            for k, v in result.metrics.items():
                if isinstance(v, float):
                    if math.isnan(v):
                        sanitized_metrics[k] = 0.0  # Replace NaN with 0
                    elif math.isinf(v):
                        sanitized_metrics[k] = "Infinity" if v > 0 else "-Infinity"
                    else:
                        sanitized_metrics[k] = v
                else:
                    sanitized_metrics[k] = v
        
        # Defensive approach to sanitize equity curve
        sanitized_equity_curve = []
        if hasattr(result, 'equity_curve') and result.equity_curve is not None:
            for point in result.equity_curve:
                if not isinstance(point, dict):
                    continue  # Skip non-dict points
                    
                sanitized_point = {}
                for k, v in point.items():
                    if isinstance(v, float):
                        if math.isnan(v):
                            sanitized_point[k] = 0.0  # Replace NaN with 0
                        elif math.isinf(v):
                            sanitized_point[k] = 1e10 if v > 0 else -1e10  # Replace infinity with large numbers
                        else:
                            sanitized_point[k] = v
                    elif isinstance(v, (str, int, bool)) or v is None:
                        sanitized_point[k] = v
                    else:
                        # For non-primitive types, convert to string
                        sanitized_point[k] = str(v)
                sanitized_equity_curve.append(sanitized_point)
        
        # Process trades with NaN handling and extra safety
        sanitized_trades = []
        if hasattr(result, 'trades') and result.trades is not None:
            for trade in result.trades:
                if not isinstance(trade, dict):
                    continue  # Skip non-dict trades
                    
                sanitized_trade = {}
                for k, v in trade.items():
                    if isinstance(v, float):
                        if math.isnan(v):
                            sanitized_trade[k] = 0.0
                        elif math.isinf(v):
                            sanitized_trade[k] = 1e10 if v > 0 else -1e10
                        else:
                            sanitized_trade[k] = v
                    elif isinstance(v, (str, int, bool)) or v is None:
                        sanitized_trade[k] = v
                    else:
                        # For non-primitive types, convert to string
                        sanitized_trade[k] = str(v)
                sanitized_trades.append(sanitized_trade)
        
        serialized_result = {
            'strategy_name': result.strategy_name,
            'start_date': result.start_date.strftime('%Y-%m-%d'),
            'end_date': result.end_date.strftime('%Y-%m-%d'),
            'initial_capital': result.initial_capital,
            'metrics': sanitized_metrics,
            'equity_curve': sanitized_equity_curve,
            'trades': sanitized_trades,
            'data_sources': data_sources  # Add data source information
        }
        return {'results': serialized_result}
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
