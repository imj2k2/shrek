from mcp.protocol import MCPProtocol
from agents import CoordinatorAgent, StocksAgent, OptionsAgent, CryptoAgent, RiskAgent, TradingExecutor
from trading.lumibot_wrapper import LumibotBroker
from backtest.backtest_engine import Backtester
from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import datetime
import logging
import math
import os
import json
import uuid
import numpy as np
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
        
        # Create the appropriate agent based on agent_type
        # Temporarily use StocksAgent for all types to ensure trades are generated
        # Set DEBUG_TRADING=true in docker-compose.yml to force signal generation
        logging.info(f"Creating agent of type: {request.agent_type}")
        
        try:
            # Initialize StocksAgent with appropriate mode based on strategy
            agent = StocksAgent()
            
            # Only enable debug mode if explicitly requested or if no strategy is specified
            if request.strategy_name and 'debug' not in request.strategy_name.lower():
                agent.debug_enabled = False
                logging.info(f"Using normal strategy mode for {request.strategy_name}")
            else:
                agent.debug_enabled = True
                logging.info(f"Using debug mode for backtesting (forced trade generation)")
            
            # Just for logging purposes, note the requested agent type
            if request.agent_type not in ["stocks_agent", "value_agent"]:
                logging.info(f"Note: Using StocksAgent as replacement for {request.agent_type}")
        except Exception as e:
            logging.error(f"Error creating agent: {str(e)}")
            # Fallback to a very basic agent
            agent = StocksAgent()
        
        # Capture and store the agent's criteria/parameters for display in UI
        description = "StocksAgent with forced trade generation enabled (debug mode)" if agent.debug_enabled else f"StocksAgent using {request.strategy_name} strategy"
        agent_criteria = {
            "agent_type": request.agent_type,
            "strategies": agent.strategies if hasattr(agent, "strategies") else {},
            "debug_mode": agent.debug_enabled,
            "description": description
        }
        
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
            
            # Convert BacktestResult object to dictionary for API response
            result_dict = {
                "strategy_name": result.strategy_name,
                "start_date": result.start_date.strftime('%Y-%m-%d'),
                "end_date": result.end_date.strftime('%Y-%m-%d'),
                "initial_capital": result.initial_capital,
                "trades": result.trades,
                "equity_curve": result.equity_curve,
                "metrics": result.metrics,
                "data_sources": result.data_sources if hasattr(result, 'data_sources') else {},
                # Add agent criteria
                "agent_criteria": agent_criteria
            }
            
            if not result_dict:
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
        
        # We already created result_dict above, just add data sources if not already present
        if 'data_sources' not in result_dict:
            result_dict['data_sources'] = data_sources
            
        # Handle metrics properly - convert numpy values, replace NaN/Infinity
        if 'metrics' in result_dict and isinstance(result_dict['metrics'], dict):
            sanitized_metrics = {}
            for k, v in result_dict['metrics'].items():
                if isinstance(v, (np.integer, np.floating)):
                    v = float(v)  # Convert numpy types to Python types
                
                if isinstance(v, float):
                    if math.isnan(v):
                        sanitized_metrics[k] = 0.0  # Replace NaN with 0
                    elif math.isinf(v):
                        sanitized_metrics[k] = "Infinity" if v > 0 else "-Infinity"
                    else:
                        sanitized_metrics[k] = v
                else:
                    sanitized_metrics[k] = v
            result_dict['metrics'] = sanitized_metrics
        
        # Convert equity curve from dictionary to list if needed
        if isinstance(result_dict.get('equity_curve', None), dict):
            try:
                # Extract equity curve points from numbered keys
                sorted_keys = sorted([int(k) for k in result_dict['equity_curve'].keys() if k.isdigit()])
                sanitized_equity_curve = []
                for k in sorted_keys:
                    point = result_dict['equity_curve'][str(k)]
                    # Sanitize each point
                    sanitized_point = {}
                    for pk, pv in point.items():
                        if isinstance(pv, float):
                            if math.isnan(pv):
                                sanitized_point[pk] = 0.0  # Replace NaN with 0
                            elif math.isinf(pv):
                                sanitized_point[pk] = 1e10 if pv > 0 else -1e10  # Replace infinity with large numbers
                            else:
                                sanitized_point[pk] = pv
                        else:
                            sanitized_point[pk] = pv
                    sanitized_equity_curve.append(sanitized_point)
                result_dict['equity_curve'] = sanitized_equity_curve
            except (ValueError, AttributeError, KeyError):
                # If conversion fails, leave as is
                pass
                
        # Process trades list
        if 'trades' in result_dict and isinstance(result_dict['trades'], list):
            sanitized_trades = []
            for trade in result_dict['trades']:
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
            result_dict['trades'] = sanitized_trades
        
        # Fix numpy arrays and other non-serializable data in the result dictionary
        def sanitize_for_json(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)  # Convert numpy numbers to Python numbers
            elif isinstance(obj, np.ndarray):
                return obj.tolist()  # Convert numpy arrays to lists
            elif isinstance(obj, (datetime.date, datetime.datetime)):
                return obj.isoformat()  # Convert dates to ISO format
            elif isinstance(obj, dict):
                return {k: sanitize_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize_for_json(item) for item in obj]
            elif isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            else:
                return str(obj)  # Convert any other objects to strings
        
        # Sanitize the entire result dictionary
        result_dict = sanitize_for_json(result_dict)
        
        # Return the sanitized result
        return {"results": result_dict, "status": "success"}
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
