from mcp.protocol import MCPProtocol
from agents import CoordinatorAgent, StocksAgent, OptionsAgent, CryptoAgent, RiskAgent, TradingExecutor
from trading.lumibot_wrapper import LumibotBroker
from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
from typing import Dict, Any

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

# FastAPI endpoints
router = APIRouter()

class MarketData(BaseModel):
    data: Dict[str, Any]

@router.post('/agents/signal')
def agent_signal(market_data: MarketData):
    signals = coordinator.coordinate(market_data.data)
    return {'signals': signals}

@router.post('/agents/execute')
def execute_trade(signal: Dict[str, Any]):
    result = executor.execute(signal)
    return {'result': result}

# For UI
@router.get('/portfolio')
def get_portfolio():
    # Placeholder: fetch from broker or DB
    return {'portfolio': broker.get_portfolio()}
