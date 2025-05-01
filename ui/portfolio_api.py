"""
API endpoint for portfolio backtesting
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import logging

from backtest.portfolio_backtest import PortfolioBacktester
from data.data_fetcher import DataFetcher

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize components
data_fetcher = DataFetcher()
portfolio_backtester = PortfolioBacktester(data_fetcher=data_fetcher)

class PortfolioBacktestRequest(BaseModel):
    """Request model for portfolio backtest"""
    allocations: Dict[str, float]
    symbol_strategies: Optional[Dict[str, str]] = None
    strategies: Optional[Dict[str, Any]] = None
    start_date: str
    end_date: str
    generate_insights: Optional[bool] = False

@router.post("/portfolio/backtest")
async def run_portfolio_backtest(request: PortfolioBacktestRequest):
    """Run a portfolio backtest with the given configuration"""
    try:
        logger.info(f"Running portfolio backtest with allocations: {request.allocations}")
        
        # Prepare portfolio config
        portfolio_config = {
            "allocations": request.allocations,
            "symbol_strategies": request.symbol_strategies or {},
            "strategies": request.strategies or {}
        }
        
        # Run the backtest
        result = portfolio_backtester.run_portfolio_backtest(
            portfolio_config=portfolio_config,
            start_date=request.start_date,
            end_date=request.end_date,
            generate_insights=request.generate_insights
        )
        
        return {"results": result}
    
    except Exception as e:
        logger.error(f"Error in portfolio backtest: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error running portfolio backtest: {str(e)}")
