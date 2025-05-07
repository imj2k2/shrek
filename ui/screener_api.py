from fastapi import APIRouter, HTTPException
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from ui.screener_functions import run_stock_screener

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

class ScreenerRequest(BaseModel):
    universe: Optional[str] = None
    min_price: Optional[float] = 0
    max_price: Optional[float] = 10000
    min_volume: Optional[int] = 0
    min_volatility: Optional[float] = 0
    max_volatility: Optional[float] = 1.0
    min_rsi: Optional[float] = 0
    max_rsi: Optional[float] = 100
    price_above_sma50: Optional[bool] = False
    price_below_sma50: Optional[bool] = False
    price_above_sma200: Optional[bool] = False
    price_below_sma200: Optional[bool] = False
    macd_positive: Optional[bool] = False
    macd_negative: Optional[bool] = False
    pe_min: Optional[float] = 0
    pe_max: Optional[float] = 100
    eps_min: Optional[float] = 0
    eps_growth_min: Optional[float] = 0
    dividend_yield_min: Optional[float] = 0
    market_cap_min: Optional[float] = 0
    market_cap_max: Optional[float] = 0
    debt_to_equity_max: Optional[float] = 3
    profit_margin_min: Optional[float] = 0
    roe_min: Optional[float] = 0
    sort_by: Optional[str] = "Volume"
    sort_ascending: Optional[bool] = False
    max_results: Optional[int] = 50

@router.post("/screener/run")
def run_screener(request: ScreenerRequest):
    """Run stock screener with the specified parameters"""
    try:
        # Log the request for debugging
        logger.info(f"Received screener request: {request}")
        
        # Call the screener function with the request parameters
        results = run_stock_screener(
            universe=request.universe,
            min_price=request.min_price,
            max_price=request.max_price,
            min_volume=request.min_volume,
            min_volatility=request.min_volatility,
            max_volatility=request.max_volatility,
            min_rsi=request.min_rsi,
            max_rsi=request.max_rsi,
            price_above_sma50=request.price_above_sma50,
            price_below_sma50=request.price_below_sma50,
            price_above_sma200=request.price_above_sma200,
            price_below_sma200=request.price_below_sma200,
            macd_positive=request.macd_positive,
            macd_negative=request.macd_negative,
            pe_min=request.pe_min,
            pe_max=request.pe_max,
            eps_min=request.eps_min,
            eps_growth_min=request.eps_growth_min,
            dividend_yield_min=request.dividend_yield_min,
            market_cap_min=request.market_cap_min,
            market_cap_max=request.market_cap_max,
            debt_to_equity_max=request.debt_to_equity_max,
            profit_margin_min=request.profit_margin_min,
            roe_min=request.roe_min,
            sort_by=request.sort_by,
            sort_ascending=request.sort_ascending,
            max_results=request.max_results
        )
        
        # Convert results to a list of dictionaries for API response
        if isinstance(results, list):
            return {"results": results, "count": len(results)}
        else:
            # If results is a DataFrame, convert to records
            return {"results": results.to_dict(orient="records"), "count": len(results)}
            
    except Exception as e:
        logger.error(f"Error running screener: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error running screener: {str(e)}")
