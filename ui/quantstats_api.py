"""
FastAPI router for QuantStats metrics
"""
from fastapi import APIRouter, HTTPException
from typing import Optional, Dict, Any
import logging
import os

from backtest.quantstats_integration import QuantStatsAnalyzer

# Create router
router = APIRouter()

# Initialize QuantStats analyzer
analyzer = QuantStatsAnalyzer()

@router.get("/metrics/{backtest_id}")
def get_metrics(backtest_id: str):
    """Get metrics for a specific backtest"""
    try:
        metrics = analyzer.load_metrics(backtest_id)
        # Return the metrics even if it contains an error key
        # This allows the frontend to handle and display the error gracefully
        return metrics
    except Exception as e:
        logging.error(f"Error getting metrics for backtest {backtest_id}: {str(e)}")
        return {"error": "Server error", "message": str(e)}

@router.get("/returns/{backtest_id}")
def get_returns(backtest_id: str, freq: Optional[str] = None):
    """
    Get returns for a specific backtest
    
    Args:
        backtest_id: ID of the backtest
        freq: Optional frequency for resampling ('D', 'W', 'M', 'Q', 'Y')
    """
    try:
        returns = analyzer.load_returns(backtest_id, freq)
        # Return the data even if it contains an error key
        # This allows the frontend to handle and display errors gracefully
        return returns
    except Exception as e:
        logging.error(f"Error getting returns for backtest {backtest_id}: {str(e)}")
        return {"error": "Server error", "message": str(e)}

@router.get("/html/{backtest_id}")
def get_html_path(backtest_id: str):
    """Get the path to the HTML tearsheet for a specific backtest"""
    html_path = os.path.join(analyzer.output_dir, backtest_id, "tear_sheet.html")
    if not os.path.exists(html_path):
        # Return error in response rather than exception
        return {"error": "Not found", "message": "HTML tearsheet not found for this backtest"}
    return {"html_path": html_path}

@router.post("/process_backtest")
def process_backtest(data: Dict[str, Any]):
    """
    Process backtest results to generate QuantStats metrics
    
    Args:
        data: Dictionary containing:
            - backtest_id: Unique identifier for the backtest
            - equity_curve: List of dictionaries with date and equity values
            - benchmark_data: Optional benchmark data for comparison
    """
    try:
        backtest_id = data.get("backtest_id")
        equity_curve = data.get("equity_curve")
        benchmark_data = data.get("benchmark_data")
        
        if not backtest_id or not equity_curve:
            return {
                "error": "Missing parameters",
                "message": "Missing required parameters: backtest_id and equity_curve"
            }
        
        result = analyzer.process_backtest_results(backtest_id, equity_curve, benchmark_data)
        return result
    except Exception as e:
        logging.error(f"Error processing backtest: {str(e)}")
        return {"error": "Processing error", "message": str(e)}
