"""
API endpoints for database operations.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import sys

# Add parent directory to path to import from data module
sys.path.append('/app')

# Import data modules
from data.startup import refresh_database
from data.database import get_market_db

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/database")

class RefreshDatabaseRequest(BaseModel):
    """Request model for database refresh"""
    data_source: str = "polygon_s3"  # 'polygon_s3', 'polygon_api', or 'yahoo'
    symbols: Optional[List[str]] = None
    days: int = 7
    reset_db: bool = False

class DatabaseStatusResponse(BaseModel):
    """Response model for database status"""
    status: str
    message: str
    details: Optional[Dict[str, Any]] = None

@router.post("/refresh", response_model=DatabaseStatusResponse)
async def refresh_db(request: RefreshDatabaseRequest):
    """Refresh the database with data from the specified source"""
    try:
        # Validate data source
        valid_sources = ["polygon_s3", "polygon_api", "yahoo"]
        if request.data_source not in valid_sources:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid data source. Must be one of: {', '.join(valid_sources)}"
            )
        
        # Call the refresh function
        result = refresh_database(
            data_source=request.data_source,
            symbols=request.symbols,
            days=request.days,
            reset_db=request.reset_db
        )
        
        # Check for errors
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("message", "Unknown error"))
        
        return {
            "status": "success",
            "message": result.get("message", "Database refreshed successfully"),
            "details": result
        }
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error refreshing database: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error refreshing database: {str(e)}")

@router.get("/status", response_model=DatabaseStatusResponse)
async def get_db_status():
    """Get the current status of the database"""
    try:
        db = get_market_db()
        
        # Get table counts
        conn = db._get_connection()
        cursor = conn.cursor()
        
        # Get list of tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        # Get count of records in each table
        table_counts = {}
        for table in tables:
            table_name = table[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            table_counts[table_name] = count
        
        # Get last sync dates
        last_sync_dates = {}
        cursor.execute("SELECT source, data_type, end_date FROM data_sync_log WHERE status = 'success' ORDER BY end_date DESC LIMIT 10")
        sync_logs = cursor.fetchall()
        
        for log in sync_logs:
            source, data_type, end_date = log
            key = f"{source}_{data_type}"
            if key not in last_sync_dates:
                last_sync_dates[key] = end_date
        
        return {
            "status": "success",
            "message": "Database is operational",
            "details": {
                "tables": table_counts,
                "last_sync_dates": last_sync_dates
            }
        }
    
    except Exception as e:
        logger.error(f"Error getting database status: {str(e)}")
        return {
            "status": "error",
            "message": f"Error getting database status: {str(e)}",
            "details": None
        }
