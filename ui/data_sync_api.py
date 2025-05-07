"""
API endpoints for data synchronization operations.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import sys

# Add parent directory to path to import from data module
sys.path.append('/app')

# Import data modules
from data.data_sync import get_data_synchronizer, run_manual_sync

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/data")

class DataSyncRequest(BaseModel):
    """Request model for data synchronization"""
    source: str = "polygon_s3"  # 'polygon_s3', 'polygon_api', or 'yahoo'
    data_type: str = "prices"  # 'prices' or 'fundamentals'
    symbols: Optional[List[str]] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class DataSyncResponse(BaseModel):
    """Response model for data synchronization"""
    status: str
    message: str
    details: Optional[Dict[str, Any]] = None

@router.post("/sync", response_model=DataSyncResponse)
async def sync_data(request: DataSyncRequest):
    """Synchronize data from the specified source"""
    try:
        # Validate data source
        valid_sources = ["polygon_s3", "polygon_api", "yahoo"]
        if request.source not in valid_sources:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid data source. Must be one of: {', '.join(valid_sources)}"
            )
        
        # Validate data type
        valid_types = ["prices", "fundamentals"]
        if request.data_type not in valid_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid data type. Must be one of: {', '.join(valid_types)}"
            )
        
        # Call the sync function
        result = run_manual_sync(
            source=request.source,
            data_type=request.data_type,
            symbols=request.symbols,
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        # Check for errors
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("message", "Unknown error"))
        
        return {
            "status": "success",
            "message": f"Data sync from {request.source} completed successfully",
            "details": result
        }
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error syncing data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error syncing data: {str(e)}")
