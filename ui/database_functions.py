"""
Functions for database management in the UI.
"""
import gradio as gr
import pandas as pd
import logging
import sys
import os
import json
import requests
from datetime import datetime
import time

# Add parent directory to path to import from data module
sys.path.append('/app')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API base URL
API_BASE_URL = os.environ.get("API_BASE_URL", "http://backend:8000")

def refresh_database(data_source, symbols_input, days, reset_db):
    """
    Refresh the database with data from the specified source
    
    Args:
        data_source: The data source to use ('polygon_s3', 'polygon_api', or 'yahoo')
        symbols_input: Comma-separated string of symbols to refresh
        days: Number of days of historical data to sync
        reset_db: Whether to reset the database before refreshing
        
    Returns:
        Tuple of (status_message, refresh_logs)
    """
    try:
        # Parse symbols
        symbols = None
        if symbols_input and symbols_input.strip():
            symbols = [s.strip() for s in symbols_input.split(',')]
        
        # Prepare request payload
        payload = {
            "data_source": data_source,
            "symbols": symbols,
            "days": int(days),
            "reset_db": bool(reset_db)
        }
        
        logger.info(f"Refreshing database with payload: {payload}")
        
        # Show initial status
        status_message = f"Refreshing database from {data_source}..."
        
        # Make API request to backend
        response = requests.post(
            f"{API_BASE_URL}/database/refresh",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        # Check if request was successful
        if response.status_code == 200:
            result = response.json()
            
            # Format status message
            if result.get("status") == "success":
                status_message = result.get("message", "Database refreshed successfully")
            else:
                status_message = f"Error: {result.get('message', 'Unknown error')}"
            
            # Return the refresh details as JSON
            return status_message, result.get("details", {})
        else:
            error_msg = f"Error: API returned status {response.status_code}"
            try:
                error_details = response.json().get("detail", "Unknown error")
                error_msg = f"Error: {error_details}"
            except:
                pass
            
            logger.error(error_msg)
            return error_msg, {"status": "error", "message": error_msg}
    
    except Exception as e:
        error_msg = f"Error refreshing database: {str(e)}"
        logger.error(error_msg)
        return error_msg, {"status": "error", "message": error_msg}

def get_database_status():
    """
    Get the current status of the database
    
    Returns:
        String with database status information
    """
    try:
        # Make API request to backend
        response = requests.get(f"{API_BASE_URL}/database/status")
        
        # Check if request was successful
        if response.status_code == 200:
            result = response.json()
            
            # Format status message
            status_text = "Database Status:\n"
            
            if result.get("status") == "success":
                status_text += f"Status: {result.get('status')} - {result.get('message')}\n\n"
                
                # Add table information
                details = result.get("details", {})
                tables = details.get("tables", {})
                if tables:
                    status_text += "Tables:\n"
                    for table, count in tables.items():
                        status_text += f"- {table}: {count} records\n"
                    status_text += "\n"
                
                # Add last sync dates
                last_sync_dates = details.get("last_sync_dates", {})
                if last_sync_dates:
                    status_text += "Last Sync Dates:\n"
                    for source, date in last_sync_dates.items():
                        status_text += f"- {source}: {date}\n"
            else:
                status_text += f"Error: {result.get('message', 'Unknown error')}"
            
            return status_text
        else:
            error_msg = f"Error: API returned status {response.status_code}"
            try:
                error_details = response.json().get("detail", "Unknown error")
                error_msg = f"Error: {error_details}"
            except:
                pass
            
            logger.error(error_msg)
            return error_msg
    
    except Exception as e:
        error_msg = f"Error getting database status: {str(e)}"
        logger.error(error_msg)
        return error_msg
