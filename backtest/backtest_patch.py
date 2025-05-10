"""
Patch file to fix market_value KeyError in backtesting engine
"""
import logging
from typing import Dict, Any

logger = logging.getLogger("BacktestPatch")

def patch_position_data(position: Dict[str, Any], symbol: str) -> Dict[str, Any]:
    """
    Ensure position dictionary has all required fields initialized
    """
    # List of required fields with default values
    required_fields = {
        'quantity': 0,
        'entry_price': 0,
        'current_price': 0,
        'market_value': 0,
        'cost_basis': 0,
        'profit_loss': 0,
        'stop_loss': None,
        'take_profit': None
    }
    
    # Initialize missing fields with defaults
    for field, default_value in required_fields.items():
        if field not in position:
            logger.warning(f"Added missing field '{field}' to position data for {symbol}")
            position[field] = default_value
            
    # Recalculate market_value if we have quantity and current_price
    if position.get('quantity', 0) > 0 and 'current_price' in position:
        position['market_value'] = position['quantity'] * position['current_price']
        
    return position

# Monkey patch the Backtester._calculate_equity method
def safe_calculate_equity(self, portfolio: Dict[str, Any], data: Dict[str, Any], date) -> float:
    """Safely calculate total portfolio equity"""
    equity = portfolio['cash']
    
    for symbol, position in portfolio['positions'].items():
        # Ensure position has all required fields
        patch_position_data(position, symbol)
        
        # Now safely use market_value
        equity += position.get('market_value', 0)
    
    return equity

# This function will be called at startup to apply the monkey patch
def apply_patches():
    """Apply all patches to fix backtesting issues"""
    from backtest.backtest_engine import Backtester
    logger.info("Applying backtesting engine patches...")
    
    # Store original method for reference
    original_method = Backtester._calculate_equity
    
    # Apply the monkey patch
    Backtester._calculate_equity = safe_calculate_equity
    
    logger.info("Backtesting engine patches applied successfully")
    return True
