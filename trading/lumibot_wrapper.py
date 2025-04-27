# Wrapper for Lumibot integration with Alpaca and Robinhood
import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from data.storage import Storage

class LumibotBroker:
    def __init__(self):
        """Initialize broker connection with Alpaca or Robinhood via Lumibot"""
        self.broker_type = os.getenv("BROKER_TYPE", "paper")  # paper, alpaca, robinhood
        self.storage = Storage()
        self.logger = logging.getLogger("LumibotBroker")
        
        # Store API credentials securely
        self.credentials = {
            "alpaca": {
                "api_key": os.getenv("ALPACA_API_KEY", ""),
                "api_secret": os.getenv("ALPACA_API_SECRET", ""),
                "paper": True  # Set to False for live trading
            },
            "robinhood": {
                "username": os.getenv("ROBINHOOD_USERNAME", ""),
                "password": os.getenv("ROBINHOOD_PASSWORD", ""),
            }
        }
        
        # Initialize connection based on broker type
        self._initialize_connection()
        
        # Mock portfolio for development/testing
        self._mock_portfolio = self._load_or_create_portfolio()
        self._transaction_history = self._load_or_create_transactions()
    
    def _initialize_connection(self):
        """Initialize the appropriate broker connection"""
        if self.broker_type == "paper":
            self.logger.info("Using paper trading mode")
            return
            
        try:
            if self.broker_type == "alpaca":
                # In production, this would use the actual Lumibot Alpaca connection
                self.logger.info("Connecting to Alpaca")
                # from lumibot.brokers import Alpaca
                # self.connection = Alpaca(self.credentials["alpaca"])
            elif self.broker_type == "robinhood":
                # In production, this would use the actual Lumibot Robinhood connection
                self.logger.info("Connecting to Robinhood")
                # from lumibot.brokers import Robinhood
                # self.connection = Robinhood(self.credentials["robinhood"])
        except Exception as e:
            self.logger.error(f"Failed to connect to broker: {e}")
            self.logger.warning("Falling back to paper trading mode")
            self.broker_type = "paper"
    
    def _load_or_create_portfolio(self) -> Dict[str, Any]:
        """Load portfolio from storage or create a new one"""
        portfolio = self.storage.get("portfolio")
        if not portfolio:
            # Initialize with mock data
            portfolio = {
                'value': 100000,
                'cash': 50000,
                'positions': [
                    {'symbol': 'AAPL', 'qty': 10, 'entry': 150, 'current': 155, 'pnl': 50},
                    {'symbol': 'TSLA', 'qty': 5, 'entry': 700, 'current': 710, 'pnl': 50}
                ],
                'history': [
                    {'date': (datetime.now() - timedelta(days=30)).isoformat(), 'portfolio_value': 95000},
                    {'date': (datetime.now() - timedelta(days=20)).isoformat(), 'portfolio_value': 97000},
                    {'date': (datetime.now() - timedelta(days=10)).isoformat(), 'portfolio_value': 99000},
                    {'date': datetime.now().isoformat(), 'portfolio_value': 100000}
                ],
                'drawdown': 0.03,
                'peak_value': 100000
            }
            self.storage.set("portfolio", portfolio)
        return portfolio
    
    def _load_or_create_transactions(self) -> List[Dict[str, Any]]:
        """Load transaction history from storage or create a new one"""
        transactions = self.storage.get("transactions")
        if not transactions:
            # Initialize with mock data
            transactions = [
                {
                    'id': '1',
                    'symbol': 'AAPL',
                    'action': 'buy',
                    'qty': 10,
                    'price': 150,
                    'timestamp': (datetime.now() - timedelta(days=15)).isoformat(),
                    'status': 'filled'
                },
                {
                    'id': '2',
                    'symbol': 'TSLA',
                    'action': 'buy',
                    'qty': 5,
                    'price': 700,
                    'timestamp': (datetime.now() - timedelta(days=5)).isoformat(),
                    'status': 'filled'
                }
            ]
            self.storage.set("transactions", transactions)
        return transactions
    
    def place_order(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Place an order with the broker based on the signal"""
        self.logger.info(f"Placing order: {signal}")
        
        # Validate signal
        required_fields = ['symbol', 'action', 'qty']
        for field in required_fields:
            if field not in signal:
                error_msg = f"Missing required field: {field}"
                self.logger.error(error_msg)
                return {'status': 'error', 'message': error_msg}
        
        # In production mode, send to actual broker
        if self.broker_type != "paper":
            try:
                # This would use the actual Lumibot API
                # result = self.connection.submit_order(
                #     symbol=signal['symbol'],
                #     order_type='market',
                #     qty=signal['qty'],
                #     side=signal['action']
                # )
                # return {'status': 'submitted', 'order_id': result.id}
                pass
        
        # For paper trading, update the mock portfolio
        symbol = signal['symbol']
        action = signal['action'].lower()
        qty = signal['qty']
        
        # Get current price (mock)
        # In production, this would fetch the actual price
        current_price = 0
        for position in self._mock_portfolio['positions']:
            if position['symbol'] == symbol:
                current_price = position['current']
                break
        if current_price == 0:
            # Mock price if not found
            if symbol == 'AAPL':
                current_price = 155
            elif symbol == 'TSLA':
                current_price = 710
            else:
                current_price = 100
        
        # Create transaction record
        transaction = {
            'id': str(len(self._transaction_history) + 1),
            'symbol': symbol,
            'action': action,
            'qty': qty,
            'price': current_price,
            'timestamp': datetime.now().isoformat(),
            'status': 'filled'
        }
        
        # Update mock portfolio
        if action == 'buy':
            # Deduct cash
            cost = qty * current_price
            if cost > self._mock_portfolio['cash']:
                return {'status': 'error', 'message': 'Insufficient funds'}
            
            self._mock_portfolio['cash'] -= cost
            
            # Add to position or create new one
            position_exists = False
            for position in self._mock_portfolio['positions']:
                if position['symbol'] == symbol:
                    # Update existing position
                    total_qty = position['qty'] + qty
                    total_cost = (position['entry'] * position['qty']) + (current_price * qty)
                    position['entry'] = total_cost / total_qty
                    position['qty'] = total_qty
                    position['current'] = current_price
                    position['pnl'] = (current_price - position['entry']) * position['qty']
                    position_exists = True
                    break
            
            if not position_exists:
                # Create new position
                self._mock_portfolio['positions'].append({
                    'symbol': symbol,
                    'qty': qty,
                    'entry': current_price,
                    'current': current_price,
                    'pnl': 0
                })
        
        elif action == 'sell':
            # Find position
            position_index = None
            for i, position in enumerate(self._mock_portfolio['positions']):
                if position['symbol'] == symbol:
                    position_index = i
                    break
            
            if position_index is None:
                return {'status': 'error', 'message': f'No position found for {symbol}'}
            
            position = self._mock_portfolio['positions'][position_index]
            
            if qty > position['qty']:
                return {'status': 'error', 'message': f'Insufficient shares: have {position["qty"]}, trying to sell {qty}'}
            
            # Add to cash
            proceeds = qty * current_price
            self._mock_portfolio['cash'] += proceeds
            
            # Update position
            if qty == position['qty']:
                # Remove position entirely
                self._mock_portfolio['positions'].pop(position_index)
            else:
                # Reduce position
                position['qty'] -= qty
                position['pnl'] = (current_price - position['entry']) * position['qty']
        
        # Update portfolio value
        self._update_portfolio_value()
        
        # Save transaction
        self._transaction_history.append(transaction)
        self.storage.set("transactions", self._transaction_history)
        
        # Save updated portfolio
        self.storage.set("portfolio", self._mock_portfolio)
        
        return {'status': 'filled', 'transaction': transaction}
    
    def _update_portfolio_value(self):
        """Update the total portfolio value"""
        positions_value = sum(p['qty'] * p['current'] for p in self._mock_portfolio['positions'])
        new_value = self._mock_portfolio['cash'] + positions_value
        
        # Update portfolio value
        self._mock_portfolio['value'] = new_value
        
        # Update peak value if needed
        if new_value > self._mock_portfolio.get('peak_value', 0):
            self._mock_portfolio['peak_value'] = new_value
        
        # Calculate drawdown
        peak = self._mock_portfolio.get('peak_value', new_value)
        if peak > 0:
            self._mock_portfolio['drawdown'] = (peak - new_value) / peak
        
        # Add to history
        if 'history' not in self._mock_portfolio:
            self._mock_portfolio['history'] = []
        
        self._mock_portfolio['history'].append({
            'date': datetime.now().isoformat(),
            'portfolio_value': new_value
        })
        
        # Keep only the last 100 history points
        if len(self._mock_portfolio['history']) > 100:
            self._mock_portfolio['history'] = self._mock_portfolio['history'][-100:]
    
    def get_portfolio(self) -> Dict[str, Any]:
        """Get the current portfolio state"""
        if self.broker_type != "paper":
            try:
                # In production, this would fetch from the actual broker
                # return self.connection.get_portfolio()
                pass
        
        # For paper trading or fallback, return the mock portfolio
        return self._mock_portfolio
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions"""
        return self._mock_portfolio['positions']
    
    def get_transactions(self, symbol: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        """Get transaction history, optionally filtered by symbol"""
        if symbol:
            filtered = [t for t in self._transaction_history if t['symbol'] == symbol]
            return filtered[-limit:]
        return self._transaction_history[-limit:]
    
    def get_cash(self) -> float:
        """Get available cash"""
        return self._mock_portfolio['cash']
    
    def get_buying_power(self) -> float:
        """Get buying power (may include margin if enabled)"""
        # For simplicity, buying power equals cash in this implementation
        return self._mock_portfolio['cash']
