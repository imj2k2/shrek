from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import asyncio
import threading
import os

# Try to import Discord notification, but provide fallback if not available
try:
    from discord_bot.bot import notify_risk_alert
    DISCORD_AVAILABLE = True
except (ImportError, ValueError):
    DISCORD_AVAILABLE = False
    
    # Dummy function when Discord is not available
    async def notify_risk_alert(message, alert_type="unknown"):
        logging.info(f"[DISCORD DISABLED] Risk alert ({alert_type}): {message}")
        return True

class AdvancedRiskManager:
    """Advanced risk management with trailing stops, drawdown limits, and position sizing"""
    
    def __init__(self, 
                 max_drawdown: float = 0.1, 
                 max_position_size: float = 0.2,
                 trailing_stop_pct: float = 0.05,
                 notify_discord: bool = True):
        """
        Initialize risk manager with risk parameters
        
        Args:
            max_drawdown: Maximum allowable drawdown as a decimal (0.1 = 10%)
            max_position_size: Maximum position size as a decimal of portfolio (0.2 = 20%)
            trailing_stop_pct: Trailing stop percentage (0.05 = 5%)
            notify_discord: Whether to send alerts to Discord
        """
        self.max_drawdown = max_drawdown
        self.max_position_size = max_position_size
        self.trailing_stop_pct = trailing_stop_pct
        self.notify_discord = notify_discord
        
        # Peak tracking for drawdown calculation
        self.peak_equity = None
        self.peak_date = None
        
        # Trailing stop tracking
        self.position_peaks = {}  # {symbol: {peak_price: float, entry_price: float, qty: int}}
        
        # Trading status
        self.trading_halted = False
        self.halt_reason = None
        self.halt_timestamp = None
        
        # Logging
        self.logger = logging.getLogger("AdvancedRiskManager")
        
        # Alert history
        self.alert_history = []
    
    def assess(self, portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """Assess portfolio risk and generate alerts
        
        Args:
            portfolio: Portfolio data including equity, positions, etc.
            
        Returns:
            Dict with alerts and risk metrics
        """
        if not portfolio:
            return {'error': 'Invalid portfolio data'}
        
        equity = portfolio.get('value', 0)
        positions = portfolio.get('positions', [])
        timestamp = datetime.now()
        
        alerts = []
        risk_metrics = {
            'drawdown': 0,
            'largest_position_pct': 0,
            'trailing_stops': [],
            'trading_status': 'active' if not self.trading_halted else 'halted'
        }
        
        # 1. Assess drawdown
        drawdown_alert = self._check_drawdown(equity, timestamp)
        if drawdown_alert:
            alerts.append(drawdown_alert)
        
        # 2. Check position sizes
        position_alerts = self._check_position_sizes(positions, equity)
        alerts.extend(position_alerts)
        
        # 3. Update and check trailing stops
        trailing_stop_alerts = self._check_trailing_stops(positions)
        alerts.extend(trailing_stop_alerts)
        
        # 4. Calculate risk metrics
        risk_metrics['drawdown'] = self._calculate_drawdown(equity)
        risk_metrics['largest_position_pct'] = self._calculate_largest_position(positions, equity)
        risk_metrics['trailing_stops'] = self._get_trailing_stop_levels()
        
        # 5. Process alerts
        if alerts and self.notify_discord:
            self._send_discord_alerts(alerts)
        
        # Store alerts in history
        for alert in alerts:
            alert['timestamp'] = timestamp.isoformat()
            self.alert_history.append(alert)
        
        # Limit history size
        if len(self.alert_history) > 100:
            self.alert_history = self.alert_history[-100:]
        
        return {
            'alerts': alerts,
            'risk_metrics': risk_metrics,
            'trading_halted': self.trading_halted,
            'halt_reason': self.halt_reason
        }
    
    def _check_drawdown(self, equity: float, timestamp: datetime) -> Optional[Dict[str, Any]]:
        """Check if current drawdown exceeds maximum allowed"""
        # Update peak if this is a new high
        if self.peak_equity is None or equity > self.peak_equity:
            self.peak_equity = equity
            self.peak_date = timestamp
        
        # Calculate drawdown
        drawdown = self._calculate_drawdown(equity)
        
        # Check if drawdown exceeds maximum
        if drawdown > self.max_drawdown:
            # Halt trading if not already halted
            if not self.trading_halted:
                self.trading_halted = True
                self.halt_reason = f"Max drawdown breached: {drawdown:.2%} > {self.max_drawdown:.2%}"
                self.halt_timestamp = timestamp
                
                self.logger.warning(self.halt_reason)
                
                return {
                    'type': 'halt_trading',
                    'reason': 'max_drawdown',
                    'current_drawdown': drawdown,
                    'max_drawdown': self.max_drawdown,
                    'peak_equity': self.peak_equity,
                    'current_equity': equity,
                    'days_since_peak': (timestamp - self.peak_date).days if self.peak_date else 0
                }
        
        return None
    
    def _check_position_sizes(self, positions, equity: float) -> List[Dict[str, Any]]:
        """Check if any positions exceed maximum size"""
        alerts = []
        
        # Handle case where positions is a dictionary instead of a list
        if isinstance(positions, dict):
            # Convert dictionary of positions to list for processing
            positions_list = []
            for symbol, pos_data in positions.items():
                if isinstance(pos_data, dict):
                    pos_data['symbol'] = symbol
                    positions_list.append(pos_data)
            positions = positions_list
        
        # Handle case where positions is not a list or dict
        if not positions:
            self.logger.warning(f"Empty positions data")
            return alerts
        elif not isinstance(positions, list):
            self.logger.warning(f"Invalid positions format: {type(positions)}")
            return alerts
            
        for position in positions:
            # Handle case where position is not a dictionary
            if not isinstance(position, dict):
                self.logger.warning(f"Invalid position format: {type(position)} - {position}")
                continue
                
            symbol = position.get('symbol')
            qty = position.get('qty', 0)
            current_price = position.get('current', 0)
            
            if not all([symbol, qty, current_price]):
                continue
            
            position_value = qty * current_price
            position_pct = position_value / equity if equity > 0 else 0
            
            if position_pct > self.max_position_size:
                # Calculate how much to reduce
                target_value = equity * self.max_position_size
                reduce_qty = int((position_value - target_value) / current_price)
                reduce_qty = max(1, min(reduce_qty, qty - 1))  # Ensure we don't reduce to zero or negative
                
                alert = {
                    'type': 'reduce_position',
                    'symbol': symbol,
                    'current_pct': position_pct,
                    'target_pct': self.max_position_size,
                    'current_qty': qty,
                    'reduce_qty': reduce_qty,
                    'reason': 'position_size_exceeded'
                }
                
                alerts.append(alert)
                self.logger.warning(
                    f"Position size limit breached for {symbol}: {position_pct:.2%} > {self.max_position_size:.2%}. "
                    f"Recommend reducing by {reduce_qty} shares."
                )
        
        return alerts
    
    def _check_trailing_stops(self, positions) -> List[Dict[str, Any]]:
        """Update and check trailing stops for all positions"""
        alerts = []
        
        # Handle case where positions is a dictionary instead of a list
        if isinstance(positions, dict):
            # Convert dictionary of positions to list for processing
            positions_list = []
            for symbol, pos_data in positions.items():
                if isinstance(pos_data, dict):
                    pos_data['symbol'] = symbol
                    positions_list.append(pos_data)
            positions = positions_list
        
        # Handle case where positions is not a list or dict
        if not positions:
            self.logger.warning(f"Empty positions data in trailing stops")
            return alerts
        elif not isinstance(positions, list):
            self.logger.warning(f"Invalid positions format in trailing stops: {type(positions)}")
            return alerts
            
        for position in positions:
            # Handle case where position is not a dictionary
            if not isinstance(position, dict):
                self.logger.warning(f"Invalid position format in trailing stops: {type(position)} - {position}")
                continue
                
            symbol = position.get('symbol')
            qty = position.get('qty', 0)
            current_price = position.get('current', 0)
            entry_price = position.get('entry', 0)
            
            if not all([symbol, qty, current_price, entry_price]):
                continue
            
            # Initialize or update peak price for this position
            if symbol not in self.position_peaks:
                self.position_peaks[symbol] = {
                    'peak_price': current_price,
                    'entry_price': entry_price,
                    'qty': qty
                }
            else:
                # Update peak if current price is higher
                if current_price > self.position_peaks[symbol]['peak_price']:
                    self.position_peaks[symbol]['peak_price'] = current_price
                # Update entry and qty in case they changed
                self.position_peaks[symbol]['entry_price'] = entry_price
                self.position_peaks[symbol]['qty'] = qty
            
            # Calculate trailing stop level
            peak_price = self.position_peaks[symbol]['peak_price']
            stop_level = peak_price * (1 - self.trailing_stop_pct)
            
            # Check if current price is below trailing stop
            if current_price < stop_level:
                alert = {
                    'type': 'trailing_stop_triggered',
                    'symbol': symbol,
                    'current_price': current_price,
                    'peak_price': peak_price,
                    'stop_level': stop_level,
                    'qty': qty,
                    'action': 'sell',
                    'reason': 'trailing_stop'
                }
                
                alerts.append(alert)
                self.logger.warning(
                    f"Trailing stop triggered for {symbol}: Current {current_price:.2f} < Stop {stop_level:.2f} "
                    f"(Peak: {peak_price:.2f})"
                )
        
        # Clean up position_peaks for positions we no longer hold
        current_symbols = {p['symbol'] for p in positions}
        symbols_to_remove = set(self.position_peaks.keys()) - current_symbols
        for symbol in symbols_to_remove:
            del self.position_peaks[symbol]
        
        return alerts
    
    def _calculate_drawdown(self, equity: float) -> float:
        """Calculate current drawdown as a percentage"""
        if not self.peak_equity or self.peak_equity <= 0:
            return 0
        
        return (self.peak_equity - equity) / self.peak_equity
    
    def _calculate_largest_position(self, positions: List[Dict[str, Any]], equity: float) -> float:
        """Calculate largest position as percentage of portfolio"""
        if not positions or equity <= 0:
            return 0
        
        position_sizes = [(p['qty'] * p['current']) / equity for p in positions if 'qty' in p and 'current' in p]
        return max(position_sizes) if position_sizes else 0
    
    def _get_trailing_stop_levels(self) -> List[Dict[str, Any]]:
        """Get current trailing stop levels for all positions"""
        stop_levels = []
        
        for symbol, data in self.position_peaks.items():
            peak_price = data['peak_price']
            stop_level = peak_price * (1 - self.trailing_stop_pct)
            
            stop_levels.append({
                'symbol': symbol,
                'peak_price': peak_price,
                'stop_level': stop_level,
                'qty': data['qty']
            })
        
        return stop_levels
    
    def _send_discord_alerts(self, alerts: List[Dict[str, Any]]):
        """Send alerts to Discord"""
        if not alerts:
            return
        
        # Skip if Discord notifications are disabled or unavailable
        if not self.notify_discord or not DISCORD_AVAILABLE:
            self.logger.info(f"Discord alerts disabled or unavailable. Skipping {len(alerts)} alerts.")
            return
        
        # Format alerts for Discord
        for alert in alerts:
            alert_type = alert.get('type', 'unknown')
            message = self._format_alert_for_discord(alert)
            
            # Use threading to avoid blocking
            threading.Thread(target=self._send_discord_message, args=(message, alert_type)).start()
    
    def _send_discord_message(self, message: str, alert_type: str):
        """Send message to Discord using asyncio"""
        # Skip if Discord is not available
        if not DISCORD_AVAILABLE:
            self.logger.info(f"Discord not available. Would have sent alert: {alert_type} - {message}")
            return
            
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(notify_risk_alert(message, alert_type))
            loop.close()
        except Exception as e:
            self.logger.error(f"Failed to send Discord alert: {str(e)}")
    
    def _format_alert_for_discord(self, alert: Dict[str, Any]) -> str:
        """Format alert data for Discord message"""
        alert_type = alert.get('type', 'unknown')
        
        if alert_type == 'halt_trading':
            return f"🚨 **TRADING HALTED** 🚨\n" \
                   f"Reason: {alert.get('reason', 'Unknown')}\n" \
                   f"Drawdown: {alert.get('current_drawdown', 0):.2%} (Max: {alert.get('max_drawdown', 0):.2%})\n" \
                   f"Current Equity: ${alert.get('current_equity', 0):,.2f} (Peak: ${alert.get('peak_equity', 0):,.2f})"
        
        elif alert_type == 'reduce_position':
            return f"⚠️ **POSITION SIZE ALERT** ⚠️\n" \
                   f"Symbol: {alert.get('symbol', 'Unknown')}\n" \
                   f"Current Size: {alert.get('current_pct', 0):.2%} (Max: {alert.get('target_pct', 0):.2%})\n" \
                   f"Recommendation: Reduce by {alert.get('reduce_qty', 0)} shares"
        
        elif alert_type == 'trailing_stop_triggered':
            return f"🔻 **TRAILING STOP TRIGGERED** 🔻\n" \
                   f"Symbol: {alert.get('symbol', 'Unknown')}\n" \
                   f"Current: ${alert.get('current_price', 0):.2f} < Stop: ${alert.get('stop_level', 0):.2f}\n" \
                   f"Peak: ${alert.get('peak_price', 0):.2f}\n" \
                   f"Action: Sell {alert.get('qty', 0)} shares"
        
        else:
            return f"⚠️ **RISK ALERT** ⚠️\n{str(alert)}"
    
    def reset_trading_halt(self):
        """Reset trading halt status (manual override)"""
        if self.trading_halted:
            self.trading_halted = False
            self.halt_reason = None
            self.halt_timestamp = None
            self.logger.info("Trading halt manually reset")
            
            if self.notify_discord and DISCORD_AVAILABLE:
                threading.Thread(
                    target=self._send_discord_message,
                    args=("✅ **TRADING HALT LIFTED** ✅\nTrading has been manually re-enabled.", "halt_reset")
                ).start()
    
    def get_alert_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent alert history"""
        return self.alert_history[-limit:] if self.alert_history else []
