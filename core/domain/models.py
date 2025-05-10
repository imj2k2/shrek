"""
Core domain models for the Shrek Trading Platform.
These models represent the fundamental business entities in the system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Union, Any
from uuid import UUID, uuid4


class PositionType(str, Enum):
    """Position types for trading."""
    LONG = "long"
    SHORT = "short"


class OrderType(str, Enum):
    """Order types for trading."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(str, Enum):
    """Order statuses."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class TimeInForce(str, Enum):
    """Time in force options for orders."""
    DAY = "day"
    GTC = "good_till_cancel"
    IOC = "immediate_or_cancel"
    FOK = "fill_or_kill"


class TradeDirection(str, Enum):
    """Trade directions."""
    BUY = "buy"
    SELL = "sell"


class StrategyType(str, Enum):
    """Strategy types for trading."""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    VALUE = "value"
    CUSTOM = "custom"


@dataclass
class Symbol:
    """Represents a tradable financial instrument."""
    ticker: str
    name: str = ""
    exchange: str = ""
    asset_class: str = "equity"
    currency: str = "USD"
    is_tradable: bool = True
    
    def __str__(self) -> str:
        return self.ticker
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ticker": self.ticker,
            "name": self.name,
            "exchange": self.exchange,
            "asset_class": self.asset_class,
            "currency": self.currency,
            "is_tradable": self.is_tradable
        }


@dataclass
class MarketData:
    """Market data for a specific symbol and time period."""
    symbol: Symbol
    timestamp: datetime
    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    close_price: Decimal
    volume: int
    additional_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol.ticker,
            "timestamp": self.timestamp.isoformat(),
            "open": float(self.open_price),
            "high": float(self.high_price),
            "low": float(self.low_price),
            "close": float(self.close_price),
            "volume": self.volume,
            **self.additional_data
        }


@dataclass
class Order:
    """Trading order."""
    id: UUID = field(default_factory=uuid4)
    symbol: Symbol = field(default_factory=lambda: Symbol(""))
    quantity: Decimal = field(default_factory=lambda: Decimal('0'))
    direction: TradeDirection = TradeDirection.BUY
    order_type: OrderType = OrderType.MARKET
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    status: OrderStatus = OrderStatus.PENDING
    position_type: PositionType = PositionType.LONG
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    filled_price: Optional[Decimal] = None
    filled_quantity: Decimal = field(default_factory=lambda: Decimal('0'))
    commission: Decimal = field(default_factory=lambda: Decimal('0'))
    strategy: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def is_filled(self) -> bool:
        """Check if the order is filled."""
        return self.status == OrderStatus.FILLED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "symbol": self.symbol.ticker,
            "quantity": float(self.quantity),
            "direction": self.direction.value,
            "order_type": self.order_type.value,
            "price": float(self.price) if self.price else None,
            "stop_price": float(self.stop_price) if self.stop_price else None,
            "time_in_force": self.time_in_force.value,
            "status": self.status.value,
            "position_type": self.position_type.value,
            "created_at": self.created_at.isoformat(),
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "filled_price": float(self.filled_price) if self.filled_price else None,
            "filled_quantity": float(self.filled_quantity),
            "commission": float(self.commission),
            "strategy": self.strategy,
            "tags": self.tags
        }


@dataclass
class Position:
    """Trading position."""
    symbol: Symbol
    quantity: Decimal
    entry_price: Decimal
    entry_date: datetime
    position_type: PositionType = PositionType.LONG
    current_price: Optional[Decimal] = None
    last_update: Optional[datetime] = None
    realized_pnl: Decimal = field(default_factory=lambda: Decimal('0'))
    strategy: Optional[str] = None
    id: UUID = field(default_factory=uuid4)
    
    @property
    def unrealized_pnl(self) -> Optional[Decimal]:
        """Calculate unrealized P&L."""
        if self.current_price is None:
            return None
        
        if self.position_type == PositionType.LONG:
            return (self.current_price - self.entry_price) * self.quantity
        else:  # SHORT position
            return (self.entry_price - self.current_price) * self.quantity
    
    @property
    def pnl_percentage(self) -> Optional[float]:
        """Calculate P&L as percentage."""
        if self.current_price is None or self.entry_price == 0:
            return None
        
        if self.position_type == PositionType.LONG:
            return float((self.current_price / self.entry_price - 1) * 100)
        else:  # SHORT position
            return float((self.entry_price / self.current_price - 1) * 100)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "symbol": self.symbol.ticker,
            "quantity": float(self.quantity),
            "entry_price": float(self.entry_price),
            "entry_date": self.entry_date.isoformat(),
            "position_type": self.position_type.value,
            "current_price": float(self.current_price) if self.current_price else None,
            "last_update": self.last_update.isoformat() if self.last_update else None,
            "realized_pnl": float(self.realized_pnl),
            "unrealized_pnl": float(self.unrealized_pnl) if self.unrealized_pnl else None,
            "pnl_percentage": self.pnl_percentage,
            "strategy": self.strategy
        }


@dataclass
class Portfolio:
    """Trading portfolio."""
    id: UUID = field(default_factory=uuid4)
    name: str = "Default Portfolio"
    cash: Decimal = field(default_factory=lambda: Decimal('100000'))
    positions: Dict[str, Position] = field(default_factory=dict)
    closed_positions: List[Position] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)
    
    @property
    def total_value(self) -> Decimal:
        """Calculate total portfolio value."""
        positions_value = sum(
            (pos.current_price or pos.entry_price) * pos.quantity
            for pos in self.positions.values()
        )
        return self.cash + positions_value
    
    @property
    def realized_pnl(self) -> Decimal:
        """Calculate total realized P&L."""
        return sum(pos.realized_pnl for pos in self.closed_positions)
    
    @property
    def unrealized_pnl(self) -> Decimal:
        """Calculate total unrealized P&L."""
        return sum(
            pos.unrealized_pnl or Decimal('0')
            for pos in self.positions.values()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "name": self.name,
            "cash": float(self.cash),
            "positions": {k: v.to_dict() for k, v in self.positions.items()},
            "closed_positions": [p.to_dict() for p in self.closed_positions],
            "created_at": self.created_at.isoformat(),
            "last_update": self.last_update.isoformat(),
            "total_value": float(self.total_value),
            "realized_pnl": float(self.realized_pnl),
            "unrealized_pnl": float(self.unrealized_pnl)
        }


@dataclass
class Signal:
    """Trading signal generated by a strategy."""
    symbol: Symbol
    timestamp: datetime
    action: TradeDirection
    price: Optional[Decimal] = None
    confidence: float = 0.0  # 0.0 to 1.0
    target_price: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    strategy: str = ""
    timeframe: str = "1d"
    position_type: PositionType = PositionType.LONG
    expiration: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: UUID = field(default_factory=uuid4)
    
    def is_valid(self) -> bool:
        """Check if the signal is still valid."""
        if self.expiration and datetime.now() > self.expiration:
            return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "symbol": self.symbol.ticker,
            "timestamp": self.timestamp.isoformat(),
            "action": self.action.value,
            "price": float(self.price) if self.price else None,
            "confidence": self.confidence,
            "target_price": float(self.target_price) if self.target_price else None,
            "stop_loss": float(self.stop_loss) if self.stop_loss else None,
            "strategy": self.strategy,
            "timeframe": self.timeframe,
            "position_type": self.position_type.value,
            "expiration": self.expiration.isoformat() if self.expiration else None,
            "metadata": self.metadata
        }


@dataclass
class BacktestResult:
    """Results of a backtest run."""
    id: UUID = field(default_factory=uuid4)
    strategy: str
    symbols: List[str]
    start_date: datetime
    end_date: datetime
    initial_capital: Decimal
    final_capital: Decimal
    total_return: float  # Percentage
    annualized_return: float  # Percentage
    sharpe_ratio: float
    max_drawdown: float  # Percentage
    trades: List[Dict[str, Any]]
    positions: Dict[str, List[Dict[str, Any]]]
    metrics: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "strategy": self.strategy,
            "symbols": self.symbols,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "initial_capital": float(self.initial_capital),
            "final_capital": float(self.final_capital),
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "trades": self.trades,
            "positions": self.positions,
            "metrics": self.metrics,
            "parameters": self.parameters,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class NewsSentiment:
    """News sentiment for a symbol."""
    symbol: Symbol
    timestamp: datetime
    title: str
    source: str
    url: str
    sentiment_score: float  # -1.0 to 1.0
    relevance_score: float  # 0.0 to 1.0
    summary: str = ""
    content: str = ""
    id: UUID = field(default_factory=uuid4)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": str(self.id),
            "symbol": self.symbol.ticker,
            "timestamp": self.timestamp.isoformat(),
            "title": self.title,
            "source": self.source,
            "url": self.url,
            "sentiment_score": self.sentiment_score,
            "relevance_score": self.relevance_score,
            "summary": self.summary,
            "content": self.content
        }
