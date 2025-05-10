"""
Repository interfaces for the Shrek Trading Platform.
These interfaces define the contract for data access across the application.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from uuid import UUID

from core.domain.models import (
    Symbol, MarketData, Order, Position, Portfolio, 
    Signal, BacktestResult, NewsSentiment
)

class SymbolRepository(ABC):
    """Repository interface for symbols."""
    
    @abstractmethod
    async def get_by_ticker(self, ticker: str) -> Optional[Symbol]:
        """Get a symbol by its ticker."""
        pass
    
    @abstractmethod
    async def get_all(self) -> List[Symbol]:
        """Get all symbols."""
        pass
    
    @abstractmethod
    async def save(self, symbol: Symbol) -> Symbol:
        """Save a symbol."""
        pass
    
    @abstractmethod
    async def delete(self, ticker: str) -> bool:
        """Delete a symbol."""
        pass
    
    @abstractmethod
    async def search(self, query: str) -> List[Symbol]:
        """Search for symbols."""
        pass


class MarketDataRepository(ABC):
    """Repository interface for market data."""
    
    @abstractmethod
    async def get_historical_data(
        self, 
        symbol: str, 
        start_date: datetime, 
        end_date: Optional[datetime] = None,
        timeframe: str = "1d"
    ) -> List[MarketData]:
        """Get historical market data for a symbol."""
        pass
    
    @abstractmethod
    async def get_latest_price(self, symbol: str) -> Optional[MarketData]:
        """Get the latest price for a symbol."""
        pass
    
    @abstractmethod
    async def save(self, data: MarketData) -> MarketData:
        """Save market data."""
        pass
    
    @abstractmethod
    async def save_batch(self, data: List[MarketData]) -> List[MarketData]:
        """Save a batch of market data."""
        pass
    
    @abstractmethod
    async def get_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """Get fundamental data for a symbol."""
        pass


class OrderRepository(ABC):
    """Repository interface for orders."""
    
    @abstractmethod
    async def get_by_id(self, order_id: UUID) -> Optional[Order]:
        """Get an order by its ID."""
        pass
    
    @abstractmethod
    async def get_by_symbol(self, symbol: str) -> List[Order]:
        """Get orders for a specific symbol."""
        pass
    
    @abstractmethod
    async def get_active_orders(self) -> List[Order]:
        """Get all active orders."""
        pass
    
    @abstractmethod
    async def save(self, order: Order) -> Order:
        """Save an order."""
        pass
    
    @abstractmethod
    async def update_status(self, order_id: UUID, status: str, **kwargs) -> Order:
        """Update the status of an order."""
        pass
    
    @abstractmethod
    async def delete(self, order_id: UUID) -> bool:
        """Delete an order."""
        pass


class PositionRepository(ABC):
    """Repository interface for positions."""
    
    @abstractmethod
    async def get_by_id(self, position_id: UUID) -> Optional[Position]:
        """Get a position by its ID."""
        pass
    
    @abstractmethod
    async def get_by_symbol(self, symbol: str) -> List[Position]:
        """Get positions for a specific symbol."""
        pass
    
    @abstractmethod
    async def get_active_positions(self) -> List[Position]:
        """Get all active positions."""
        pass
    
    @abstractmethod
    async def save(self, position: Position) -> Position:
        """Save a position."""
        pass
    
    @abstractmethod
    async def close(self, position_id: UUID, close_price: float, close_date: datetime) -> Position:
        """Close a position."""
        pass
    
    @abstractmethod
    async def update(self, position: Position) -> Position:
        """Update a position."""
        pass


class PortfolioRepository(ABC):
    """Repository interface for portfolios."""
    
    @abstractmethod
    async def get_by_id(self, portfolio_id: UUID) -> Optional[Portfolio]:
        """Get a portfolio by its ID."""
        pass
    
    @abstractmethod
    async def get_by_name(self, name: str) -> Optional[Portfolio]:
        """Get a portfolio by its name."""
        pass
    
    @abstractmethod
    async def get_all(self) -> List[Portfolio]:
        """Get all portfolios."""
        pass
    
    @abstractmethod
    async def save(self, portfolio: Portfolio) -> Portfolio:
        """Save a portfolio."""
        pass
    
    @abstractmethod
    async def delete(self, portfolio_id: UUID) -> bool:
        """Delete a portfolio."""
        pass
    
    @abstractmethod
    async def update_cash(self, portfolio_id: UUID, cash: float) -> Portfolio:
        """Update the cash balance of a portfolio."""
        pass


class SignalRepository(ABC):
    """Repository interface for trading signals."""
    
    @abstractmethod
    async def get_by_id(self, signal_id: UUID) -> Optional[Signal]:
        """Get a signal by its ID."""
        pass
    
    @abstractmethod
    async def get_by_symbol(self, symbol: str) -> List[Signal]:
        """Get signals for a specific symbol."""
        pass
    
    @abstractmethod
    async def get_active_signals(self) -> List[Signal]:
        """Get all active signals."""
        pass
    
    @abstractmethod
    async def save(self, signal: Signal) -> Signal:
        """Save a signal."""
        pass
    
    @abstractmethod
    async def delete(self, signal_id: UUID) -> bool:
        """Delete a signal."""
        pass
    
    @abstractmethod
    async def mark_executed(self, signal_id: UUID) -> Signal:
        """Mark a signal as executed."""
        pass


class BacktestResultRepository(ABC):
    """Repository interface for backtest results."""
    
    @abstractmethod
    async def get_by_id(self, backtest_id: UUID) -> Optional[BacktestResult]:
        """Get a backtest result by its ID."""
        pass
    
    @abstractmethod
    async def get_by_strategy(self, strategy: str) -> List[BacktestResult]:
        """Get backtest results for a specific strategy."""
        pass
    
    @abstractmethod
    async def get_all(self) -> List[BacktestResult]:
        """Get all backtest results."""
        pass
    
    @abstractmethod
    async def save(self, result: BacktestResult) -> BacktestResult:
        """Save a backtest result."""
        pass
    
    @abstractmethod
    async def delete(self, backtest_id: UUID) -> bool:
        """Delete a backtest result."""
        pass


class NewsSentimentRepository(ABC):
    """Repository interface for news sentiment."""
    
    @abstractmethod
    async def get_by_id(self, sentiment_id: UUID) -> Optional[NewsSentiment]:
        """Get news sentiment by its ID."""
        pass
    
    @abstractmethod
    async def get_by_symbol(self, symbol: str) -> List[NewsSentiment]:
        """Get news sentiment for a specific symbol."""
        pass
    
    @abstractmethod
    async def get_recent(self, limit: int = 100) -> List[NewsSentiment]:
        """Get recent news sentiment."""
        pass
    
    @abstractmethod
    async def save(self, sentiment: NewsSentiment) -> NewsSentiment:
        """Save news sentiment."""
        pass
    
    @abstractmethod
    async def delete(self, sentiment_id: UUID) -> bool:
        """Delete news sentiment."""
        pass
