import uuid
from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, ForeignKey, Text, JSON
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func
import datetime

# Create base class
Base = declarative_base()

class Symbol(Base):
    __tablename__ = "symbols"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol = Column(String(20), unique=True, nullable=False, index=True)
    name = Column(String(255))
    description = Column(Text)
    sector = Column(String(100))
    industry = Column(String(100))
    exchange = Column(String(20))
    is_active = Column(Boolean, default=True)
    is_etf = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    # Relationships
    market_data = relationship("MarketData", back_populates="symbol_ref", cascade="all, delete-orphan")
    fundamentals = relationship("Fundamental", back_populates="symbol_ref", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Symbol(symbol='{self.symbol}', name='{self.name}')>"


class MarketData(Base):
    __tablename__ = "market_data"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol_id = Column(UUID(as_uuid=True), ForeignKey("symbols.id"), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Integer)
    vwap = Column(Float)
    source = Column(String(50))  # Source of the data (e.g., "polygon", "yahoo")
    timeframe = Column(String(10))  # e.g., "1min", "1h", "1d"
    
    # Technical indicators (nullable to allow calculation after insertion)
    sma_50 = Column(Float)
    sma_200 = Column(Float)
    ema_12 = Column(Float)
    ema_26 = Column(Float)
    macd = Column(Float)
    macd_signal = Column(Float)
    macd_hist = Column(Float)
    rsi = Column(Float)
    
    # Relationships
    symbol_ref = relationship("Symbol", back_populates="market_data")
    
    # Add TimescaleDB hypertable statement in migration
    
    def __repr__(self):
        return f"<MarketData(symbol='{self.symbol_ref.symbol}', timestamp='{self.timestamp}')>"


class Fundamental(Base):
    __tablename__ = "fundamentals"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    symbol_id = Column(UUID(as_uuid=True), ForeignKey("symbols.id"), nullable=False)
    date = Column(DateTime, nullable=False)
    pe_ratio = Column(Float)
    eps = Column(Float)
    dividend_yield = Column(Float)
    market_cap = Column(Float)
    price_to_book = Column(Float)
    price_to_sales = Column(Float)
    debt_to_equity = Column(Float)
    profit_margin = Column(Float)
    return_on_equity = Column(Float)
    beta = Column(Float)
    
    # Relationships
    symbol_ref = relationship("Symbol", back_populates="fundamentals")
    
    def __repr__(self):
        return f"<Fundamental(symbol='{self.symbol_ref.symbol}', date='{self.date}')>"


class BacktestResult(Base):
    __tablename__ = "backtest_results"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    strategy = Column(String(100))
    symbols = Column(JSONB)  # Store symbols and their position types
    initial_capital = Column(Float)
    final_capital = Column(Float)
    total_return = Column(Float)
    annual_return = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    win_rate = Column(Float)
    profit_factor = Column(Float)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    config = Column(JSONB)  # Store configuration parameters
    
    # Relationships
    trades = relationship("BacktestTrade", back_populates="backtest", cascade="all, delete-orphan")
    equity_curve = relationship("BacktestEquityCurve", back_populates="backtest", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<BacktestResult(name='{self.name}', strategy='{self.strategy}', return='{self.total_return}')>"


class BacktestTrade(Base):
    __tablename__ = "backtest_trades"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    backtest_id = Column(UUID(as_uuid=True), ForeignKey("backtest_results.id"), nullable=False)
    symbol = Column(String(20), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    action = Column(String(10), nullable=False)  # 'buy' or 'sell'
    position_type = Column(String(10), nullable=False, default="long")  # 'long' or 'short'
    price = Column(Float, nullable=False)
    quantity = Column(Integer, nullable=False)
    commission = Column(Float, default=0.0)
    profit_loss = Column(Float)
    
    # Relationships
    backtest = relationship("BacktestResult", back_populates="trades")
    
    def __repr__(self):
        return f"<BacktestTrade(symbol='{self.symbol}', action='{self.action}', price='{self.price}')>"


class BacktestEquityCurve(Base):
    __tablename__ = "backtest_equity_curve"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    backtest_id = Column(UUID(as_uuid=True), ForeignKey("backtest_results.id"), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    equity = Column(Float, nullable=False)
    cash = Column(Float, nullable=False)
    holdings_value = Column(Float, nullable=False)
    daily_return = Column(Float)
    drawdown = Column(Float, default=0.0)
    
    # Relationships
    backtest = relationship("BacktestResult", back_populates="equity_curve")
    
    # Add TimescaleDB hypertable statement in migration
    
    def __repr__(self):
        return f"<BacktestEquityCurve(timestamp='{self.timestamp}', equity='{self.equity}')>"


class ScreenerCriteria(Base):
    __tablename__ = "screener_criteria"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    criteria = Column(JSONB, nullable=False)  # Store all criteria as JSON
    position_type = Column(String(10), default="long")  # 'long' or 'short' as default
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    last_used = Column(DateTime)
    
    # Relationships
    results = relationship("ScreenerResult", back_populates="criteria_ref", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<ScreenerCriteria(name='{self.name}')>"


class ScreenerResult(Base):
    __tablename__ = "screener_results"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    criteria_id = Column(UUID(as_uuid=True), ForeignKey("screener_criteria.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    results = Column(JSONB)  # Store results as JSON with symbols and data
    count = Column(Integer)
    
    # Relationships
    criteria_ref = relationship("ScreenerCriteria", back_populates="results")
    
    def __repr__(self):
        return f"<ScreenerResult(count='{self.count}', timestamp='{self.timestamp}')>"
