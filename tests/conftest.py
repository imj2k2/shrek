"""
Pytest configuration file for the Shrek Trading Platform.
Provides fixtures for testing.
"""

import asyncio
import os
import pytest
from typing import AsyncGenerator, Generator

import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from redis.asyncio import Redis

# Create test app
from ui.app import app

# Import models and database connections
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from market_data.database.connection import Base, get_async_db
from core.domain.models import (
    Symbol, MarketData, Order, Position, Portfolio, 
    Signal, BacktestResult, NewsSentiment
)

# Test database URLs
TEST_DB_URL = "sqlite:///./test.db"
TEST_ASYNC_DB_URL = "sqlite+aiosqlite:///./test.db"
TEST_REDIS_URL = "redis://localhost:6379/1"  # Use database 1 for tests


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_engine():
    """Create a SQLAlchemy engine for testing."""
    engine = create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="session")
async def async_test_engine():
    """Create an async SQLAlchemy engine for testing."""
    engine = create_async_engine(TEST_ASYNC_DB_URL, connect_args={"check_same_thread": False})
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture(scope="function")
def db_session(test_engine):
    """Create a SQLAlchemy session for testing."""
    connection = test_engine.connect()
    transaction = connection.begin()
    TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    session = TestSessionLocal()
    
    yield session
    
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture(scope="function")
async def async_db_session(async_test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create an async SQLAlchemy session for testing."""
    TestAsyncSessionLocal = async_sessionmaker(expire_on_commit=False, bind=async_test_engine)
    async with TestAsyncSessionLocal() as session:
        yield session


@pytest.fixture(scope="module")
def client() -> Generator:
    """Create a FastAPI test client."""
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
async def redis():
    """Create a Redis client for testing."""
    redis_client = Redis.from_url(TEST_REDIS_URL)
    yield redis_client
    await redis_client.flushdb()
    await redis_client.close()


@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    # Create a DataFrame with sample market data
    dates = pd.date_range(start="2023-01-01", periods=100)
    np.random.seed(42)
    
    # Generate random price data
    prices = 100 * (1 + np.cumsum(np.random.normal(0, 0.01, 100)))
    
    data = pd.DataFrame({
        "open": prices * 0.99,
        "high": prices * 1.01,
        "low": prices * 0.98,
        "close": prices,
        "volume": np.random.randint(1000, 100000, 100)
    }, index=dates)
    
    return data


@pytest.fixture
def sample_symbol():
    """Create a sample symbol for testing."""
    return Symbol(
        ticker="AAPL",
        name="Apple Inc.",
        exchange="NASDAQ",
        asset_class="equity",
        currency="USD",
        is_tradable=True
    )


@pytest.fixture
def sample_portfolio():
    """Create a sample portfolio for testing."""
    return Portfolio(
        name="Test Portfolio",
        cash=100000.0
    )
