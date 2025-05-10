"""
Database connection and session management for TimescaleDB.
"""

import logging
import os
from typing import AsyncGenerator, Optional

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import (
    AsyncSession, 
    create_async_engine, 
    async_sessionmaker
)
from sqlalchemy.orm import declarative_base, sessionmaker

logger = logging.getLogger(__name__)

# Base class for SQLAlchemy models
Base = declarative_base()

# Get database connection settings from environment variables
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "shrek")

# Connection strings
SYNC_DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
ASYNC_DATABASE_URL = f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# SQLite fallback for development and testing
SQLITE_URL = os.getenv("SQLITE_URL", "sqlite:///./shrek.db")
ASYNC_SQLITE_URL = os.getenv("ASYNC_SQLITE_URL", "sqlite+aiosqlite:///./shrek.db")

# Use SQLite if environment variable is set or TimescaleDB connection fails
USE_SQLITE = os.getenv("USE_SQLITE", "false").lower() == "true"


def get_sync_engine(use_sqlite: Optional[bool] = None):
    """Get a synchronous SQLAlchemy engine."""
    if use_sqlite is None:
        use_sqlite = USE_SQLITE
    
    try:
        if use_sqlite:
            engine = create_engine(
                SQLITE_URL, 
                connect_args={"check_same_thread": False},
                echo=os.getenv("SQL_ECHO", "false").lower() == "true"
            )
            logger.info("Using SQLite database")
        else:
            engine = create_engine(
                SYNC_DATABASE_URL,
                echo=os.getenv("SQL_ECHO", "false").lower() == "true"
            )
            # Test connection
            with engine.connect() as conn:
                conn.execute("SELECT 1")
            logger.info("Connected to PostgreSQL/TimescaleDB")
        
        return engine
    except Exception as e:
        if not use_sqlite:
            logger.warning(f"Failed to connect to PostgreSQL/TimescaleDB: {str(e)}")
            logger.warning("Falling back to SQLite")
            return get_sync_engine(use_sqlite=True)
        else:
            logger.error(f"Failed to connect to SQLite: {str(e)}")
            raise


def get_async_engine(use_sqlite: Optional[bool] = None):
    """Get an asynchronous SQLAlchemy engine."""
    if use_sqlite is None:
        use_sqlite = USE_SQLITE
    
    try:
        if use_sqlite:
            engine = create_async_engine(
                ASYNC_SQLITE_URL,
                connect_args={"check_same_thread": False},
                echo=os.getenv("SQL_ECHO", "false").lower() == "true"
            )
            logger.info("Using SQLite database (async)")
        else:
            engine = create_async_engine(
                ASYNC_DATABASE_URL,
                echo=os.getenv("SQL_ECHO", "false").lower() == "true"
            )
            logger.info("Connected to PostgreSQL/TimescaleDB (async)")
        
        return engine
    except Exception as e:
        if not use_sqlite:
            logger.warning(f"Failed to connect to PostgreSQL/TimescaleDB (async): {str(e)}")
            logger.warning("Falling back to SQLite (async)")
            return get_async_engine(use_sqlite=True)
        else:
            logger.error(f"Failed to connect to SQLite (async): {str(e)}")
            raise


def get_session_maker(engine=None):
    """Get a synchronous session maker."""
    if engine is None:
        engine = get_sync_engine()
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_async_session_maker(engine=None):
    """Get an asynchronous session maker."""
    if engine is None:
        engine = get_async_engine()
    return async_sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Default session factories
SessionLocal = get_session_maker()
AsyncSessionLocal = get_async_session_maker()


def get_db():
    """Get a synchronous database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """Get an asynchronous database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


def init_db():
    """Initialize the database."""
    engine = get_sync_engine()
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized")


async def init_async_db():
    """Initialize the database asynchronously."""
    engine = get_async_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database initialized (async)")
