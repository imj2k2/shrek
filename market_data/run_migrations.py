#!/usr/bin/env python3
"""
Script to run database migrations on startup.
This ensures the database schema is up to date when the application starts.
"""
import os
import sys
import time
import logging
import subprocess
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("migration_runner")

# Get migration directory
MIGRATION_DIR = Path(__file__).parent / "migrations"

# Import psycopg2 for better database connection handling
import psycopg2

# Attempt to connect to TimescaleDB and run migrations
def run_migrations():
    """Run Alembic migrations to update the database schema."""
    # Print environment variables for debugging
    logger.info("Running migrations with the following environment:")
    for env_var in ['TIMESCALE_HOST', 'TIMESCALE_PORT', 'TIMESCALE_DB', 'TIMESCALE_USER']:
        logger.info(f"{env_var}: {os.environ.get(env_var)}")
    
    # Get database connection parameters from environment
    host = os.environ.get('TIMESCALE_HOST', 'timescaledb')
    port = os.environ.get('TIMESCALE_PORT', '5432')
    dbname = os.environ.get('TIMESCALE_DB', 'shrek')
    user = os.environ.get('TIMESCALE_USER', 'postgres')
    password = os.environ.get('TIMESCALE_PASSWORD', 'postgres')
    
    # Wait for database to be ready
    retry_count = 0
    max_retries = 60  # Increased max retries
    retry_delay = 5   # Increased delay between retries
    
    logger.info("Waiting for database to be ready...")
    while retry_count < max_retries:
        try:
            # Try to connect directly using psycopg2
            conn = psycopg2.connect(
                host=host,
                port=port,
                dbname=dbname,
                user=user,
                password=password,
                connect_timeout=3
            )
            conn.close()
            logger.info("Successfully connected to the database.")
            break
        except psycopg2.OperationalError as e:
            logger.warning(f"Database not ready (attempt {retry_count + 1}/{max_retries}): {e}")
            retry_count += 1
            time.sleep(retry_delay)
        except Exception as e:
            logger.warning(f"Unexpected error connecting to database (attempt {retry_count + 1}/{max_retries}): {e}")
            retry_count += 1
            time.sleep(retry_delay)
    
    if retry_count >= max_retries:
        logger.error("Failed to connect to the database after maximum retry attempts.")
        return False
    
    # Add a short delay to ensure database is fully ready
    time.sleep(5)
    
    # Run migrations
    try:
        logger.info("Running database migrations...")
        # Show the current migration state
        current_cmd = ["alembic", "-c", str(MIGRATION_DIR / "alembic.ini"), "current"]
        logger.info(f"Checking current migration state with: {' '.join(current_cmd)}")
        current_result = subprocess.run(
            current_cmd,
            capture_output=True,
            text=True,
            check=False
        )
        logger.info(f"Current migration state: {current_result.stdout.strip() if current_result.returncode == 0 else 'No migrations applied yet'}")
        
        # Run the actual migration
        upgrade_cmd = ["alembic", "-c", str(MIGRATION_DIR / "alembic.ini"), "upgrade", "head"]
        logger.info(f"Running migration with: {' '.join(upgrade_cmd)}")
        result = subprocess.run(
            upgrade_cmd,
            capture_output=True,
            text=True,
            check=False  # Don't raise exception so we can capture output
        )
        
        if result.returncode == 0:
            logger.info(f"Migration completed successfully: {result.stdout.strip()}")
            return True
        else:
            logger.error(f"Migration failed with return code {result.returncode}")
            logger.error(f"Migration error output: {result.stderr.strip()}")
            logger.error(f"Migration standard output: {result.stdout.strip()}")
            return False
    except Exception as e:
        logger.error(f"Exception during migration: {str(e)}")
        return False

if __name__ == "__main__":
    successful = run_migrations()
    if not successful:
        sys.exit(1)
