#!/bin/bash

# Don't use set -e so we can handle errors gracefully

# Print environment for debugging
echo "==== Database Environment Variables ===="
echo "TIMESCALE_HOST: ${TIMESCALE_HOST}"
echo "TIMESCALE_PORT: ${TIMESCALE_PORT}"
echo "TIMESCALE_DB: ${TIMESCALE_DB}"
echo "TIMESCALE_USER: ${TIMESCALE_USER}"
echo "===================================="

# Run database migrations
echo "Running database migrations..."
python -m market_data.run_migrations
MIGRATION_RESULT=$?

# Check if migrations were successful
if [ $MIGRATION_RESULT -ne 0 ]; then
    echo "WARNING: Database migrations failed with exit code $MIGRATION_RESULT"
    echo "The application will continue to start, but may not function correctly."
    echo "Check the logs for more information."
else
    echo "Database migrations completed successfully."
fi

# Start the application with the provided command
echo "Starting application..."
exec "$@"
