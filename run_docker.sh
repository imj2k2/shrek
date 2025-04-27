#!/bin/bash
# run_docker.sh - Script to run the trading platform in Docker

# Set default environment variables if not already set
export POLYGON_API_KEY=${POLYGON_API_KEY:-"demo_key"}
export ALPACA_API_KEY=${ALPACA_API_KEY:-"demo_key"}
export ALPACA_API_SECRET=${ALPACA_API_SECRET:-"demo_secret"}
export DISCORD_TOKEN=${DISCORD_TOKEN:-""}
export BROKER_TYPE=${BROKER_TYPE:-"mock"}

# Display banner
echo "====================================================="
echo "  AI-driven Multi-Agent Trading Platform"
echo "====================================================="
echo "Starting services with Docker Compose..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
  echo "Error: Docker is not running or not installed."
  echo "Please start Docker and try again."
  exit 1
fi

# Parse command line arguments
COMMAND="up -d"
SERVICES=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --build)
      COMMAND="up -d --build"
      shift
      ;;
    --down)
      COMMAND="down"
      shift
      ;;
    --logs)
      COMMAND="logs -f"
      shift
      ;;
    --backtest)
      SERVICES="backtest-service"
      shift
      ;;
    --help)
      echo "Usage: ./run_docker.sh [options]"
      echo ""
      echo "Options:"
      echo "  --build      Rebuild containers before starting"
      echo "  --down       Stop and remove containers"
      echo "  --logs       Show logs for all services"
      echo "  --backtest   Run the backtesting service only"
      echo "  --help       Show this help message"
      echo ""
      echo "Environment variables:"
      echo "  POLYGON_API_KEY     API key for Polygon.io"
      echo "  ALPACA_API_KEY      API key for Alpaca"
      echo "  ALPACA_API_SECRET   API secret for Alpaca"
      echo "  DISCORD_TOKEN       Token for Discord bot"
      echo "  BROKER_TYPE         Broker type (mock, alpaca, robinhood)"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information."
      exit 1
      ;;
  esac
done

# Run Docker Compose with the specified command
if [ "$COMMAND" = "up -d" ] || [ "$COMMAND" = "up -d --build" ]; then
  if [ -z "$SERVICES" ]; then
    echo "Starting all services..."
    docker-compose $COMMAND
    
    # Wait for services to start
    echo "Waiting for services to start..."
    sleep 5
    
    # Show service status
    echo "====================================================="
    echo "Service Status:"
    docker-compose ps
    
    # Show access URLs
    echo "====================================================="
    echo "Access the services at:"
    echo "  - Backend API: http://localhost:8000"
    echo "  - Gradio UI:   http://localhost:7860"
    echo "====================================================="
  else
    echo "Starting $SERVICES..."
    docker-compose $COMMAND $SERVICES
  fi
else
  docker-compose $COMMAND
fi
