#!/bin/bash
# run_docker.sh - Script to run the trading platform in Docker

# Set default environment variables if not already set
POLYGON_API_KEY="g2sSLO_cQpKCQPWuLkA2w3d35IoNDAse"
export POLYGON_API_KEY=${POLYGON_API_KEY:-"demo"}
export ALPACA_API_KEY=${ALPACA_API_KEY:-"demo"}
export ALPACA_API_SECRET=${ALPACA_API_SECRET:-"demo"}
export DISCORD_TOKEN=${DISCORD_TOKEN:-"demo"}
export BROKER_TYPE=${BROKER_TYPE:-"mock"}

# Display banner
echo "====================================================="
echo "  AI-driven Multi-Agent Trading Platform"
echo "====================================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
  echo "Error: Docker is not running or not installed."
  echo "Please start Docker and try again."
  exit 1
fi

# Parse command line arguments
COMMAND="up -d"
SERVICES=""
PROFILES=""
USE_DISCORD=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --build)
      COMMAND="up -d --build"
      shift
      ;;
    --rebuild)
      echo "Rebuilding all containers..."
      docker-compose down
      docker-compose build --no-cache
      COMMAND="up -d"
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
    --discord)
      USE_DISCORD=true
      shift
      ;;
    --backtest)
      PROFILES="--profile backtest"
      shift
      ;;
    --help)
      echo "Usage: ./run_docker.sh [options]"
      echo ""
      echo "Options:"
      echo "  --build      Rebuild containers before starting"
      echo "  --rebuild    Force complete rebuild of all containers"
      echo "  --down       Stop and remove containers"
      echo "  --logs       Show logs for all services"
      echo "  --discord    Enable Discord bot (requires DISCORD_TOKEN)"
      echo "  --backtest   Run the backtesting service"
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

# Add Discord profile if requested
if [ "$USE_DISCORD" = true ]; then
  if [ "$DISCORD_TOKEN" = "demo" ]; then
    echo "Error: DISCORD_TOKEN must be set to use Discord bot."
    echo "Please set the DISCORD_TOKEN environment variable and try again."
    exit 1
  fi
  PROFILES="$PROFILES --profile discord"
  echo "Discord bot enabled."
fi

# Run Docker Compose with the specified command
if [ "$COMMAND" = "up -d" ] || [ "$COMMAND" = "up -d --build" ]; then
  if [ -z "$SERVICES" ]; then
    echo "Starting services..."
    docker-compose $COMMAND $PROFILES
    
    # Wait for services to start
    echo "Waiting for services to start..."
    sleep 10
    
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
    
    # Check if services are healthy
    echo "Checking service health..."
    sleep 5
    backend_status=$(docker inspect --format='{{.State.Health.Status}}' $(docker-compose ps -q backend) 2>/dev/null || echo "container not found")
    gradio_status=$(docker inspect --format='{{.State.Health.Status}}' $(docker-compose ps -q gradio) 2>/dev/null || echo "container not found")
    
    echo "Backend status: $backend_status"
    echo "Gradio UI status: $gradio_status"
    
    if [ "$backend_status" != "healthy" ] || [ "$gradio_status" != "healthy" ]; then
      echo "\nSome services may not be fully healthy yet. Check logs with './run_docker.sh --logs'"
    fi
  else
    echo "Starting $SERVICES..."
    docker-compose $COMMAND $PROFILES $SERVICES
  fi
else
  docker-compose $COMMAND $PROFILES
fi
