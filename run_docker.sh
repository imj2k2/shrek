#!/bin/bash
# run_docker.sh - Script to run the trading platform in Docker

# Load environment variables from .env file if it exists
if [ -f ".env" ]; then
  echo "Loading environment variables from .env file..."
  set -a  # automatically export all variables
  source .env
  set +a  # stop automatically exporting
else
  echo "Warning: .env file not found. Using default values."
  # You can create a .env file from the template with: cp .env.template .env
fi

# Set default values for environment variables if not set in .env
export POLYGON_API_KEY=${POLYGON_API_KEY:-"demo"}
export ALPACA_API_KEY=${ALPACA_API_KEY:-"demo"}
export ALPACA_API_SECRET=${ALPACA_API_SECRET:-"demo"}
export DISCORD_TOKEN=${DISCORD_TOKEN:-"demo"}
export BROKER_TYPE=${BROKER_TYPE:-"mock"}
export DEBUG_TRADING=${DEBUG_TRADING:-"true"}

# TimescaleDB settings
export TIMESCALE_USER=${TIMESCALE_USER:-"postgres"}
export TIMESCALE_PASSWORD=${TIMESCALE_PASSWORD:-"postgres"}
export TIMESCALE_HOST=${TIMESCALE_HOST:-"timescaledb"}
export TIMESCALE_PORT=${TIMESCALE_PORT:-5432}
export TIMESCALE_DB=${TIMESCALE_DB:-"shrek"}

# OpenTelemetry settings
export OTEL_ENABLED=${OTEL_ENABLED:-"false"}
export OTEL_SERVICE_NAME=${OTEL_SERVICE_NAME:-"shrek-backend"}
export OTEL_EXPORTER_OTLP_ENDPOINT=${OTEL_EXPORTER_OTLP_ENDPOINT:-"http://otel-collector:4317"}
export ENVIRONMENT=${ENVIRONMENT:-"development"}

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
USE_REACT_UI=true
USE_MONITORING=true

while [[ $# -gt 0 ]]; do
  case $1 in
    --build)
      COMMAND="up -d --build"
      shift
      ;;
    --rebuild)
      echo "Rebuilding all containers..."
      docker-compose down
      docker-compose build 
      COMMAND="up -d"
      shift
      ;;
    --full-rebuild)
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
    --no-react)
      USE_REACT_UI=false
      shift
      ;;
    --no-monitoring)
      USE_MONITORING=false
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
      echo "  --no-react   Disable React UI (use Gradio only)"
      echo "  --no-monitoring Disable monitoring components (TimescaleDB and OpenTelemetry)"
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

# Configure services based on options
if [ "$USE_REACT_UI" = false ]; then
  export COMPOSE_PROFILES="$COMPOSE_PROFILES,no-react"
  echo "React UI disabled. Using Gradio UI only."
fi

if [ "$USE_MONITORING" = false ]; then
  export COMPOSE_PROFILES="$COMPOSE_PROFILES,no-monitoring"
  echo "Monitoring components disabled."
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
    echo "  - Backend API:   http://localhost:8080"
    echo "  - Gradio UI:     http://localhost:7860"
    if [ "$USE_REACT_UI" = true ]; then
      echo "  - Modern React UI: http://localhost:3000"
    fi
    if [ "$USE_MONITORING" = true ]; then
      echo "  - TimescaleDB:   localhost:5432"
      echo "  - Metrics:        http://localhost:8888"
    fi
    echo "====================================================="
    
    # Check if services are healthy
    echo "Checking service health..."
    sleep 5
    backend_status=$(docker inspect --format='{{.State.Health.Status}}' $(docker-compose ps -q backend) 2>/dev/null || echo "container not found")
    gradio_status=$(docker inspect --format='{{.State.Health.Status}}' $(docker-compose ps -q gradio) 2>/dev/null || echo "container not found")
    
    echo "Backend status: $backend_status"
    echo "Gradio UI status: $gradio_status"
    
    if [ "$USE_REACT_UI" = true ]; then
      react_status=$(docker inspect --format='{{.State.Health.Status}}' $(docker-compose ps -q react-ui) 2>/dev/null || echo "container not found")
      echo "React UI status: $react_status"
    fi
    
    if [ "$USE_MONITORING" = true ]; then
      timescale_status=$(docker inspect --format='{{.State.Health.Status}}' $(docker-compose ps -q timescaledb) 2>/dev/null || echo "container not found")
      echo "TimescaleDB status: $timescale_status"
    fi
    
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
