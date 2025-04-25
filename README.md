# AI-driven Multi-Agent Algorithmic Trading Platform

## Overview
This platform leverages multi-agent AI for automated trading across stocks, options, and crypto, with real-time data, risk management, and interactive UI/Discord integration.

---

## Features
- Modular agents (stocks, options, crypto, risk, coordinator)
- FastAPI backend, Gradio UI, Discord bot
- Real-time data via Polygon.io/Yahoo
- Persistent storage (Redis/file)
- Advanced technical indicators
- Docker Compose deployment

---

## Setup
1. Clone the repo and enter the directory:
   ```bash
   git clone <repo-url>
   cd shrek
   ```
2. Set environment variables (see `.env.example`):
   - `POLYGON_API_KEY`, `ALPACA_API_KEY`, `ALPACA_API_SECRET`, `ROBINHOOD_USERNAME`, `ROBINHOOD_PASSWORD`, `DISCORD_TOKEN`, etc.
3. Build and run with Docker Compose:
   ```bash
   docker-compose up --build
   ```
4. Access:
   - FastAPI: http://localhost:8000/docs
   - Gradio UI: http://localhost:7860
   - Discord bot: Invite to your server

---

## Architecture
- `agents/`: Modular agent implementations
- `indicators/`: Technical indicators
- `mcp/`: Model Context Protocol for agent comms
- `trading/`: Broker integration (Lumibot, Alpaca, Robinhood)
- `risk/`: Real-time risk management
- `data/`: Market data and persistent storage
- `ui/`: FastAPI backend, Gradio UI
- `discord_bot/`: Discord integration
- `config/`: Environment/config management

---

## Usage
- Use Gradio UI for portfolio, signals, and trade execution
- Use Discord bot for real-time alerts and scheduled reports
- Extend agents or add new strategies in `agents/`

---

## Security
- All secrets/configs via environment variables
- No hardcoded credentials

---

## Testing
- Add test scripts in `/tests` (not included by default)
- Run with `pytest` or similar

---

## License
MIT (or specify your license)
