# Discord Bot Module Documentation

## Overview
The `discord_bot/` directory provides integration with Discord for trading platform notifications and (potentially) commands. It is designed for extensibility, allowing the bot to notify users of trading signals, executions, and risk alerts, and can be run as a standalone service.

## Files & Purpose
- **bot.py**: Main Discord bot logic, notification functions, and runner.

---

## bot.py

- **Purpose**: Sends notifications about trading events (signals, executions, risk alerts) to a Discord channel or logs them if Discord is not configured. Provides a runner for starting the bot.
- **Key Functions**:
    - `notify_signal(signal)`: Async. Notifies about new trading signals.
    - `notify_execution(result)`: Async. Notifies about trade executions.
    - `notify_risk_alert(alert, alert_type="unknown")`: Async. Notifies about risk management alerts.
    - `run_discord_bot()`: Starts the Discord bot using the token from environment variables (stub for real implementation).
- **Environment Integration**: Uses `DISCORD_TOKEN` and `API_BASE_URL` from environment variables for configuration. If no token is provided, notifications are logged instead of sent.
- **Design**: All notification functions are async for future integration with Discord APIs. Logging is used for both normal and fallback operation.

---

*Extend this document if/when the Discord bot is expanded to support commands, richer notifications, or two-way communication.*
