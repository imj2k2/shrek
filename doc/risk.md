# Risk Module Documentation

## Overview
The `risk/` directory provides advanced risk management for algorithmic trading, including drawdown limits, position sizing, trailing stops, and integration with Discord for real-time alerts.

## Files & Purpose
- **advanced_risk_manager.py**: Implements all core risk management logic and alerting.

---

## advanced_risk_manager.py

### Classes
- **AdvancedRiskManager**: Advanced risk management with trailing stops, drawdown limits, and position sizing.
    - `__init__(max_drawdown=0.1, max_position_size=0.2, trailing_stop_pct=0.05, notify_discord=True)`: Initializes risk parameters and alerting.
    - `assess(portfolio)`: Evaluates portfolio risk, generates alerts, and returns risk metrics.
    - `_check_drawdown(equity, timestamp)`: Checks if drawdown exceeds threshold.
    - `_check_position_sizes(positions, equity)`: Checks if any position exceeds allowed size.
    - `_check_trailing_stops(positions)`: Updates and checks trailing stops for all positions.
    - `_calculate_drawdown(equity)`: Computes current drawdown as a percentage.
    - `_calculate_largest_position(positions, equity)`: Computes largest position as percent of portfolio.
    - `_get_trailing_stop_levels()`: Returns current trailing stop levels for all positions.
    - `_send_discord_alerts(alerts)`: Sends alerts to Discord asynchronously if enabled.
    - `_send_discord_message(message, alert_type)`: Helper for async Discord notification.
    - `_format_alert_for_discord(alert)`: Formats alert for Discord message.
    - `reset_trading_halt()`: Manually resets trading halt status and notifies Discord.
    - `get_alert_history(limit=20)`: Returns recent alert history.

### Functions
- **notify_risk_alert(message, alert_type="unknown")**: Async function for sending risk alerts; logs if Discord is unavailable.

---

## Design Notes
- **Real-time alerting**: Integrates with Discord for instant risk notifications.
- **Multi-risk checks**: Simultaneously checks drawdown, position size, and trailing stops.
- **Extensible**: Easily add new risk checks or alerting channels.
- **Thread-safe**: Uses threading for non-blocking notifications.

---

*Extend this document as new risk management features or alert types are added.*
