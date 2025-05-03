# Indicators Module Documentation

## Overview
The `indicators/` directory implements all technical analysis indicators used by trading agents and strategies. Each indicator is implemented as a standalone function, designed for use with pandas Series (or numpy arrays where appropriate), and can be easily extended or reused.

## Files & Purpose
- **atr.py**: Average True Range (ATR) calculation for volatility measurement.
- **bollinger.py**: Bollinger Bands for mean reversion and volatility.
- **macd.py**: Moving Average Convergence Divergence (MACD) for trend/momentum.
- **moving_averages.py**: Simple, exponential, and generic moving averages.
- **options.py**: Black-Scholes pricing, implied volatility, and Greeks for options.
- **rsi.py**: Relative Strength Index (RSI) for momentum/overbought/oversold.
- **vwap.py**: Volume Weighted Average Price (VWAP) for price/volume analysis.

---

## Function-Level Documentation

### atr.py
- **atr(high, low, close, period=14)**: Computes ATR using rolling max of true range over the specified period.

### bollinger.py
- **bollinger_bands(series, period=20, num_std=2)**: Returns upper band, SMA, and lower band for given period and standard deviation multiplier.

### macd.py
- **macd(series, fast=12, slow=26, signal=9)**: Returns MACD line, signal line, and histogram for a price series.

### moving_averages.py
- **sma(series, period=20)**: Simple moving average.
- **ema(series, period=20)**: Exponential moving average.
- **ma(series, period=20)**: Alias for SMA.

### options.py
- **black_scholes_price(S, K, T, r, sigma, option_type='call')**: Black-Scholes price for call/put.
- **implied_volatility(option_price, S, K, T, r, option_type='call', tol=1e-5, max_iterations=100)**: Computes implied volatility via Newton-Raphson.
- **greeks(S, K, T, r, sigma, option_type='call')**: Returns delta, gamma, theta, vega for option.

### rsi.py
- **rsi(series, period=14)**: Computes RSI using rolling average gains/losses.

### vwap.py
- **vwap(high, low, close, volume)**: Computes VWAP as cumulative typical price * volume divided by cumulative volume.

---

## Design Notes
- All indicators are vectorized for efficient computation on pandas Series.
- Used extensively by agents for signal generation and strategy logic.
- Easily extensible for new indicators or variants.

---

*Extend this document as new indicators are added or existing ones are enhanced.*
