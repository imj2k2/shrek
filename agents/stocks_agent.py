from indicators import moving_averages, rsi, macd, bollinger, atr, vwap

class StocksAgent:
    def __init__(self):
        pass
    def generate_signals(self, data):
        close = data['close']
        signals = {}
        signals['rsi'] = rsi.rsi(close).iloc[-1]
        signals['ma_20'] = moving_averages.ma(close, 20).iloc[-1]
        signals['ema_20'] = moving_averages.ema(close, 20).iloc[-1]
        signals['macd'], _, _ = macd.macd(close)
        signals['bollinger_upper'], signals['bollinger_mid'], signals['bollinger_lower'] = bollinger.bollinger_bands(close)
        signals['atr'] = atr.atr(data['high'], data['low'], data['close']).iloc[-1]
        signals['vwap'] = vwap.vwap(data['high'], data['low'], data['close'], data['volume']).iloc[-1]
        return signals
