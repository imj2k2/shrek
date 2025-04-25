class TradingExecutor:
    def __init__(self, broker):
        self.broker = broker
    def execute(self, signal):
        # Send signal to broker (Lumibot/Alpaca/Robinhood)
        return self.broker.place_order(signal)
