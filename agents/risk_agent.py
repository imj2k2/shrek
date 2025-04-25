class RiskAgent:
    def __init__(self, max_drawdown=0.1, trailing_stop=0.05):
        self.max_drawdown = max_drawdown
        self.trailing_stop = trailing_stop
        self.peak = None
    def assess(self, portfolio):
        # Example: check drawdown, stop-loss, position sizing
        equity = portfolio['equity']
        if self.peak is None or equity > self.peak:
            self.peak = equity
        drawdown = (self.peak - equity) / self.peak if self.peak else 0
        if drawdown > self.max_drawdown:
            return {'halt_trading': True, 'reason': 'drawdown'}
        # Trailing stop logic
        for pos in portfolio.get('positions', []):
            if pos['pnl_pct'] < -self.trailing_stop:
                return {'halt_trading': True, 'reason': 'trailing_stop'}
        return {'halt_trading': False}
