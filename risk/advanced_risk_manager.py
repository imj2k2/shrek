from typing import List, Dict

class AdvancedRiskManager:
    def __init__(self, max_drawdown=0.1, trailing_stop=0.05, max_position_size=0.2):
        self.max_drawdown = max_drawdown
        self.trailing_stop = trailing_stop
        self.max_position_size = max_position_size
        self.peak = None
    def assess(self, portfolio: Dict) -> Dict:
        equity = portfolio['equity']
        if self.peak is None or equity > self.peak:
            self.peak = equity
        drawdown = (self.peak - equity) / self.peak if self.peak else 0
        alerts = []
        if drawdown > self.max_drawdown:
            alerts.append({'halt_trading': True, 'reason': 'drawdown'})
        for pos in portfolio.get('positions', []):
            if pos['pnl_pct'] < -self.trailing_stop:
                alerts.append({'halt_trading': True, 'reason': f'trailing_stop on {pos["symbol"]}'})
            if pos['size_pct'] > self.max_position_size:
                alerts.append({'reduce_position': True, 'reason': f'position_size on {pos["symbol"]}'})
        return {'alerts': alerts, 'drawdown': drawdown}
