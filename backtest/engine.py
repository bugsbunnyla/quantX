import numpy as np

class Backtest:

    def run(self, returns, weights):
        pnl = (returns * weights).sum(axis=1)
        equity = (1 + pnl).cumprod()
        return equity, pnl