import numpy as np

class Risk:
    """
    RISK SYSTEM

    VOL:
        σ = std(r)

    SHARPE:
        SR = E[r] / σ

    DRAWDOWN:
        DD = P / max(P) - 1
    """

    def volatility(self, r):
        vol = r.rolling(10).std()
        return vol.replace([np.inf, -np.inf], np.nan).fillna(0)

    #def sharpe(self, r):
    #    std = r.std()
    #    return 0 if (std == 0 or np.isnan(std)) else (r.mean() / std)
    def sharpe(self,r, mask=None, eps=1e-12):

        # guard against object leakage
        if hasattr(r, "returns"):
            r = r.returns
        
        r = np.asarray(r, dtype=float)

        # optional alignment mask (recommended in your system)
        if mask is not None:
           r = r[mask]

        # remove NaN / inf (critical for regression pipelines)
           r = r[np.isfinite(r)]

        # guard: not enough data
        if r.size < 2:
           return 0.0

        mu = np.mean(r)
        sigma = np.std(r)

        # guard: no volatility case
        if sigma < eps or not np.isfinite(sigma):
           return 0.0

        return mu / sigma

    def drawdown(self, p):
        return (p / p.cummax()) - 1

    def correlation(self, r):
        return r.corr()