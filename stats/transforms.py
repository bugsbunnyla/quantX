import numpy as np

class Transform:
    """
    SIGNAL ENGINE

    Z:
        (x - μ)/σ

    WIN:
        clip(5%,95%)

    TANH:
        squash nonlinear

    DETERND:
        x - rolling_mean
    """

    def rank(self, x):
        return x.rank()

    #def zscore(self, x):
    #    return (x - x.mean()) / (x.std()+1e-9)
    def zscore(self, x):
        mean = x.mean()
        std = x.std()
        return (x - mean) / std if (std != 0 and not np.isnan(std)) else (x * 0)

    def winsorize(self, x):
        lo, hi = x.quantile(0.05), x.quantile(0.95)
        return x.clip(lo, hi)

    def tanh(self, x):
        return np.tanh(x)

    def detrend(self, x):
        return x - x.rolling(5).mean()