import numpy as np

class Pairs:

    """
    PAIRS TRADING (Z-SPREAD)

    FORMULA:
        spread = A - βB
        z = (spread - μ)/σ
    """

    #def spread(self, a, b):

    #   beta = np.cov(a, b)[0,1] / (np.var(b) + 1e-9)
    #    spread = a - beta * b

    #    z = (spread - np.mean(spread)) / (np.std(spread)+1e-9)

    #    return z

    def spread(self, a, b):
        a = np.asarray(a)
        b = np.asarray(b)

        # clean NaNs / inf
        mask = np.isfinite(a) & np.isfinite(b)
        a = a[mask]
        b = b[mask]

        # guard against empty or degenerate data
        if len(a) < 2 or len(b) < 2:
           return np.zeros_like(a)

        cov = np.cov(a, b)[0, 1]
        var_b = np.var(b)

        if var_b == 0 or np.isnan(var_b) or np.isnan(cov):
             beta = 0.0
        else:
             beta = cov / var_b

        spread = a - beta * b

        mean = np.mean(spread)
        std = np.std(spread)

        if std == 0 or np.isnan(std):
             return np.zeros_like(spread)

        z = (spread - mean) / std
            return z