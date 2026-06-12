import numpy as np

class TSAlpha:
    """
    MOMENTUM MODEL

    FORMULA:
        MOM_t = Σ r_{t-i} / n
    """

    def momentum(self, r):
        return r.rolling(5).mean().fillna(0)