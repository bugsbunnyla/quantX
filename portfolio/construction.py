import numpy as np

class Portfolio:
    """
    PORTFOLIO CONSTRUCTION

    FORMULA:
        w = μ / σ
        normalized weights
    """

    def weights(self, mu, vol):
        w = mu / (vol + 1e-9)
        return w / (np.sum(np.abs(w)) + 1e-9)