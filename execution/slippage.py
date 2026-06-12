import numpy as np

class Slippage:
    """
    SLIPPAGE MODEL

    S = k * sqrt(vol)
    """

    def model(self, vol):
        return 0.0001 * np.sqrt(abs(vol))