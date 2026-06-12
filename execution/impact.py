import numpy as np

class Impact:
    """
    MARKET IMPACT

    I = k * sqrt(position)
    """

    def model(self, w):
        return 0.0001 * np.sqrt(abs(w))