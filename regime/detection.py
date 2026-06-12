import numpy as np

class Regime:

    """
    REGIME DETECTION

    RULE:
        high vol → risk-on/off switching
    """

    def detect(self, r):

        vol = np.std(r)

        if vol > np.percentile(r, 75):
            return "HIGH_VOL"
        elif vol < np.percentile(r, 25):
            return "LOW_VOL"
        return "NORMAL"