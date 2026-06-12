import numpy as np

class Decorrelate:
    """
    DE-CORRELATION ENGINE

    FORMULA:
        anti_corr_score = 1 / mean(|corr|)
    """

    def score(self, corr):
        return 1 / (np.mean(np.abs(corr.values)) + 1e-9)