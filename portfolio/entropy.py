import numpy as np

class Entropy:

    """
    DIVERSIFICATION ENTROPY

    FORMULA:
        H = - Σ w log(w)
    """

    #def compute(self, w):
    #    w = np.abs(w) / (np.sum(np.abs(w)) + 1e-9)
    #    return -np.sum(w * np.log(w + 1e-9))

    def compute(self, w): 
       w = np.asarray(w, dtype=float)

       total = np.sum(np.abs(w))
       if total <= 1e-12:
          return 0.0

       p = np.abs(w) / total

       p = p[p > 0]  # remove exact zeros

       return -np.sum(p * np.log(p))