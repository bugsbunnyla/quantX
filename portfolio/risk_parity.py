import numpy as np

class RiskParity:

    """
    RISK PARITY

    FORMULA:
        w_i ∝ 1 / σ_i
    Formula: wi = (1/σi) / (∑j 1/σj ) where σi is asset volatility.
    """

    #def weights(self, vol):

    #    inv = 1 / (vol + 1e-9)
    #    return inv / np.sum(inv)


    def weights(self,vol):
        vol_wt = np.asarray(vol, dtype=float)

        vol_wt = np.maximum(vol_wt, 1e-12)

        inv = 1.0 / vol_wt
        total = inv.sum()

        if total <= 0:
           return np.ones_like(vol) / len(vol_Wt)

        return inv / total