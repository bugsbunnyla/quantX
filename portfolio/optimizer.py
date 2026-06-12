import numpy as np


#
# simple signal-to-risk weighting Formula  wi ∝ μi /σi
# Portfolio optimization requires a vector of expected returns and a vector of volatilities:
# μ=[ μBTC ​μETH ​] σ=[ σBTC  σETH​] Then: wi= (μi/σi )/ ∑j ​∣μj/σj∣
# For a single asset - w= (μ/σ)/∣μ/σ∣
# Mean Variance Optimizer Formula wi​∝​μi​​/ σi**2   --> pass variance in denominator
#
class Portfolio:

    #def weights(self, mu, vol):
    #   w = mu / (vol + 1e-9)
    #  return w / (np.sum(np.abs(w)) + 1e-9)
    def weights(self, mu, vol):
        mu = np.asarray(mu, dtype=float)
        vol = np.asarray(vol, dtype=float)

        score = mu / np.maximum(vol, 1e-12)

        denom = np.sum(np.abs(score))

        if denom <= 1e-12:
           return np.zeros_like(score)

        return score / denom