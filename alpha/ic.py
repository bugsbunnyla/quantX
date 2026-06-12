import numpy as np

class IC:

    """
    INFORMATION COEFFICIENT

    IC = corr(signal, forward_returns)
    """
#
# Correlation is: corr(x,y)=cov(x,y)/ σx σy
# It fails when:
# std(x) == 0 (constant series)
# std(y) == 0
# NaNs present
# empty or near-empty slices
# NumPy does NOT guard this.
#

#    def compute(self, signal, future_returns):
#        return np.corrcoef(signal, future_returns)[0,1]


    def compute(self, signal, future_returns):
      x = np.asarray(signal)
      y = np.asarray(future_returns)

      mask = np.isfinite(x) & np.isfinite(y)
      x, y = x[mask], y[mask]

      if x.size < 2:
         return 0.0

      x_std = x.std()
      y_std = y.std()

      if x_std == 0 or y_std == 0:
        return 0.0

      return np.cov(x, y, bias=True)[0, 1] / (x_std * y_std)