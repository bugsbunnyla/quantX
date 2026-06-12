import numpy as np

class ExecutionCost:

    def slippage(self, vol):
        return 0.0001 * np.sqrt(np.abs(vol))

    def impact(self, w):
        return 0.0001 * np.sqrt(np.abs(w))