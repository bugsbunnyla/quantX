import numpy as np

class Liquidity:

    """
    LIQUIDITY MODEL

    FORMULA:
        cost ∝ 1 / volume
    """

    def adjust(self, vol, volume):
        return vol / (volume + 1e-9)