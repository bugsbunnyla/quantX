class Pairs:
    """
    PAIRS TRADING

    FORMULA:
        spread = P_A - P_B
    """

    def spread(self, p):
        return p.iloc[:,0] - p.iloc[:,1]