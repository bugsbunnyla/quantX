class Wiggle:
    """
    WIGGLE CO-MOVEMENT

    FORMULA:
        ΔCorr_t = Corr_t - Corr_{t-1}
    """

    def compute(self, r):
        return r.corr().diff().fillna(0).mean().mean()