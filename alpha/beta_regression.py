import numpy as np

class AlphaBeta:

    """
    REGRESSION MODEL

    MODEL:
        r_i = α + β * r_m + ε

    WHERE:
        α = alpha (excess return)
        β = sensitivity to market
        ε = idiosyncratic return
    """

    #def fit(self, r_asset, r_market):

    #    x = r_market.values
    #    y = r_asset.values

    #    beta = np.cov(x, y)[0,1] / (np.var(x) + 1e-9)
    #    alpha = np.mean(y) - beta * np.mean(x)

    #    residual = y - (alpha + beta * x)

    #    return alpha, beta, residual

    def fit(self, y, x):

        mask = y.notna() & x.notna()
        y = y[mask]
        x = x[mask]

        x_mean = x.mean()
        y_mean = y.mean()
        eps = 1e-12
        cov = ((x - x_mean) * (y - y_mean)).mean()
        var = ((x - x_mean) ** 2).mean()

        # SAFE VAR PROTECTION
        if not np.isfinite(var) or var < eps:
            beta = 0.0
            alpha = y_mean
            residual = y - alpha
            return alpha, beta, residual

        beta = cov / var
        alpha = y_mean - beta * x_mean
        residual = y - (alpha + beta * x)
        if np.std(residual) < 1e-6:
             print("WARNING: near-deterministic factor model")

        return alpha, beta, residual