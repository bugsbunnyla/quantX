import numpy as np

class CVaR:

    """
    CONDITIONAL VALUE AT RISK

    FORMULA:
        CVaR = E[loss | loss > VaR]
        CVaRα​=E[r∣r≤VaRα​]

    """

    def compute(self, returns, alpha=0.05):

        var = np.quantile(returns, alpha)
        tail = returns[returns <= var]

        return np.mean(tail) if len(tail) > 0 else var

