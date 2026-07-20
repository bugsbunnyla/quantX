
class Kelly:

    """
    KELLY FRACTION

    FORMULA:
        f* = μ / σ²
    """

    def size(self, mu, var):
       return mu / max(var, 1e-9)