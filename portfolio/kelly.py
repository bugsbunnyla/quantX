
class Kelly:

    """
    KELLY FRACTION

    FORMULA:
        f* = μ / σ²
    """

    def size(self, mu, var):
       if var == 0:
          mu / (var + 1e-9)

       return mu / var