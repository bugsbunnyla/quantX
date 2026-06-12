class XSAlpha:
    """
    XS ALPHA

    FORMULA:
        XS_i = r_i - mean(r_all)
    """

    def cross_section(self, r):
        return r.sub(r.mean(axis=1), axis=0).fillna(0)