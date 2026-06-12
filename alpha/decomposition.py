import numpy as np
#class Decompose:
#    """
#    PURE STRATEGY ALPHA

#    FORMULA:
#        α_pure = TS_alpha + XS_alpha - noise
#        IR = mean(α) / std(α)
#    """

#    def purify(self, ts, xs):
#        return ts - xs.mean()

##class Decompose:
#    """
#    PURE STRATEGY ALPHA

#    α_pure = TS_alpha + XS_alpha - noise
#    IR = mean(α) / std(α)
#    """

#    def purify(self, ts, xs):
#        ts = np.asarray(ts)
#        xs = np.asarray(xs)

#        ts = np.nan_to_num(ts)
#        xs = np.nan_to_num(xs)

#        return ts + xs - xs.mean()

#    def information_ratio(self, alpha):
#        alpha = np.asarray(alpha)

#        mean = np.mean(alpha)
#        std = np.std(alpha)

#        if std == 0 or np.isnan(std):
#            return 0

#        return mean / std

import numpy as np
import pandas as pd

class Decompose:

    def purify(self, ts, xs):
        # Preserve original index if ts is a Series
        index = getattr(ts, "index", None)

        ts = np.asarray(ts)
        xs = np.asarray(xs)

        # Ensure ts is 1D
        ts = np.squeeze(ts)

        # Compute noise over all non-time dimensions
        if xs.ndim == 1:
            noise = np.nanmean(xs)
        else:
            noise = np.nanmean(xs, axis=tuple(range(1, xs.ndim)))

        # Formula: ts + xs - noise
        # To return a Series, reduce xs to the same shape as ts
        if xs.ndim == 1:
            xs_term = xs
        else:
            xs_term = np.nanmean(xs, axis=tuple(range(1, xs.ndim)))

        pure = ts + xs_term - noise

        # Clean invalid results
        pure = np.nan_to_num(pure, nan=0.0, posinf=0.0, neginf=0.0)

        return pd.Series(pure, index=index)