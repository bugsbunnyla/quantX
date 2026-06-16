import numpy as np
import pandas as pd

from scipy.stats import linregress
from ..StrategyResult import StrategyResult
from ..strategies.BaseStrategy import BaseStrategy
#
#  rolling beta estimator = beta_t = Cov(asset, market) / Var(market)
#  residual extractor = alpha = asset - beta * market
#  standardization = z-score residual → comparable across assets
# Addres pure alpha αt​=rportfolio,t​−rSPY,t​
# 1rolling beta βi,t	​=Var(rm )Cov(ri	​,rm)
# 2beta-neutral return (correct time series) ri,tBN	​=ri,t	​−βi,t * ​rm,t	​
# 3portfolio aggregation Instead of mean():correct weighting: wi=1/σi
# 4cumulative 4Y curve (for chart)Ct​=k≤t∑​Sk​
import numpy as np
import pandas as pd

from scipy.stats import linregress
from ..StrategyResult import StrategyResult
from ..strategies.BaseStrategy import BaseStrategy


class BetaNeutralStrategy(BaseStrategy):

    # ==================================================
    # 1. FORCE DATETIME INDEX (INTRADAY PATTERN)
    # ==================================================
    def _prepare_data(self):

        cleaned = {}

        for sym, df in self.data.items():

            if "ret" not in df.columns:
                continue

            tmp = df.copy()

            # ---- enforce datetime index ----
            if "date" in tmp.columns:
                tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
                tmp = tmp.dropna(subset=["date"])
                tmp = tmp[tmp["date"] > pd.Timestamp("2000-01-01")]
                tmp = tmp.sort_values("date")
                tmp = tmp.set_index("date")

            else:
                tmp.index = pd.to_datetime(tmp.index, errors="coerce")
                tmp = tmp[tmp.index.notna()]
                tmp = tmp[tmp.index > pd.Timestamp("2000-01-01")]
                tmp = tmp.sort_index()

            cleaned[sym] = tmp

        return cleaned

    # ==================================================
    # 2. FORCE CLEAN OUTPUT (CRITICAL FIX)
    # ==================================================
    def _clean_series(self, s):

        if s is None:
            return None

        s = pd.Series(s).copy()

        s.index = pd.to_datetime(s.index, errors="coerce")
        s = s[s.index.notna()]
        s = s[s.index > pd.Timestamp("2000-01-01")]
        s = s.sort_index()
        s = s[~s.index.duplicated(keep="last")]

        return s

    # ==================================================
    # 3. FORCE 4Y WINDOW (INTRADAY STYLE)
    # ==================================================
    def _last_4y(self, s):

        if s is None:
            return None

        s = self._clean_series(s)

        if s is None:
            return None

        return s.tail(252 * 4)

    # ==================================================
    def run(self):

        benchmark = self.get_cfg(
            "benchmark",
            self.runtime_cfg.get("benchmark", "SPY")
        )

        beta_window = self.cfg.get("beta_window", 63)

        data = self._prepare_data()

        if benchmark not in data:
            return StrategyResult(
                name="BetaNeutralStrategy",
                data={},
                metrics={"error": f"Benchmark '{benchmark}' missing"},
                signals={},
                chart=None
            )

        benchmark_ret = data[benchmark]["ret"].fillna(0)

        bn_returns = {}
        beta_stats = {}

        # ==================================================
        # CORE LOOP (UNCHANGED FORMULA, FIXED TIME)
        # ==================================================
        for sym, df in data.items():

            if sym == benchmark:
                continue

            asset_ret = df["ret"].fillna(0)

            n = min(len(asset_ret), len(benchmark_ret))

            if n < beta_window:
                continue

            x = benchmark_ret.iloc[-n:]
            y = asset_ret.iloc[-n:]

            bn_series = []
            beta_series = []

            for i in range(beta_window, n):

                x_win = x.iloc[i - beta_window:i]
                y_win = y.iloc[i - beta_window:i]

                beta = linregress(x_win, y_win).slope
                beta_series.append(beta)

                bn_series.append(y.iloc[i] - beta * x.iloc[i])

            # ==================================================
            # 🔥 KEY FIX: PRESERVE TRUE TIME INDEX
            # ==================================================
            idx = y.iloc[beta_window:].index

            series = pd.Series(bn_series, index=idx)
            series = series / (series.std() + 1e-8)

            bn_returns[sym] = self._last_4y(series)

            beta_stats[sym] = {
                "mean_beta": float(np.mean(beta_series)),
                "beta_vol": float(np.std(beta_series)),
                "observations": len(beta_series)
            }

        # ==================================================
        # PORTFOLIO (TIME SAFE)
        # ==================================================
        aligned = pd.concat(bn_returns, axis=1).fillna(0)

        vol = aligned.std() + 1e-8
        weights = 1.0 / vol
        weights = weights / weights.sum()

        portfolio = aligned.dot(weights)
        portfolio_curve = self._last_4y(portfolio.cumsum())

        benchmark_series = self._last_4y(
            benchmark_ret.reindex(portfolio_curve.index).ffill()
        )

        chartdata = {
            "pnl": portfolio_curve,
            benchmark: benchmark_series
        }

        return StrategyResult(
            name="BetaNeutralStrategy",
            data=chartdata,
            metrics={
                "benchmark": benchmark,
                "assets": len(bn_returns),
                "beta_window": beta_window,
                "beta_stats": beta_stats,
                "portfolio_weights": weights.to_dict()
            },
            signals=bn_returns,
            chart=self.build_chart(
                charttype="line",
                chartmode="line",
                title="Beta Neutral Strategy (4Y Performance)",
                chartdata=chartdata,
                series=[]
            )
        )