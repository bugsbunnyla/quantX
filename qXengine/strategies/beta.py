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

    def run(self):

        # ==================================================
        # CONFIG
        # ==================================================

        benchmark = self.get_cfg(
            "benchmark",
            self.runtime_cfg.get("benchmark", "SPY")
        )

        beta_window = self.cfg.get("beta_window", 63)

        chart_cfg = self.get_cfg("chart", {})
        series_cfg = chart_cfg.get("series", [])

        # ==================================================
        # VALIDATION
        # ==================================================

        if benchmark not in self.data:
            return StrategyResult(
                name="BetaNeutralStrategy",
                data={},
                metrics={
                    "error": f"Benchmark '{benchmark}' missing"
                },
                signals={},
                chart=None
            )

        benchmark_ret = self.data[benchmark]["ret"].fillna(0)

        bn_returns = {}
        beta_stats = {}

        # ==================================================
        # BUILD BETA-NEUTRAL RETURNS
        # ==================================================

        for sym, df in self.data.items():

            if sym == benchmark:
                continue

            if "ret" not in df.columns:
                continue

            asset_ret = df["ret"].fillna(0)

            n = min(
                len(asset_ret),
                len(benchmark_ret)
            )

            if n < beta_window:
                continue

            x = benchmark_ret.iloc[-n:]
            y = asset_ret.iloc[-n:]

            bn_series = []
            beta_series = []

            for i in range(beta_window, n):

                x_win = x.iloc[i - beta_window:i]
                y_win = y.iloc[i - beta_window:i]

                beta = linregress(
                    x_win,
                    y_win
                ).slope

                beta_series.append(beta)

                # beta-neutral return
                r_bn = y.iloc[i] - beta * x.iloc[i]

                bn_series.append(r_bn)

            idx = y.iloc[beta_window:].index

            bn_series = pd.Series(
                bn_series,
                index=idx
            )

            # normalize for comparability
            bn_series = (
                bn_series /
                (bn_series.std() + 1e-8)
            )

            bn_returns[sym] = bn_series

            beta_stats[sym] = {
                "mean_beta": float(np.mean(beta_series)),
                "beta_vol": float(np.std(beta_series)),
                "observations": len(beta_series)
            }

        # ==================================================
        # EMPTY UNIVERSE CHECK
        # ==================================================

        if not bn_returns:
            return StrategyResult(
                name="BetaNeutralStrategy",
                data={},
                metrics={
                    "error": "No valid assets produced beta-neutral returns"
                },
                signals={},
                chart=None
            )

        # ==================================================
        # PORTFOLIO CONSTRUCTION
        # ==================================================

        aligned = pd.concat(
            bn_returns,
            axis=1
        ).fillna(0)

        vol = aligned.std() + 1e-8

        weights = 1.0 / vol
        weights = weights / weights.sum()

        portfolio = aligned.dot(weights)

        # cumulative 4Y strategy curve
        portfolio_curve = portfolio.cumsum()

        # ==================================================
        # ALIGN BENCHMARK TO PORTFOLIO INDEX
        # ==================================================

        benchmark_series = (
            benchmark_ret
            .reindex(portfolio_curve.index)
            .ffill()
        )

        # ==================================================
        # CHART DATA
        # ==================================================

        chartdata = {
            "pnl": portfolio_curve,
            "benchmark": benchmark_series
        }

        chart = self.build_chart(
            charttype=chart_cfg.get("type", "line"),
            chartmode="line+markers",
            title=chart_cfg.get(
                "title",
                "Beta Neutral Strategy (4Y Performance)"
            ),
            chartdata=chartdata,
            series=series_cfg
        )

        # ==================================================
        # RESULT
        # ==================================================

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

            chart=chart
        )