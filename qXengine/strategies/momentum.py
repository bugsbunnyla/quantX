from ..StrategyResult import StrategyResult
from ..strategies.BaseStrategy import BaseStrategy

import numpy as np
import pandas as pd

# UMD - winners and losers, quantile based and not raw scores trading 
# cross sectional ranking Timestamp
# rank top 20% long, bottom 20% short middle flat
# institutional data driven
# implementation = rank assets each day → pick top 20% long / bottom 20% short
#Correct UMD math (fully fixed) wi,t={+1/nL	​−1/nS	​winners	​losers	,​rt​=∑wi,t * ​ri,t  ,	​Ct​=∏(1+rt​)
# 1 Cross-sectional return input For each asset i at time t: ri,t(k)	​=Pi,t/Pi,t−k ​−1 In UMD you typically use 12–1 momentum: momentumi,t=ri,t252 −ri,t21
# 2 Cross-sectional ranking Convert raw momentum into percentile ranks: ranki,t​ = ​ rank(momentumi,t/Nt)  Where:Nt = number of assets at time t rank is ascending order
# So: 1.0 = best performer 0.0 = worst performer
#3 Portfolio selection rule  Define thresholds: qhigh=1−0.2=0.8 qlow	​=0.2 Then: Long set Lt={i:ranki,t	​≥qhigh	​}  Short set St={i:ranki,t	​≤qlow	​}
#4 Position assignment Basic equal-weight version: wi,t	​=⎩⎨⎧	​+1/∣Lt​∣,  1/−∣St∣	​​,0,	​i∈Lt	​i∈St	​otherwise	​
#5 Daily portfolio return Let ri,t+1	 be next-day return:  rtUMD	​=i=1∑N	​wi,t *	​⋅ri,t+1 This is the true factor return series.
#6 Optional volatility scaling (recommended)  To stabilize equity curve:r~t	​=rt/σt	​	​or target volatility:rtscaled	​=rt⋅σ∗/σt	​	​
#7 Cumulative 4-year curve (what you plot) This is what your chart should use:Ct	​=k=1∏t	​(1+rkUMD)  or log form:     Ct	​=k=1∑t	​log(1+rk)
# Concept	Code equivalent
#  rank		rank(axis=1, pct=True)
#  top 20%	>= 0.8
#  bottom 20%	<= 0.2
#  weights	equal-weight long/short
#  return	dot(weights, returns)
#  chart	cumulative sum/product
# Now UMD rt = Sigma wi,tri,t

# UMD - winners and losers, quantile based and not raw scores trading 
# cross sectional ranking Timestamp
# rank top 20% long, bottom 20% short middle flat
# institutional data driven
# implementation = rank assets each day → pick top 20% long / bottom 20% short
#Correct UMD math (fully fixed) wi,t={+1/nL	​−1/nS	​winners	​losers	,​rt​=∑wi,t * ​ri,t  ,	​Ct​=∏(1+rt​)
# 1 Cross-sectional return input For each asset i at time t: ri,t(k)	​=Pi,t/Pi,t−k ​−1 In UMD you typically use 12–1 momentum: momentumi,t=ri,t252 −ri,t21
# 2 Cross-sectional ranking Convert raw momentum into percentile ranks: ranki,t​ = ​ rank(momentumi,t/Nt)  Where:Nt = number of assets at time t rank is ascending order
# So: 1.0 = best performer 0.0 = worst performer
#3 Portfolio selection rule  Define thresholds: qhigh=1−0.2=0.8 qlow	​=0.2 Then: Long set Lt={i:ranki,t	​≥qhigh	​}  Short set St={i:ranki,t	​≤qlow	​}
#4 Position assignment Basic equal-weight version: wi,t	​=⎩⎨⎧	​+1/∣Lt​∣,  1/−∣St∣	​​,0,	​i∈Lt	​i∈St	​otherwise	​
#5 Daily portfolio return Let ri,t+1	 be next-day return:  rtUMD	​=i=1∑N	​wi,t *	​⋅ri,t+1 This is the true factor return series.
#6 Optional volatility scaling (recommended)  To stabilize equity curve:r~t	​=rt/σt	​	​or target volatility:rtscaled	​=rt⋅σ∗/σt	​	​
#7 Cumulative 4-year curve (what you plot) This is what your chart should use:Ct	​=k=1∏t	​(1+rkUMD)  or log form:     Ct	​=k=1∑t	​log(1+rk)
# Concept	Code equivalent
#  rank		rank(axis=1, pct=True)
#  top 20%	>= 0.8
#  bottom 20%	<= 0.2
#  weights	equal-weight long/short
#  return	dot(weights, returns)
#  chart	cumulative sum/product
# Now UMD rt = Sigma wi,tri,t

class UMDMomentum(BaseStrategy):

    # =========================================================
    # PRICE MATRIX BUILDER (CRITICAL FIX)
    # =========================================================
    def _build_prices(self):

        frames = {}

        for sym, df in self.data.items():

            if "close" not in df.columns:
                continue

            tmp = df.copy()

            if "date" in tmp.columns:

                tmp["date"] = pd.to_datetime(
                    tmp["date"],
                    errors="coerce"
                )

                tmp = tmp.dropna(subset=["date"])

                tmp = tmp[
                    tmp["date"] > pd.Timestamp("2000-01-01")
                ]

                tmp = tmp.sort_values("date")

                tmp = tmp.set_index("date")

            elif not isinstance(
                tmp.index,
                pd.DatetimeIndex
            ):
                continue

            frames[sym] = tmp["close"]

        if not frames:
            return pd.DataFrame()

        prices = pd.concat(
            frames,
            axis=1
        )

        prices = prices.sort_index()

        end = prices.index.max()
        start = end - pd.DateOffset(years=4)

        prices = prices.loc[start:end]

        return prices


    # =========================================================
    # CHARTDATA BUILDER
    # =========================================================
    def _build_chartdata(self, data_dict):

        df = pd.DataFrame(data_dict)

        if not isinstance(
            df.index,
            pd.DatetimeIndex
        ):
            raise ValueError(
                "chartdata index must be DatetimeIndex"
            )

        df = df.sort_index()

        df = df.replace(
            [np.inf, -np.inf],
            np.nan
        )

        return df


    # =========================================================
    # MAIN RUN
    # =========================================================
    def run(self):

        formation = self.cfg.get(
            "formation",
            252
        )

        skip = self.cfg.get(
            "skip_month",
            21
        )

        top_q = self.cfg.get(
            "top_quantile",
            0.20
        )

        bottom_q = self.cfg.get(
            "bottom_quantile",
            0.20
        )

        prices = self._build_prices()

        if (
            prices.empty
            or len(prices) < formation
        ):
            return StrategyResult(
                name="UMDMomentum",
                data=self.data,
                metrics={
                    "error":
                    "insufficient data"
                },
                signals={},
                chart=None
            )

        returns = (
            prices
            .pct_change()
            .fillna(0)
        )

        formation_ret = (
            prices
            .pct_change(formation)
        )

        skip_ret = (
            prices
            .pct_change(skip)
        )

        raw_momentum = (
            formation_ret
            - skip_ret
        )

        ranks = raw_momentum.rank(
            axis=1,
            pct=True
        )

        portfolio_returns = []

        for t in prices.index:

            if t not in ranks.index:
                portfolio_returns.append(0)
                continue

            row_rank = ranks.loc[t]
            row_ret = returns.loc[t]

            long_mask = (
                row_rank >=
                (1 - top_q)
            )

            short_mask = (
                row_rank <=
                bottom_q
            )

            n_long = long_mask.sum()
            n_short = short_mask.sum()

            if (
                n_long == 0
                or n_short == 0
            ):
                portfolio_returns.append(0)
                continue

            w = pd.Series(
                0.0,
                index=row_rank.index
            )

            w[long_mask] = (
                1.0 / n_long
            )

            w[short_mask] = (
                -1.0 / n_short
            )

            portfolio_returns.append(
                (w * row_ret).sum()
            )

        portfolio_returns = pd.Series(
            portfolio_returns,
            index=prices.index
        )

        portfolio_curve = (
            1 + portfolio_returns
        ).cumprod()

        benchmark = None

        if "SPY" in prices.columns:

            benchmark = (
                1
                + prices["SPY"]
                .pct_change()
                .fillna(0)
            ).cumprod()

            benchmark = benchmark.reindex(
                portfolio_curve.index
            ).ffill()

        chartdata = self._build_chartdata({

            "portfolio":
                portfolio_curve,

            "benchmark":
                benchmark

        })

        chart = self.build_chart(

            chartdata=chartdata,

            series=self.cfg["chart"]["series"],

            title=self.cfg["chart"]["title"],

            charttype=self.cfg["chart"]["type"],

            chartmode="line"
        )

        metrics = {

            "formation":
                formation,

            "skip_month":
                skip,

            "top_quantile":
                top_q,

            "bottom_quantile":
                bottom_q,

            "assets":
                len(prices.columns),

            "return":
                float(
                    portfolio_curve.iloc[-1] - 1
                )
        }

        return StrategyResult(

            name="UMDMomentum",

            data=self.data,

            metrics=metrics,

            signals={},

            chart=chart
        )
# strategies/time_series_momentum.py
# Trend Signal - timeseries TSMOM
# risk normalization and volatility targeting
import numpy as np
import pandas as pd
from .BaseStrategy import BaseStrategy
from ..StrategyResult import StrategyResult
#
#
#
#
#
#
#  Feature		UMD				TimeSeriesMomentum
#structure		cross-sectional			time-series
#signal			ranks				continuous signal
#portfolio		long/short basket		per-asset trend exposure
#chart type		multi-line equity curve	        trend line + benchmark
#markers			rebalance points		regime switches (optional)
#
import numpy as np
import pandas as pd
from .BaseStrategy import BaseStrategy
from ..StrategyResult import StrategyResult
#| Strategy           | Chart Type             | Output                         |
#| ------------------ | ---------------------- | ------------------------------ |
#| UMD                | portfolio equity curve | long/short basket performance  |
#| TimeSeriesMomentum | signal trend curve     | directional exposure over time |
# Formula 
# Momentum Mt	​=	​Pt/Pt−k	​	​−1
# Volatility σt	​=std(rt,63)
# Risk-adjusted signal      St	​=Mt/σt+ϵ	​
# Vol targeting 	St	​=St	​⋅θ
#
import numpy as np
import pandas as pd
from .BaseStrategy import BaseStrategy
from ..StrategyResult import StrategyResult

import numpy as np
import pandas as pd

from .BaseStrategy import BaseStrategy
from ..StrategyResult import StrategyResult


class TimeSeriesMomentum(BaseStrategy):

    # =========================================================
    # PRICE CLEANER
    # =========================================================
    def _prepare_symbol(self, df):

        tmp = df.copy()

        if "date" in tmp.columns:

            tmp["date"] = pd.to_datetime(
                tmp["date"],
                errors="coerce"
            )

            tmp = tmp.dropna(
                subset=["date"]
            )

            tmp = tmp[
                tmp["date"] >
                pd.Timestamp("2000-01-01")
            ]

            tmp = tmp.sort_values(
                "date"
            )

            tmp = tmp.set_index(
                "date"
            )

        tmp = tmp.sort_index()

        return tmp


    # =========================================================
    # CHARTDATA
    # =========================================================
    def _build_chartdata(self, data_dict):

        df = pd.DataFrame(data_dict)

        if not isinstance(
            df.index,
            pd.DatetimeIndex
        ):
            raise ValueError(
                "chartdata index must be DatetimeIndex"
            )

        df = df.sort_index()

        df = df.replace(
            [np.inf, -np.inf],
            np.nan
        )

        return df


    # =========================================================
    # MAIN RUN
    # =========================================================
    def run(self):

        formation = self.cfg.get(
            "formation",
            252
        )

        vol_target = self.cfg.get(
            "vol_target",
            0.15
        )

        lookback_vol = self.cfg.get(
            "lookback_vol",
            63
        )

        signals = {}

        for sym, df in self.data.items():

            if "close" not in df.columns:
                continue

            df = self._prepare_symbol(df)

            ret = (
                df["close"]
                .pct_change()
            )

            mom = (
                df["close"]
                .pct_change(
                    formation
                )
            )

            vol = (
                ret
                .rolling(
                    lookback_vol
                )
                .std()
            )

            signal = (
                mom /
                (vol + 1e-8)
            ) * vol_target

            signal = signal.fillna(0)

            signals[sym] = signal

        if not signals:

            return StrategyResult(
                name="TimeSeriesMomentum",
                data=self.data,
                metrics={
                    "error":
                    "no signals"
                },
                signals={},
                chart=None
            )

        signal_df = pd.DataFrame(
            signals
        ).fillna(0)

        signal_df = signal_df.sort_index()

        end = signal_df.index.max()
        start = end - pd.DateOffset(
            years=4
        )

        signal_df = signal_df.loc[
            start:end
        ]

        portfolio_signal = (
            signal_df.mean(axis=1)
        )

        benchmark = None

        if "SPY" in self.data:

            spy = self._prepare_symbol(
                self.data["SPY"]
            )

            benchmark = (
                spy["close"]
                .pct_change()
                .cumsum()
            )

            benchmark = benchmark.reindex(
                portfolio_signal.index
            ).ffill()

        chartdata = self._build_chartdata({

            "signal":
                portfolio_signal,

            "benchmark":
                benchmark,

            "regime_switches":
                portfolio_signal
                .diff()
                .fillna(0)

        })

        chart = self.build_chart(

            chartdata=chartdata,

            series=self.cfg["chart"]["series"],

            title=self.cfg["chart"]["title"],

            charttype=self.cfg["chart"]["type"],

            chartmode="line"
        )

        metrics = {

            "formation":
                formation,

            "vol_target":
                vol_target,

            "lookback_vol":
                lookback_vol,

            "assets":
                len(signals)
        }

        return StrategyResult(

            name="TimeSeriesMomentum",

            data=self.data,

            metrics=metrics,

            signals=signals,

            chart=chart
        )