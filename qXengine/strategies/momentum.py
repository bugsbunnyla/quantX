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
class UMDMomentum(BaseStrategy):

    def run(self):

        # ==================================================
        # CONFIG
        # ==================================================
        formation = self.cfg.get("formation", 252)
        skip = self.cfg.get("skip_month", 21)
        holding = self.cfg.get("holding", 21)

        top_q = self.cfg.get("top_quantile", 0.2)
        bottom_q = self.cfg.get("bottom_quantile", 0.2)

        chart_cfg = self.get_cfg("chart", {})
        series_cfg = chart_cfg.get("series", [])

        # ==================================================
        # PRICE MATRIX
        # ==================================================
        prices = pd.DataFrame({
            sym: df["close"]
            for sym, df in self.data.items()
            if "close" in df.columns
        })

        if prices.empty:
            return StrategyResult(
                name="UMDMomentum",
                data={},
                metrics={"error": "no price data"},
                signals={},
                chart=None
            )

        returns = prices.pct_change().fillna(0)

        # ==================================================
        # MOMENTUM (12-1 STYLE)
        # ==================================================
        formation_ret = prices.pct_change(formation)
        skip_ret = prices.pct_change(skip)

        raw_momentum = formation_ret - skip_ret

        # ==================================================
        # CROSS-SECTIONAL RANKING
        # ==================================================
        ranks = raw_momentum.rank(axis=1, pct=True)

        # ==================================================
        # PORTFOLIO BUILD
        # ==================================================
        portfolio_returns = []
        signals = {}

        for t in ranks.index:

            row_rank = ranks.loc[t]
            row_ret = returns.loc[t]

            long_mask = row_rank >= (1 - top_q)
            short_mask = row_rank <= bottom_q

            n_long = long_mask.sum()
            n_short = short_mask.sum()

            if n_long == 0 or n_short == 0:
                portfolio_returns.append(0.0)
                continue

            weights = pd.Series(0.0, index=row_rank.index)
            weights[long_mask] = 1.0 / n_long
            weights[short_mask] = -1.0 / n_short

            signals[t] = weights

            portfolio_returns.append((weights * row_ret).sum())

        portfolio_returns = pd.Series(
            portfolio_returns,
            index=ranks.index
        ).fillna(0.0)

        # ==================================================
        # EQUITY CURVE (4Y)
        # ==================================================
        portfolio_curve = (1 + portfolio_returns).cumprod()

        # ==================================================
        # BENCHMARK (SPY)
        # ==================================================
        benchmark_curve = None
        if "SPY" in prices.columns:
            spy_ret = prices["SPY"].pct_change().fillna(0)
            benchmark_curve = (1 + spy_ret).cumprod()
            benchmark_curve = benchmark_curve.reindex(portfolio_curve.index).ffill()

        # ==================================================
        # CHARTDATA (SOURCE OF TRUTH FOR CHART)
        # ==================================================
        chartdata = {
            "portfolio": portfolio_curve,
            "benchmark": benchmark_curve
        }

        # ==================================================
        # BUILD CHART OBJECT (CRITICAL FIX)
        # ==================================================
        chart = self.build_chart(
            charttype=chart_cfg.get("type", "line"),
            chartmode="line+markers",
            title=chart_cfg.get("title", "UMD Momentum Strategy"),
            chartdata=chartdata,
            series=series_cfg
        )

        # ==================================================
        # METRICS
        # ==================================================
        metrics = {
            "formation": formation,
            "skip_month": skip,
            "holding": holding,
            "top_quantile": top_q,
            "bottom_quantile": bottom_q,
            "assets": len(prices.columns),
            "total_return": float(portfolio_curve.iloc[-1] - 1)
        }

        # ==================================================
        # FINAL RESULT (NOW INCLUDES CHART OBJECT)
        # ==================================================
        return StrategyResult(
            name="UMDMomentum",

            data=chartdata,

            metrics=metrics,

            signals=signals,

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


class TimeSeriesMomentum(BaseStrategy):

    def run(self):

        formation = self.cfg.get("formation", 252)
        vol_target = self.cfg.get("vol_target", 0.15)
        lookback_vol = self.cfg.get("lookback_vol", 63)

        chart_cfg = self.get_cfg("chart", {})
        series_cfg = chart_cfg.get("series", [])

        signals = {}
        regime_switches = {}

        # ==================================================
        # BUILD SIGNALS
        # ==================================================
        for sym, df in self.data.items():

            if "close" not in df.columns:
                continue

            df = df.copy()
            df["ret"] = df["close"].pct_change()

            momentum = df["close"].pct_change(formation)
            vol = df["ret"].rolling(lookback_vol).std()

            signal = (momentum / (vol + 1e-8)) * vol_target

            signal = signal.fillna(0)

            signals[sym] = signal

            # regime detection (optional markers)
            regime_switches[sym] = (np.sign(signal).diff().fillna(0) != 0).astype(int)

        # ==================================================
        # PORTFOLIO AGGREGATION (IMPORTANT FIX)
        # ==================================================
        signal_df = pd.DataFrame(signals).fillna(0)

        portfolio_signal = signal_df.mean(axis=1)

        # ==================================================
        # BENCHMARK (SPY)
        # ==================================================
        benchmark = self.get_cfg(
            "benchmark",
            self.runtime_cfg.get("benchmark", "SPY")
        )

        benchmark_series = None
        if benchmark in self.data:
            benchmark_series = self.data[benchmark]["close"].pct_change().cumsum()

        # ==================================================
        # CHARTDATA (CFG DRIVEN)
        # ==================================================
        chartdata = {
            "signal": portfolio_signal,
            "benchmark": benchmark_series,
            "regime_switches": pd.Series(portfolio_signal).diff().fillna(0)
        }

        # ==================================================
        # CHART OBJECT (REQUIRED)
        # ==================================================
        chart = self.build_chart(
            charttype=chart_cfg.get("type", "time_series"),
            chartmode="line+markers",
            title=chart_cfg.get("title"),
            chartdata=chartdata,
            series=series_cfg
        )

        # ==================================================
        # METRICS
        # ==================================================
        metrics = {
            "formation": formation,
            "vol_target": vol_target,
            "lookback_vol": lookback_vol,
            "assets": len(signals)
        }

        # ==================================================
        # RESULT
        # ==================================================
        return StrategyResult(
            name="TimeSeriesMomentum",

            data=chartdata,

            metrics=metrics,

            signals=signals,

            chart=chart
        )