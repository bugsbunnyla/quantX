from ..StrategyResult import StrategyResult
from ..strategies.BaseStrategy import BaseStrategy

import numpy as np
import pandas as pd
#
# STREV - reversal
# Core Strategy = Identify short-term overreaction and profit from mean reversion over a 5-day holding period.
# The core quant signal is:STREVt	​=−​(R20,t​−μ60)/σ60 where:R20,t = 20-day return μ60 = rolling mean return σ60 = rolling std return signal = negative z-score (contrarian)
# 
# Signalt	​=−Zt	​and Portfolio t	​=Signalt × FutureReturnt+5	​
# Not a volatility Adjusted Reveral and Remove Signal=−Rolling Volatility20D Return​
# Formulas used : 
#1.Lookback Return Rt=​(Pt/Pt−20  ​−1)
#2.Rolling Mean μt=Mean(Rt−60:t	​)
#3.Rolling Std σt=Std(Rt−60:t)
#4.z-Score Zt​=σt​Rt​−μt​​
#5.Reversal signal = Signalt​=−Zt​ Positive signal: oversold → buy Negative signal: overbought → short
#6.Portfolio Curve Portfoliot​=i∑​Signali,t ​× Returni,t+5	​then Curvet​=∏(1+Portfoliot​)
class STREV(BaseStrategy):

    def run(self):

        lookback = self.cfg.get("lookback", 20)
        zscore_window = self.cfg.get("zscore_window", 60)
        holding = self.cfg.get("holding", 5)

        chart_cfg = self.get_cfg("chart", {})
        series_cfg = chart_cfg.get("series", [])

        benchmark = self.get_cfg(
            "benchmark",
            self.runtime_cfg.get("benchmark", "SPY")
        )

        signals = {}
        zscores = {}

        # ==========================================
        # BUILD SIGNALS
        # ==========================================
        for sym, df in self.data.items():

            if "close" not in df.columns:
                continue

            df = df.copy()

            df["ret"] = df["close"].pct_change()

            #
            # STREV RETURN
            #
            reversal_return = df["close"].pct_change(lookback)

            #
            # TRUE Z-SCORE
            #
            mean_ret = reversal_return.rolling(zscore_window).mean()
            std_ret = reversal_return.rolling(zscore_window).std()

            zscore = (
                (reversal_return - mean_ret)
                /
                (std_ret + 1e-8)
            )

            #
            # CONTRARIAN SIGNAL
            #
            signal = -zscore

            signals[sym] = signal.fillna(0)
            zscores[sym] = zscore.fillna(0)

        if not signals:

            return StrategyResult(
                name="STREV",
                data={},
                metrics={"error": "No signals"},
                signals={},
                chart=None
            )

        # ==========================================
        # PORTFOLIO
        # ==========================================
        signal_df = pd.DataFrame(signals).fillna(0)

        signal_curve = signal_df.mean(axis=1)

        #
        # Forward returns
        #
        portfolio_returns = []

        common_index = signal_curve.index

        for dt in common_index:

            pnl = []

            for sym in signals:

                df = self.data[sym]

                if dt not in df.index:
                    continue

                loc = df.index.get_loc(dt)

                if loc + holding >= len(df):
                    continue

                future_ret = (
                    df["close"].iloc[loc + holding]
                    /
                    df["close"].iloc[loc]
                    - 1
                )

                pnl.append(
                    signals[sym].loc[dt] * future_ret
                )

            portfolio_returns.append(
                np.mean(pnl) if pnl else 0
            )

        portfolio_returns = pd.Series(
            portfolio_returns,
            index=common_index
        )

        portfolio_curve = (
            1 + portfolio_returns.fillna(0)
        ).cumprod()

        # ==========================================
        # BENCHMARK
        # ==========================================
        benchmark_curve = None

        if benchmark in self.data:

            benchmark_curve = (
                1
                +
                self.data[benchmark]["close"]
                .pct_change()
                .fillna(0)
            ).cumprod()

            benchmark_curve = benchmark_curve.reindex(
                portfolio_curve.index
            ).ffill()

        # ==========================================
        # ENTRY / EXIT EVENTS
        # ==========================================
        entry_exit_events = (
            np.sign(signal_curve)
            .diff()
            .fillna(0)
        )

        # ==========================================
        # CHART DATA
        # ==========================================
        chartdata = {
            "portfolio_curve": portfolio_curve,
            "signal_curve": signal_curve,
            "benchmark": benchmark_curve,
            "entry_exit_events": entry_exit_events
        }

        chart = self.build_chart(
            charttype=chart_cfg.get(
                "type",
                "time_series"
            ),
            chartmode=chart_cfg.get(
                "mode",
                "line+markers"
            ),
            title=chart_cfg.get(
                "title",
                "STREV Mean Reversion"
            ),
            chartdata=chartdata,
            series=series_cfg
        )

        metrics = {
            "lookback": lookback,
            "zscore_window": zscore_window,
            "holding": holding,
            "assets": len(signals),
            "total_return": float(
                portfolio_curve.iloc[-1] - 1
            )
        }

        return StrategyResult(
            name="STREV",

            data=chartdata,

            metrics=metrics,

            signals=signals,

            chart=chart
        )
       
# strategies/intraday_reversal.py

import numpy as np
import pandas as pd

from ..StrategyResult import StrategyResult
from ..strategies.BaseStrategy import BaseStrategy
from ..StrategyCharts import StrategyChart
# Remove return volality in intraday reversal vol = df["ret"].rolling(vol_window).std() , z = reversal / (vol + 1e-8)
# Rt	​=−(Pt/Pt−lookback  ​−1) Volatility filter:σt	​=Std(rett​,vol_window) Normalized reversal score:Zt = Rt/σt	​	​
# Trading rules -: Z > threshold   → Short (-1) , Z < -threshold  → Long (+1) Otherwise       → Flat (0)
# Feature			Volatility-Normalized (Current)				Volume-Driven (Proposed)
#Core idea			Reversal vs recent volatility				Reversal × volume shock
#Formula			Zt​=σret	​−Returnlookback	​			​	Scoret	​=Reversalt	​×VolumeZt
#Data required			Close prices only					Close + Volume
#Signal driver			Price deviation intensity				Liquidity + participation shock
#Market condition sensitivity	Works in all regimes					Strong in event-driven regimes
#Signal frequency		Medium–High						Low–Medium
#Signal 			quality	Broader, noisier				More selective, higher conviction
#Noise sensitivity		Moderate (vol smoothing helps)				Lower (filters quiet moves)
#Best suited for		General intraday mean reversion				Panic / earnings / news reversals
#False positives		Sideways volatility spikes				Low-volume fake moves filtered out
#Missed opportunities		None based on volume					Misses low-volume reversals
#Edge source			Statistical overreaction				Liquidity imbalance / capitulation
#Execution cost profile		Higher turnover						Lower turnover
#Interpretability		Pure statistical z-score				Market microstructure signal
#
# Trading Behavior Differences
#Scenario			Volatility Model		Volume Model
#High vol, low volume drift	May trigger trades		Likely ignored
#Earnings spike			Strong signal			Very strong signal
#Quiet consolidation		Can still trigger		Usually no signal
#Panic selloff			Triggered			Stronger confirmation
#News gap move			Triggered			Amplified signal
#
#Outcome			Volatility model = “everything mean-reverts eventually” Volume model = “only crowd-driven moves matter”
#A) Volatility-Normalized Reversal Ztvol	​=σret	​−Returnt,lookback 
#Required data:
#Close prices only
#No volume required
#Only returns + rolling std
#Data nature:
#Pure price-statistics model
#Self-contained
#Stable across assets
#B) Volume-Driven Reversal Ztvolume	​=Reversalt​⋅Zscore(Volumet)
#Required data:
#Close prices
#Volume (mandatory)
#Rolling volume stats
#Data nature:
#Market microstructure model
#Sensitive to participation/liquidity
#Breaks if volume missing or distorted
# 2. Conceptual Difference (this is the real divider)
#Dimension		Volatility Model			Volume Model
#Signal type		Statistical anomaly			Liquidity event
#Normalization		Price volatility			Market participation
#Market assumption	Mean reversion always exists		Reversal only matters when crowd participates
#Missing data tolerance	High					Low
#Stability		High					Medium
#Edge source		Price overshoot				Panic / capitulation
import numpy as np
import pandas as pd

from ..StrategyResult import StrategyResult
from ..strategies.BaseStrategy import BaseStrategy


class IntradayReversal(BaseStrategy):

    # =========================================================
    # BUILD CHART (STRICT CONTRACT)
    # =========================================================
    def build_chart(
        self,
        chartdata=None,
        charttype=None,
        chartmode=None,
        title=None,
        xaxis=None,
        yaxis=None,
        series=None
    ):

        chart_cfg = self.cfg.get("chart", [])
        chart_cfg = chart_cfg[0] if isinstance(chart_cfg, list) and chart_cfg else chart_cfg

        return StrategyChart(
            charttype=charttype,
            chartmode=chartmode,
            title=title,
            xaxis=xaxis or chart_cfg.get("xaxis", {}).get("source", "date"),
            yaxis=yaxis or chart_cfg.get("yaxis", {}).get("label", "value"),
            chartdata=chartdata,
            series=series or []
        )

    # =========================================================
    # MAIN STRATEGY
    # =========================================================
    def run(self):

        cfg = self.cfg

        charts = cfg["chart"]
        active_charts = [c for c in charts if c.get("enabled", False)]

        chartcfg = self.get_cfg("chart", [])

        lookback = cfg["lookback"]
        vol_window = cfg["volume_window"]
        threshold = cfg["threshold"]

        # =========================================================
        # BASE DATA OUTPUT (PER SYMBOL MERGED LATER)
        # =========================================================
        frames = []
        signals = {}

        for sym, df in self.data.items():

            if "close" not in df.columns:
                continue

            df = df.copy()
            df["symbol"] = sym

            df["ret"] = df["close"].pct_change()

            reversal = -df["close"].pct_change(lookback)
            volatility = df["ret"].rolling(vol_window).std()

            volume_z = (
                (df["volume"] - df["volume"].rolling(vol_window).mean())
                / (df["volume"].rolling(vol_window).std() + 1e-8)
            )

            z_vol = reversal / (volatility + 1e-8)
            z_volume = reversal * volume_z

            # =====================================================
            # FULL FEATURE SPACE (ALWAYS COMPLETE)
            # =====================================================
            df["z_vol"] = z_vol
            df["z_volume"] = z_volume
            df["volatility"] = volatility

            df["reversal_event_vol"] = np.abs(z_vol) > threshold
            df["reversal_event_volume"] = np.abs(z_volume) > threshold

            frames.append(df)

            # signal per symbol
            signals[sym] = pd.Series(
                np.sign(z_vol.fillna(0) + z_volume.fillna(0)),
                index=df.index
            )

        # =========================================================
        # MERGE ALL SYMBOLS INTO SINGLE CHART SPACE
        # =========================================================
        merged_df = pd.concat(frames)

        # =========================================================
        # SERIES (UNION OF ENABLED CHARTS)
        # =========================================================
        series = []
        for c in active_charts:
            series.extend(c.get("series", []))

        # =========================================================
        # AXIS RESOLUTION (SAFE)
        # =========================================================
        def resolve_axis(key, default):
            vals = [c.get(key) for c in active_charts if c.get(key)]
            if not vals:
                return default
            return vals[0] if all(v == vals[0] for v in vals) else default

        # =========================================================
        # BUILD FINAL CHART
        # =========================================================
        chart = self.build_chart(
            charttype=chartcfg[0]["type"] if chartcfg else "line",
            chartmode="lines+markers",
            title=cfg.get("title", "Intraday Reversal"),

            xaxis=resolve_axis("xaxis", "date"),
            yaxis=resolve_axis("yaxis", "value"),

            chartdata=merged_df,
            series=series
        )

        # =========================================================
        # RETURN RESULT
        # =========================================================
        return StrategyResult(
            name=self.__class__.__name__,
            data=self.data,
            metrics={
                "lookback": lookback,
                "volume_window": vol_window,
                "threshold": threshold,
                "active_charts": [c["name"] for c in active_charts]
            },
            signals=signals,
            chart=chart
        )