# ===================================================================================
# IntradayReversal : intraday mean-reversion, liquidity shock model in Quant Xpert
# Date: 2026/06/24 
# Author : bugsbunnyla
# Comment : volume and volatility with regime as a reversal signal
# applied strategy processing in Quant Xpert
# “Has price overextended relative to its short-term equilibrium?”
# Signal=−zscore(price−MA)×volatility_regime+volume_z
# ===================================================================================

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

    # ==================================================
    # SAFE NORMALIZER (CRITICAL FIX)
    # ==================================================
    def _normalize_df(self, df):

        df = df.copy()

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
            df = df.set_index("date")
        else:
            df.index = pd.to_datetime(df.index, errors="coerce")

        df = df[~df.index.isna()]
        df = df.sort_index()

        return df


    # ==================================================
    # MAIN
    # ==================================================
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

        # ==================================================
        # STEP 1: NORMALIZE ALL INPUT DATA
        # ==================================================
        clean = {}

        for sym, df in self.data.items():
            if "close" not in df.columns:
                continue
            clean[sym] = self._normalize_df(df)

        # ==================================================
        # STEP 2: MASTER PRICE MATRIX (CRITICAL FIX)
        # ==================================================
        prices = pd.DataFrame({
            sym: df["close"]
            for sym, df in clean.items()
        }).sort_index()

        #print("\n[DEBUG STREV prices.index]")
        #print(prices.index[:5])
        #print("dtype:", prices.index.dtype)

        if prices.empty:
            return StrategyResult(
                name="STREV",
                data={},
                metrics={"error": "No data"},
                signals={},
                chart=None
            )

        # ==================================================
        # RETURNS
        # ==================================================
        returns = prices.pct_change()

        # ==================================================
        # SIGNAL STORAGE
        # ==================================================
        signals = {}
        zscores = {}

        # ==================================================
        # BUILD SIGNALS (UNCHANGED LOGIC)
        # ==================================================
        for sym, df in clean.items():

            df = df.reindex(prices.index).ffill()

            reversal_return = df["close"].pct_change(lookback)

            mean_ret = reversal_return.rolling(zscore_window).mean()
            std_ret = reversal_return.rolling(zscore_window).std()

            zscore = (reversal_return - mean_ret) / (std_ret + 1e-8)

            signal = -zscore.fillna(0)

            signals[sym] = signal
            zscores[sym] = zscore.fillna(0)

        # ==================================================
        # SIGNAL MATRIX (FIXED INDEX)
        # ==================================================
        signal_df = pd.DataFrame(signals).fillna(0)
        signal_df = signal_df.reindex(prices.index).fillna(0)

        signal_curve = signal_df.mean(axis=1)

        # ==================================================
        # PORTFOLIO RETURNS
        # ==================================================
        portfolio_returns = []

        for dt in prices.index:

            pnl = []

            for sym, df in clean.items():

                if dt not in df.index:
                    continue

                loc = df.index.get_loc(dt)

                if loc + holding >= len(df):
                    continue

                future_ret = (
                    df["close"].iloc[loc + holding]
                    / df["close"].iloc[loc]
                    - 1
                )

                pnl.append(signals[sym].loc[dt] * future_ret)

            portfolio_returns.append(np.mean(pnl) if pnl else 0)

        portfolio_returns = pd.Series(
            portfolio_returns,
            index=prices.index
        )

        portfolio_curve = (1 + portfolio_returns.fillna(0)).cumprod()

        # ==================================================
        # BENCHMARK
        # ==================================================
        benchmark_curve = None

        if benchmark in prices.columns:

            benchmark_curve = (1 + prices[benchmark].pct_change().fillna(0)).cumprod()
            benchmark_curve = benchmark_curve.reindex(prices.index).ffill()

        # ==================================================
        # ENTRY/EXIT EVENTS
        # ==================================================
        entry_exit_events = np.sign(signal_curve).diff().fillna(0)

        # ==================================================
        #  DEBUG BEFORE CHARTDATA
        # ==================================================
        #print("\n[DEBUG STREV BEFORE CHARTDATA]")
        #print("portfolio_curve:", portfolio_curve.index[:5])
        #print("signal_curve:", signal_curve.index[:5])

        # ==================================================
        # CHARTDATA (CRITICAL FIX)
        # ==================================================
        chartdata = pd.DataFrame(index=prices.index)

        chartdata["portfolio_curve"] = portfolio_curve
        chartdata["signal_curve"] = signal_curve
        chartdata["benchmark"] = benchmark_curve
        chartdata["entry_exit_events"] = entry_exit_events

        chartdata = chartdata.replace([np.inf, -np.inf], np.nan).fillna(0)

        #print("\n[DEBUG FINAL STREV chartdata]")
        #print(chartdata.index[:5])
        #print("dtype:", chartdata.index.dtype)

        # ==================================================
        # CHART
        # ==================================================
        chart = self.build_chart(
            chartdata=chartdata,
            series=series_cfg,
            title=self.cfg.get("title"),
            charttype=chart_cfg.get("type", "time_series"),
            chartmode=chart_cfg.get("mode", "line+markers")
        )

        return StrategyResult(
            name="STREV",
            data=self.data,
            metrics={
                "lookback": lookback,
                "zscore_window": zscore_window,
                "holding": holding
            },
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
#
# Reversal signal: Zreversal=−P−MA/σ
#  Liquidity adjustment: Liquidity=Zvolume +Zvolatility ​
#   Final signal: Signal=Zreversal 	​×(1+Liquidity)
#
import numpy as np
import pandas as pd

from ..StrategyResult import StrategyResult
from ..strategies.BaseStrategy import BaseStrategy
import numpy as np
import pandas as pd


class IntradayReversal(BaseStrategy):

    # ==================================================
    # INDEX NORMALIZER (SAFE + INTRADAY CORRECT)
    # ==================================================
    def _normalize(self, df):

        df = df.copy()

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
            df = df.set_index("date")
        else:
            df.index = pd.to_datetime(df.index, errors="coerce")

        df = df[df.index.notna()]
        return df.sort_index()

    # ==================================================
    # CHART BUILDER
    # ==================================================
    def build_chart(
        self,
        chartdata=None,
        charttype=None,
        chartmode=None,
        title=None,
        xaxis=None,
        yaxis=None,
        series=None,
        chartcfg=None
    ):

        return StrategyChart(
            charttype=charttype,
            chartmode=chartmode,
            title=title,
            xaxis=xaxis,
            yaxis=yaxis,
            chartdata=chartdata,
            series=series or []
        )

    # ==================================================
    # MAIN STRATEGY
    # ==================================================
    def run(self):

        cfg = self.cfg

        lookback = cfg["lookback"]
        vol_window = cfg["volume_window"]
        threshold = cfg["threshold"]

        charts = cfg.get("chart", [])
        active_charts = [c for c in charts if c.get("enabled", False)]

        signal_map = {}

        frames = []

        # ==================================================
        # PER SYMBOL PROCESSING (NO CROSS-LEAKAGE)
        # ==================================================
        for sym, df in self.data.items():

            if "close" not in df.columns:
                continue

            df = self._normalize(df)

            if len(df) < max(lookback, vol_window):
                continue

            # --------------------------------------------------
            # RETURNS
            # --------------------------------------------------
            df["ret"] = df["close"].pct_change()

            # --------------------------------------------------
            # TRUE MEAN REVERSION (CORE FIX)
            # --------------------------------------------------
            ma = df["close"].rolling(lookback).mean()
            std = df["close"].rolling(lookback).std()

            reversal_raw = (df["close"] - ma) / (std + 1e-8)

            # invert for mean reversion signal
            reversal = -reversal_raw

            # --------------------------------------------------
            # VOLATILITY REGIME (FIXED Z-SCORE)
            # --------------------------------------------------
            volatility = df["ret"].rolling(vol_window).std()

            vol_mean = volatility.rolling(50).mean()
            vol_std = volatility.rolling(50).std()

            vol_z = (volatility - vol_mean) / (vol_std + 1e-8)

            # volatility acts as regime amplifier (not divisor)
            vol_regime = 1 + vol_z.fillna(0)

            z_vol = reversal * vol_regime

            # --------------------------------------------------
            # VOLUME SIGNAL (CLEAN Z-SCORE)
            # --------------------------------------------------
            volume_mean = df["volume"].rolling(vol_window).mean()
            volume_std = df["volume"].rolling(vol_window).std()

            volume_z = (df["volume"] - volume_mean) / (volume_std + 1e-8)

            # combine (stable additive model)
            z_volume = reversal + volume_z.fillna(0)

            # --------------------------------------------------
            # FINAL SIGNAL (ROBUST COMBINATION)
            # --------------------------------------------------
            final_signal = 0.6 * z_vol + 0.4 * z_volume

            final_signal = final_signal.replace([np.inf, -np.inf], np.nan).fillna(0)

            # --------------------------------------------------
            # EVENTS (STANDARDIZED THRESHOLDING)
            # --------------------------------------------------
            event_score = final_signal / (final_signal.rolling(50).std() + 1e-8)

            df["reversal_event"] = np.abs(event_score) > threshold

            # --------------------------------------------------
            # OUTPUT SERIES FOR CHARTING
            # --------------------------------------------------
            df["z_vol"] = z_vol
            df["z_volume"] = z_volume
            df["volatility"] = volatility
            df["reversal_event_vol"] = df["reversal_event"]
            df["reversal_event_volume"] = df["reversal_event"]

            # --------------------------------------------------
            # SIGNAL MAP (KEEP MAGNITUDE - FIXED)
            # --------------------------------------------------
            signal_map[sym] = final_signal.ewm(span=lookback).mean()

            frames.append(df)

        # ==================================================
        # RETURN EMPTY SAFELY
        # ==================================================
        if not frames:
            return StrategyResult(
                name=self.__class__.__name__,
                data=self.data,
                metrics=None,
                signals = None,
                chart=None
            )

        # NOTE:
        # We DO NOT merge across symbols for signal logic
        # (avoids cross-sectional leakage)

        # ==================================================
        # CHART SERIES SELECTION
        # ==================================================
        seen = set()
        series = []

        for c in active_charts:
            for s in c.get("series", []):

                source = s.get("source")

                if source in seen:
                    continue

                seen.add(source)
                series.append(s)

        # ==================================================
        # BUILD SINGLE CHART (CONFIG DRIVEN)
        # ==================================================
        chart = self.build_chart(
            charttype="line",
            chartmode="lines",
            title=cfg.get("title", "Intraday Reversal"),
            xaxis="date",
            yaxis="value",
            chartdata=pd.concat(frames).sort_index(),
            series=series
        )

        # ==================================================
        # RESULT
        # ==================================================
        return StrategyResult(
            name=self.__class__.__name__,
            data=self.data,
            metrics={
                "lookback": lookback,
                "volume_window": vol_window,
                "threshold": threshold,
                "symbols": len(self.data)
            },
            signals=signal_map,
            chart=chart
        )

# =======================================================================
# END OF INTRADAY REVERSAL
# =======================================================================