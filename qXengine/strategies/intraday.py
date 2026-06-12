import numpy as np
import pandas as pd
from ..StrategyResult import StrategyResult
from ..strategies.BaseStrategy import BaseStrategy

# F1 momentum and reversal blend = Straw​=Momt​+Revt​ with Mom_t = intraday momentum signal, Rev_t = intraday mean reversion signal
# pairing quant trend-following + mean-reversion hybrid
# Volatility normalization (Z-score style) scale rolling volatility
# F2 Stnorm = ​Straw/σt Where:σt=RollingStd(Straw,vol_window)
# Effect:turns raw signal into a risk-adjusted score prevents high-vol assets from dominating
# F3 Volume adjustment (liquidity weighting) If volume exists:Vtratio= Volumet/MA(Volume,volume_window)
# Stliq =Stnorm x ⋅Vtratio
# Effect: boosts signals when participation is above normal , suppresses low-liquidity / dead periods
# F4 Signal aggregation (lookback mean)Final Scalar Score  Score=1/N ∑T Stliq Where:N=lookback
#                                                                   t=T−N
# F5 Threshold filter Final Signal={0    if ∣Score∣<θ
#                                  {Score otherwise
# Where:\theta = \text{signal_threshold}	​
# Dual signal - momentum + reversal  with cross asset comparable zscore signal S/σ
# A volatility- and liquidity-adjusted hybrid momentum–reversion intraday factor with noise gating.
class IntradayStrategy(BaseStrategy):

    requires_factor_engine = False

    # -------------------------------------------------
    # MOMENTUM (F1 COMPONENT)
    # -------------------------------------------------
    def momentum(self, df, window=20):

        return df["close"].pct_change(window)

    # -------------------------------------------------
    # REVERSAL (F1 COMPONENT)
    # -------------------------------------------------
    def reversal(self, df, window=5):

        return -df["close"].pct_change(window)

    # -------------------------------------------------
    # VWAP DEVIATION (LIQUIDITY FACTOR)
    # -------------------------------------------------
    def vwap_deviation(self, df):

        if "volume" not in df.columns:
            return pd.Series(0.0, index=df.index)

        pv = df["close"] * df["volume"]
        vwap = pv.cumsum() / (df["volume"].cumsum() + 1e-8)

        return (df["close"] - vwap) / (vwap + 1e-8)

    # -------------------------------------------------
    # MAIN STRATEGY
    # -------------------------------------------------
    def run(self):

        cfg = self.cfg

        lookback = cfg.get("lookback", 20)
        vol_window = cfg.get("vol_window", 20)
        volume_window = cfg.get("volume_window", 20)
        threshold = cfg.get("signal_threshold", 1.5)

        signals_out = {}
        scores_out = {}

        volume_stress_out = {}
        dislocation_events_out = {}

        # -------------------------------------------------
        # SYMBOL LOOP
        # -------------------------------------------------
        for sym, df in self.data.items():

            if "close" not in df.columns:
                continue

            if len(df) < max(lookback, vol_window, volume_window):
                continue

            # ----------------------------
            # F1: Momentum + Reversal
            # ----------------------------
            mom = self.momentum(df, lookback)
            rev = self.reversal(df, max(2, lookback // 4))

            raw_signal = (mom + rev).fillna(0)

            # ----------------------------
            # F2: Volatility Normalization
            # ----------------------------
            vol = raw_signal.rolling(vol_window).std().replace(0, np.nan)
            norm_signal = raw_signal / (vol + 1e-8)

            # ----------------------------
            # F3: Volume Adjustment (FIXED)
            # ----------------------------
            if "volume" in df.columns:

                vol_ma = df["volume"].rolling(volume_window).mean()
                vol_ratio = df["volume"] / (vol_ma + 1e-8)

                norm_signal = norm_signal * vol_ratio

                volume_stress_out[sym] = vol_ratio.fillna(1.0)

            else:
                volume_stress_out[sym] = pd.Series(
                    1.0,
                    index=df.index
                )

            # ----------------------------
            # VWAP INTEGRATION (FIXED ADDITION)
            # ----------------------------
            vwap_dev = self.vwap_deviation(df)

            norm_signal = norm_signal + 0.5 * vwap_dev

            # ----------------------------
            # CLEAN SIGNAL
            # ----------------------------
            norm_signal = norm_signal.replace(
                [np.inf, -np.inf],
                np.nan
            ).fillna(0)

            # ----------------------------
            # F4: SCORE (LOOKBACK MEAN)
            # ----------------------------
            final_score = float(
                norm_signal.tail(lookback).mean()
            )

            if np.isnan(final_score):
                continue

            # ----------------------------
            # F5: THRESHOLD FILTER
            # ----------------------------
            if abs(final_score) < threshold:
                final_score = 0.0

            # ----------------------------
            # DISLOCATION EVENTS (FOR CHART MARKERS)
            # ----------------------------
            dislocation_events_out[sym] = (
                norm_signal.abs() > threshold
            ).astype(int)

            # ----------------------------
            # STORE OUTPUTS
            # ----------------------------
            signals_out[sym] = norm_signal
            scores_out[sym] = final_score

        # -------------------------------------------------
        # CHART (UNCHANGED CONFIG COMPATIBILITY)
        # -------------------------------------------------
        chart = self.build_chart(
            series=self.cfg.get("chart").get("series"),
            title=self.cfg.get("title"),
            charttype=self.cfg.get("chart").get("type"),
            chartmode=self.cfg.get("chart").get("mode"),
        )

        # -------------------------------------------------
        # METRICS
        # -------------------------------------------------
        metrics = {
            "lookback": lookback,
            "vol_window": vol_window,
            "volume_window": volume_window,
            "signal_threshold": threshold,
            "universe_size": len(signals_out),
            "average_score": float(
                np.mean(list(scores_out.values()))
            ) if scores_out else 0.0,
        }

        # -------------------------------------------------
        # FINAL OUTPUT
        # -------------------------------------------------
        return StrategyResult(
            name="IntradayStrategy",
            data=self.data,
            metrics=metrics,
            signals={
                "signal": signals_out,
                "volume_stress": volume_stress_out,
                "dislocation_events": dislocation_events_out,
                "score": scores_out
            },
            chart=chart
        )