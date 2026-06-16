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
import numpy as np
import pandas as pd

from ..StrategyResult import StrategyResult
from ..strategies.BaseStrategy import BaseStrategy


class IntradayStrategy(BaseStrategy):

    requires_factor_engine = False

    # -------------------------------------------------
    def momentum(self, df, window=20):
        return df["close"].pct_change(window)

    # -------------------------------------------------
    def reversal(self, df, window=5):
        return -df["close"].pct_change(window)

    # -------------------------------------------------
    def vwap_deviation(self, df):

        if "volume" not in df.columns:
            return pd.Series(0.0, index=df.index)

        pv = df["close"] * df["volume"]
        vwap = pv.cumsum() / (df["volume"].cumsum() + 1e-8)

        return (df["close"] - vwap) / (vwap + 1e-8)

    # -------------------------------------------------
    # 🔥 STRICT TIME CLEANER (CORE FIX)
    # -------------------------------------------------
    def _ensure_datetime_index(self, df):

        if "date" in df.columns:
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
            df = df[df["date"] > pd.Timestamp("2000-01-01")]
            df = df.sort_values("date")
            df = df.set_index("date")
        else:
            df = df.copy()
            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df[df.index.notna()]
            df = df[df.index > pd.Timestamp("2000-01-01")]
            df = df.sort_index()

        return df

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

        # =================================================
        # CLEAN INPUT DATA (CRITICAL)
        # =================================================
        cleaned = {
            sym: self._ensure_datetime_index(df)
            for sym, df in self.data.items()
            if "close" in df.columns
        }

        # -------------------------------------------------
        for sym, df in cleaned.items():

            if len(df) < max(lookback, vol_window, volume_window):
                continue

            mom = self.momentum(df, lookback)
            rev = self.reversal(df, max(2, lookback // 4))

            raw_signal = (mom + rev).fillna(0)

            vol = raw_signal.rolling(vol_window).std().replace(0, np.nan)
            norm_signal = raw_signal / (vol + 1e-8)

            if "volume" in df.columns:

                vol_ma = df["volume"].rolling(volume_window).mean()
                vol_ratio = df["volume"] / (vol_ma + 1e-8)

                norm_signal = norm_signal * vol_ratio
                volume_stress_out[sym] = vol_ratio.fillna(1.0)

            else:
                volume_stress_out[sym] = pd.Series(1.0, index=df.index)

            vwap_dev = self.vwap_deviation(df)
            norm_signal = norm_signal + 0.5 * vwap_dev

            norm_signal = norm_signal.replace([np.inf, -np.inf], np.nan).fillna(0)

            final_score = float(norm_signal.tail(lookback).mean())

            if abs(final_score) < threshold:
                final_score = 0.0

            dislocation_events_out[sym] = (norm_signal.abs() > threshold).astype(int)

            # =================================================
            # 🔥 FINAL GUARANTEE: FORCE DATETIME INDEX
            # =================================================
            norm_signal.index = pd.to_datetime(norm_signal.index, errors="coerce")
            norm_signal = norm_signal[norm_signal.index.notna()]
            norm_signal = norm_signal.sort_index()

            signals_out[sym] = norm_signal
            scores_out[sym] = final_score

        # -------------------------------------------------
        chart = self.build_chart(
            series=self.cfg.get("chart").get("series"),
            title=self.cfg.get("title"),
            charttype=self.cfg.get("chart").get("type"),
            chartmode=self.cfg.get("chart").get("mode"),
        )

        return StrategyResult(
            name="IntradayStrategy",
            data=self.data,
            metrics={
                "lookback": lookback,
                "vol_window": vol_window,
                "volume_window": volume_window,
                "signal_threshold": threshold,
                "universe_size": len(signals_out),
                "average_score": float(np.mean(list(scores_out.values()))) if scores_out else 0.0,
            },
            signals={
                "signal": signals_out,
                "volume_stress": volume_stress_out,
                "dislocation_events": dislocation_events_out,
                "score": scores_out
            },
            chart=chart
        )