# ===================================================================================
# IntradayStrategy : intraday liquidity + momentum + VWAP dislocation in Quant Xpert
# Date: 2026/06/24 
# Author : bugsbunnyla
# Comment : volume and volatility with regime as a reversal signal
# applied strategy processing in Quant Xpert 
# Signal=(momentum+reversal)×volume_stress+VWAP_deviation
# “Is price moving abnormally given volume + deviation from fair price (VWAP)?”
# ===================================================================================
import numpy as np
import pandas as pd
from ..StrategyResult import StrategyResult
from ..strategies.BaseStrategy import BaseStrategy
#
# What VWAP actually means (core idea)
# Instead of a simple average: Mean Price=∑P/N
# VWAP weights by volume:VWAP=∑(P×V)/∑V
# So high-volume trades matter more than low-volume trades.
#
#
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

    # ==================================================
    # SAFE INDEX NORMALIZER (NO INTEGER FALLBACK)
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
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]

        return df

    # ==================================================
    # VWAP (SESSION SAFE)
    # ==================================================
    def vwap(self, df):

        if "volume" not in df.columns:
            return pd.Series(0.0, index=df.index)

        session = df.index.normalize()

        pv = df["close"] * df["volume"]

        cum_pv = pv.groupby(session).cumsum()
        cum_vol = df["volume"].groupby(session).cumsum()

        return cum_pv / (cum_vol + 1e-8)

    # ==================================================
    # REGIME
    # ==================================================
    def regime(self, df, lookback=50):

        ret = df["close"].pct_change()
        trend = ret.rolling(lookback).mean()
        vol = ret.rolling(lookback).std()

        return (trend.abs() > vol).astype(int)

    # ==================================================
    # SESSION WEIGHT
    # ==================================================
    def session_weight(self, df):

        hour = df.index.hour

        return np.where((hour <= 10) | (hour >= 15), 1.2, 0.85)

    # ==================================================
    # SAFE SERIES WRAPPER (CRITICAL FIX)
    # ==================================================
    def _series(self, values, index):
        s = pd.Series(values)
        s.index = index
        return s

    # ==================================================
    # MAIN STRATEGY
    # ==================================================
    def run(self):

        cfg = self.cfg

        lookback = cfg.get("lookback", 20)
        vol_window = cfg.get("vol_window", 20)
        volume_window = cfg.get("volume_window", 20)
        threshold = cfg.get("signal_threshold", 1.5)

        signals_out = {}
        charts_out = {}

        # ==================================================
        # LOOP SYMBOLS
        # ==================================================
        for sym, df in self.data.items():

            if "close" not in df.columns:
                continue

            df = self._normalize(df)

            if len(df) < max(lookback, vol_window, volume_window):
                continue

            # ==================================================
            # PRICE SIGNAL
            # ==================================================
            ma = df["close"].rolling(lookback).mean()
            std = df["close"].rolling(lookback).std()

            reversal = -(df["close"] - ma) / (std + 1e-8)
            momentum = df["close"].pct_change(lookback)

            price_signal = momentum - reversal
            price_signal = (price_signal - price_signal.mean()) / (price_signal.std() + 1e-8)

            # ==================================================
            # VWAP SIGNAL
            # ==================================================
            vwap = self.vwap(df)
            vwap_dev = (df["close"] - vwap) / (vwap + 1e-8)

            vwap_z = (vwap_dev - vwap_dev.rolling(50).mean()) / (vwap_dev.rolling(50).std() + 1e-8)

            # ==================================================
            # VOLUME SIGNAL
            # ==================================================
            vol_mean = df["volume"].rolling(volume_window).mean()
            vol_std = df["volume"].rolling(volume_window).std()

            volume_z = (df["volume"] - vol_mean) / (vol_std + 1e-8)
            volume_shock = np.tanh(volume_z)

            # ==================================================
            # REGIME
            # ==================================================
            regime = self.regime(df, lookback)

            # ==================================================
            # SESSION WEIGHT
            # ==================================================
            sess_w = self.session_weight(df)

            trend_mode = regime.values
            meanrev_mode = 1 - trend_mode

            micro = vwap_z + volume_shock

            final_signal = (
                (0.7 * price_signal + 0.3 * micro) * trend_mode +
                (0.4 * price_signal + 0.6 * micro) * meanrev_mode
            )

            final_signal = final_signal * sess_w
            final_signal = pd.Series(final_signal, index=df.index).fillna(0)

            # ==================================================
            # EVENTS
            # ==================================================
            event_score = final_signal / (final_signal.rolling(50).std() + 1e-8)
            events = (np.abs(event_score) > threshold).astype(int)

            # ==================================================
            # FINAL OUTPUT DF (NO LOSS)
            # ==================================================
            out_df = pd.DataFrame(index=df.index)

            out_df["price_signal"] = self._series(price_signal, df.index)
            out_df["vwap_z"] = self._series(vwap_z, df.index)
            out_df["volume_shock"] = self._series(volume_shock, df.index)
            out_df["final_signal"] = self._series(final_signal, df.index)
            out_df["event"] = events

            charts_out[sym] = out_df

            # ==================================================
            # SIGNALS (PRESERVE INDEX ALWAYS)
            # ==================================================
            signals_out[sym] = {
                "price_signal": self._series(price_signal, df.index),
                "vwap_z": self._series(vwap_z, df.index),
                "volume_shock": self._series(volume_shock, df.index),
                "final_signal": self._series(final_signal, df.index),
                "event": events
            }

        # ==================================================
        # SAFE CHARTDATA STRUCTURE (NO CONCAT LOSS)
        # ==================================================
        chartdata = charts_out

        # ==================================================
        # RETURN STRATEGY RESULT (FIXED CONTRACT)
        # ==================================================
        return StrategyResult(
            name=self.__class__.__name__,
            data=self.data,

            metrics={
                "lookback": lookback,
                "vol_window": vol_window,
                "volume_window": volume_window,
                "signal_threshold": threshold,
                "symbols": len(signals_out)
            },

            signals=signals_out,

            chart={
                "charttype": "line",
                "chartmode": "lines",
                "title": "Intraday Strategy - Fixed Contract",
                "xaxis": "date",
                "yaxis": "value",

                # IMPORTANT: KEEP STRUCTURE
                "chartdata": chartdata,

                "series": [
                    {"name": "Final Signal", "source": "final_signal"},
                    {"name": "VWAP Z", "source": "vwap_z"},
                    {"name": "Volume Shock", "source": "volume_shock"},
                    {"name": "Price Signal", "source": "price_signal"},
                    {"name": "Event", "source": "event", "style": "markers"}
                ]
            }
        )
# =======================================================================
# END OF INTRADAY STRATEGY
# =======================================================================