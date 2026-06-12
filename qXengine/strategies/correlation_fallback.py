import pandas as pd
import numpy as np
from .BaseStrategy import BaseStrategy
from ..StrategyResult import StrategyResult

#
# Correlation Fallback formulas used for the strategy
# 1. Returns Rt	​=​Pt/Pt−1 ​−1  as returns = price_df.pct_change()
# 2. Rolling Correlation Matrix Σt = Corr(Rt−w:t​)  as rolling_corr = returns.rolling(corr_window).corr()
# 3. Average Correlation Regime Signal ρt​=N1​i=j∑​∣Corrij​(t)∣ as  avg_corr_series = rolling_corr.groupby(level=0).mean().mean(axis=1)
# 4. Regimet​=⎩ ⎨ ⎧​+1 −1 0 ​ρt​≥θhigh ​ρt​≤θlow ​otherwise​ as regime_series = avg_corr_series.apply(...) 
# 5. Dispersion(Volatility proxy)  Dt​=N1​i∑​σ(Ri,t​) as dispersion_series = returns.rolling(dispersion_window).std().mean(axis=1)
# 6. z-Score  Dispersion (instability detection ) Zt​=​(Dt​−μD​​)/σD as z_dispersion = (dispersion_series - mean) / std
# 7. Signal smoothing St​=MAk​(Regimet​) as signal_series.rolling(signal_smooth).mean()
# 8. Equity Curve (market proxy) Et	​=∏(1+Rˉt​) as  equity_curve = (1 + returns.mean(axis=1)).cumprod()
class CorrelationFallback(BaseStrategy):

    def run(self):

        cfg = self.cfg

        lookback = cfg.get("lookback", 42)
        corr_window = cfg.get("corr_window", 42)

        signal_threshold = cfg.get("signal_threshold", 0.6)
        low_corr_threshold = cfg.get("low_corr_threshold", 0.3)

        dispersion_window = cfg.get("dispersion_window", 21)

        risk_on_when_corr_low = cfg.get("risk_on_when_corr_low", True)
        risk_off_when_corr_high = cfg.get("risk_off_when_corr_high", True)

        signal_smooth = cfg.get("signal_smooth", 5)

        title = cfg.get("title", "Correlation Fallback")

        assets = list(self.data.keys())

        if len(assets) < 2:
            return StrategyResult(
                "CorrelationFallback",
                self.data,
                {"error": "insufficient assets"},
                {}
            )

        # =====================================================
        # PRICE + RETURNS
        # =====================================================
        price_df = pd.DataFrame({
            a: self.data[a]["close"]
            for a in assets
        }).dropna()

        returns = price_df.pct_change().dropna()

        if len(returns) < corr_window:
            return StrategyResult("CorrelationFallback", self.data, {}, {})

        # =====================================================
        # ROLLING CORRELATION (TIME SERIES)
        # =====================================================
        rolling_corr = returns.rolling(corr_window).corr()

        avg_corr_series = (
            rolling_corr
            .groupby(level=0)
            .mean()
            .mean(axis=1)
            .dropna()
        )

        avg_corr = float(avg_corr_series.iloc[-1])

        # =====================================================
        # DISPERSION (VOLATILITY REGIME)
        # =====================================================
        dispersion_series = returns.rolling(dispersion_window).std().mean(axis=1)

        z_dispersion = (
            (dispersion_series - dispersion_series.mean())
            / (dispersion_series.std() + 1e-8)
        )

        # =====================================================
        # REGIME ENGINE (QUANT LOGIC)
        # =====================================================
        regime_series = avg_corr_series.apply(
            lambda x: 1 if x >= signal_threshold
            else (-1 if x <= low_corr_threshold else 0)
        )

        # override for instability spike
        regime_series = regime_series.copy()
        regime_series[z_dispersion > 1.5] = -1

        regime = int(regime_series.iloc[-1])

        # =====================================================
        # SMOOTH SIGNAL
        # =====================================================
        signal_series = (
            regime_series.rolling(signal_smooth)
            .mean()
            .fillna(0)
        )

        # =====================================================
        # EQUITY PROXY
        # =====================================================
        equity_curve = (1 + returns.mean(axis=1)).cumprod()

        # =====================================================
        # THRESHOLD BANDS (CHART SAFE)
        # =====================================================
        threshold_high = pd.Series(signal_threshold, index=avg_corr_series.index)
        threshold_low = pd.Series(low_corr_threshold, index=avg_corr_series.index)

        # =====================================================
        # CHART (BUILDER SYSTEM)
        # =====================================================
        chart_cfg = cfg.get("chart", {})

        chart = self.build_chart(
            series=chart_cfg.get("series"),
            title=title,
            charttype=chart_cfg.get("type", "line"),
            chartmode=chart_cfg.get("mode", "overlay")
        )

        # =====================================================
        # METRICS
        # =====================================================
        metrics = {
            "lookback": lookback,
            "corr_window": corr_window,
            "signal_threshold": signal_threshold,
            "low_corr_threshold": low_corr_threshold,
            "dispersion_window": dispersion_window,

            "avg_corr": avg_corr,
            "dispersion": float(dispersion_series.iloc[-1]),
            "regime": regime
        }

        # =====================================================
        # SIGNALS (FULL STRUCTURE)
        # =====================================================
        signals = {
            "signal": signal_series,
            "correlation": avg_corr_series,
            "dispersion": dispersion_series,
            "regime": regime_series,
            "equity_curve": equity_curve,
            "threshold_high": threshold_high,
            "threshold_low": threshold_low
        }

        # =====================================================
        # FINAL OUTPUT
        # =====================================================
        return StrategyResult(
            "CorrelationFallback",
            self.data,
            metrics,
            signals,
            chart=chart
        )