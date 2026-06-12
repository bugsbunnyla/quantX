# cross sectional risk signal timestamp based spread out returns across assets  aka time series dispersion index  optionally per asset 
import pandas as pd
import numpy as np

from .BaseStrategy import BaseStrategy
from ..StrategyResult import StrategyResult


class DispersionStrategy(BaseStrategy):

    # ----------------------------
    # MAIN STRATEGY
    # ----------------------------
    def run(self):

        cfg = self.cfg

        lookback = cfg.get("lookback", 63)
        cs_window = cfg.get("cross_sectional_window", 63)

        # ----------------------------
        # PRICE MATRIX (CLOSE)
        # ----------------------------
        prices = pd.DataFrame({
            sym: df["close"]
            for sym, df in self.data.items()
            if "close" in df.columns
        }).dropna()

        # ----------------------------
        # VALIDATION
        # ----------------------------
        if prices.empty or prices.shape[1] < 2:

            return StrategyResult(
                name="DispersionStrategy",
                data=self.data,
                metrics={"error": "insufficient price data"},
                signals={},
                chart=None
            )

        # ----------------------------
        # RETURNS MATRIX
        # ----------------------------
        rets = prices.pct_change().dropna()

        # ----------------------------
        # CROSS-SECTIONAL DISPERSION
        # (TRUE FORMULA)
        # dispersion_t = std across assets at time t
        # ----------------------------
        dispersion = rets.std(axis=1)

        # ----------------------------
        # SMOOTHING (MA)
        # ----------------------------
        ma_63 = dispersion.rolling(lookback).mean()

        # ----------------------------
        # REGIME SIGNALS
        # ----------------------------
        regime_switches = pd.Series(
            np.where(dispersion > ma_63, 1, -1),
            index=dispersion.index
        )

        # ----------------------------
        # CLEAN ALIGNMENT
        # ----------------------------
        dispersion = dispersion.fillna(0)
        ma_63 = ma_63.fillna(0)
        regime_switches = regime_switches.fillna(0)

        # ----------------------------
        # SIGNALS (CHART-READY)
        # ----------------------------
        signals = {
            "dispersion": dispersion,
            "ma_63": ma_63,
            "regime_switches": regime_switches
        }

        # ----------------------------
        # METRICS
        # ----------------------------
        metrics = {
            "lookback": lookback,
            "cross_sectional_window": cs_window,
            "mean_dispersion": float(dispersion.mean()),
            "dispersion_volatility": float(dispersion.std()),
            "assets": prices.shape[1]
        }

        # ----------------------------
        # BUILD CHART (BASE CLASS)
        # ----------------------------
        chart = self.build_chart(
            series=self.cfg.get("chart", {}).get("series"),
            title=self.cfg.get("chart", {}).get("title"),
            charttype=self.cfg.get("chart", {}).get("type"),
            chartmode=self.cfg.get("chart", {}).get("mode"),
        )

        # ----------------------------
        # FINAL OUTPUT
        # ----------------------------
        return StrategyResult(
            name="DispersionStrategy",
            data=self.data,
            metrics=metrics,
            signals=signals,
            chart=chart
        )