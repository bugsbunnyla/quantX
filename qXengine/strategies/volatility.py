import pandas as pd
import numpy as np

from .BaseStrategy import BaseStrategy
from ..StrategyResult import StrategyResult


class VolatilityStrategy(BaseStrategy):

    # ----------------------------
    # MAIN STRATEGY
    # ----------------------------
    def run(self):

        cfg = self.cfg

        lookback = cfg.get("lookback", 63)
        vol_window = cfg.get("vol_window", 21)
        target_vol = cfg.get("target_vol", 0.15)

        chart_cfg = cfg.get("chart", {})

        signals_out = {}
        vol_out = {}

        # ----------------------------
        # PER SYMBOL LOOP
        # ----------------------------
        for sym, df in self.data.items():

            if "close" not in df.columns:
                continue

            df = df.copy()

            # ----------------------------
            # RETURNS
            # ----------------------------
            df["ret"] = df["close"].pct_change()

            # ----------------------------
            # REALIZED VOLATILITY
            # ----------------------------
            #df["volatility"] = df["ret"].rolling(vol_window).std()
            df["volatility"] = df["ret"].rolling(vol_window).std() * np.sqrt(252)

            # ----------------------------
            # TARGET VOL SERIES
            # ----------------------------
            df["target_vol"] = target_vol

            # ----------------------------
            # VOLATILITY SCALING SIGNAL
            # ----------------------------
            df["vol_signal"] = target_vol / (df["volatility"] + 1e-8)

            # ----------------------------
            # STORE OUTPUTS
            # ----------------------------
            signals_out[sym] = df["vol_signal"].fillna(0)

            vol_out[sym] = {
                "volatility": df["volatility"],
                "target_vol": df["target_vol"]
            }

        # ----------------------------
        # BUILD CHART (BASE CLASS STANDARD)
        # ----------------------------
        chart = self.build_chart(
            series=chart_cfg.get("series"),
            title=chart_cfg.get("title"),
            charttype=chart_cfg.get("type"),
            chartmode=chart_cfg.get("mode"),
        )

        # ----------------------------
        # METRICS
        # ----------------------------
        metrics = {
            "lookback": lookback,
            "vol_window": vol_window,
            "target_vol": target_vol,
            "symbols": len(self.data),
            "valid_symbols": len(signals_out)
        }

        # ----------------------------
        # FINAL OUTPUT (PAIRTRADING STYLE)
        # ----------------------------
        return StrategyResult(
            name="VolatilityStrategy",
            data=self.data,
            metrics=metrics,
            signals={
                "vol_signal": signals_out,
                "vol_series": vol_out
            },
            chart=chart
        )