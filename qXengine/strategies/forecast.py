import numpy as np
import pandas as pd

from .BaseStrategy import BaseStrategy
from ..StrategyResult import StrategyResult


class ForecastStrategy(BaseStrategy):

    # ----------------------------
    # MAIN STRATEGY
    # ----------------------------
    def run(self):

        cfg = self.cfg

        lookback = cfg.get("lookback", 252)
        horizon = cfg.get("forecast_horizon", 21)
        train_window = cfg.get("train_window", 504)

        forecast_out = {}
        actual_out = {}

        # ----------------------------
        # PRICE MATRIX
        # ----------------------------
        prices = pd.DataFrame({
            sym: df["close"]
            for sym, df in self.data.items()
            if "close" in df.columns
        }).dropna()

        # ----------------------------
        # VALIDATION
        # ----------------------------
        if prices.empty or prices.shape[1] < 1:

            return StrategyResult(
                name="ForecastStrategy",
                data=self.data,
                metrics={"error": "insufficient data"},
                signals={}
            )

        # ----------------------------
        # RETURNS (ACTUAL SERIES)
        # ----------------------------
        rets = prices.pct_change().fillna(0)

        # ----------------------------
        # FORECAST LOOP (LABEL STYLE)
        # ----------------------------
        for sym in prices.columns:

            close = prices[sym]

            # future return (label / forecast target)
            future = close.shift(-horizon)

            forecast = (future / close) - 1
            forecast = forecast.replace([np.inf, -np.inf], np.nan).fillna(0)

            actual = rets[sym].replace([np.inf, -np.inf], np.nan).fillna(0)

            # ----------------------------
            # ALIGN SERIES
            # ----------------------------
            common_index = forecast.index.intersection(actual.index)

            forecast = forecast.loc[common_index]
            actual = actual.loc[common_index]

            # ----------------------------
            # TRAIN WINDOW LIMIT
            # ----------------------------
            if len(forecast) > train_window:
                forecast = forecast.iloc[-train_window:]
                actual = actual.iloc[-train_window:]

            if len(forecast) < lookback:
                continue

            forecast_out[sym] = forecast
            actual_out[sym] = actual

        # ----------------------------
        # CHART BUILD (PAIRTRADING STYLE)
        # ----------------------------
        chart = self.build_chart(
            series=self.cfg.get("chart", {}).get("series"),
            title=self.cfg.get("chart", {}).get("title"),
            charttype=self.cfg.get("chart", {}).get("type"),
            chartmode=self.cfg.get("chart", {}).get("mode"),
        )

        # ----------------------------
        # METRICS
        # ----------------------------
        metrics = {
            "lookback": lookback,
            "forecast_horizon": horizon,
            "train_window": train_window,
            "assets": len(forecast_out)
        }

        # ----------------------------
        # FINAL OUTPUT (PAIRTRADING STYLE)
        # ----------------------------
        return StrategyResult(
            name="ForecastStrategy",
            data=self.data,
            metrics=metrics,
            signals={
                "forecast": forecast_out,
                "actual": actual_out
            },
            chart=chart
        )