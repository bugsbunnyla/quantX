import numpy as np
import pandas as pd

from .BaseStrategy import BaseStrategy
from ..StrategyResult import StrategyResult

class ForecastStrategy(BaseStrategy):

    # ==================================================
    # MAIN STRATEGY
    # ==================================================
    def run(self):

        cfg = self.cfg

        lookback = cfg.get("lookback", 252)
        horizon = cfg.get("forecast_horizon", 21)
        train_window = cfg.get("train_window", 504)

        forecast_out = {}
        actual_out = {}

        # ==================================================
        # CLEAN INPUT DATA
        # ==================================================
        cleaned = {}

        for sym, df in self.data.items():

            if not isinstance(df, pd.DataFrame):
                continue

            if "close" not in df.columns:
                continue

            tmp = df.copy()

            # -----------------------------
            # DATE COLUMN HANDLING
            # -----------------------------
            if "date" in tmp.columns:

                tmp["date"] = pd.to_datetime(
                    tmp["date"],
                    errors="coerce"
                )

                tmp = tmp.dropna(subset=["date"])
                tmp = tmp.sort_values("date")
                tmp = tmp.set_index("date")

            else:

                if not isinstance(tmp.index, pd.DatetimeIndex):

                    tmp.index = pd.to_datetime(
                        tmp.index,
                        errors="coerce"
                    )

                    tmp = tmp[tmp.index.notna()]

                tmp = tmp.sort_index()

            # -----------------------------
            # REMOVE DUPLICATES
            # -----------------------------
            tmp = tmp[~tmp.index.duplicated(keep="last")]

            cleaned[sym] = tmp

        # ==================================================
        # PRICE MATRIX
        # ==================================================
        prices = pd.DataFrame({
            sym: df["close"]
            for sym, df in cleaned.items()
        })

        prices = prices.sort_index()
        prices = prices.ffill()
        prices = prices.dropna(how="all")

        # ==================================================
        # VALIDATION
        # ==================================================
        if prices.empty or prices.shape[1] < 1:

            return StrategyResult(
                name="ForecastStrategy",
                data=self.data,
                metrics={"error": "insufficient data"},
                signals={},
                chart=None
            )

        # ==================================================
        # RETURNS
        # ==================================================
        rets = prices.pct_change().fillna(0)

        # ==================================================
        # FORECAST LOOP
        # ==================================================
        for sym in prices.columns:

            close = prices[sym]

            future = close.shift(-horizon)

            forecast = (future / close) - 1
            forecast = forecast.replace([np.inf, -np.inf], np.nan).fillna(0)

            actual = rets[sym].replace([np.inf, -np.inf], np.nan).fillna(0)

            # -----------------------------
            # ALIGN
            # -----------------------------
            common_index = forecast.index.intersection(actual.index)

            forecast = forecast.loc[common_index]
            actual = actual.loc[common_index]

            # -----------------------------
            # TRAIN WINDOW
            # -----------------------------
            if len(forecast) > train_window:
                forecast = forecast.iloc[-train_window:]
                actual = actual.iloc[-train_window:]

            if len(forecast) < lookback:
                continue

            # ==================================================
            # FORCE CLEAN DATETIME SERIES (IMPORTANT FIX)
            # ==================================================
            forecast_out[sym] = pd.Series(
                forecast.values,
                index=pd.to_datetime(forecast.index),
                name=f"{sym}_forecast"
            )

            actual_out[sym] = pd.Series(
                actual.values,
                index=pd.to_datetime(actual.index),
                name=f"{sym}_actual"
            )

        # ==================================================
        # CHART
        # ==================================================
        chart = self.build_chart(
            series=self.cfg.get("chart", {}).get("series"),
            title=self.cfg.get("chart", {}).get("title"),
            charttype=self.cfg.get("chart", {}).get("type"),
            chartmode=self.cfg.get("chart", {}).get("mode"),
        )

        # ==================================================
        # METRICS
        # ==================================================
        metrics = {
            "lookback": lookback,
            "forecast_horizon": horizon,
            "train_window": train_window,
            "assets": len(forecast_out)
        }

        # ==================================================
        # OUTPUT
        # ==================================================
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