import pandas as pd
import numpy as np

from .BaseStrategy import BaseStrategy
from ..StrategyResult import StrategyResult

class VolatilityStrategy(BaseStrategy):

    # ==================================================
    # MAIN STRATEGY
    # ==================================================
    def run(self):

        cfg = self.cfg

        lookback = cfg.get("lookback", 63)
        vol_window = cfg.get("vol_window", 21)
        target_vol = cfg.get("target_vol", 0.15)

        chart_cfg = cfg.get("chart", {})

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

            # ----------------------------------------------
            # DATE COLUMN EXISTS
            # ----------------------------------------------
            if "date" in tmp.columns:

                tmp["date"] = pd.to_datetime(
                    tmp["date"],
                    errors="coerce"
                )

                tmp = tmp.dropna(
                    subset=["date"]
                )

                tmp = tmp.sort_values(
                    "date"
                )

                tmp = tmp.set_index(
                    "date"
                )

            # ----------------------------------------------
            # USE EXISTING INDEX
            # ----------------------------------------------
            else:

                if not isinstance(
                    tmp.index,
                    pd.DatetimeIndex
                ):

                    tmp.index = pd.to_datetime(
                        tmp.index,
                        errors="coerce"
                    )

                    tmp = tmp[
                        tmp.index.notna()
                    ]

                tmp = tmp.sort_index()

            # ----------------------------------------------
            # REMOVE DUPLICATES
            # ----------------------------------------------
            tmp = tmp[
                ~tmp.index.duplicated(
                    keep="last"
                )
            ]

            cleaned[sym] = tmp

        # ==================================================
        # VALIDATION
        # ==================================================
        if len(cleaned) == 0:

            return StrategyResult(
                name="VolatilityStrategy",
                data=self.data,
                metrics={
                    "error": "no valid assets"
                },
                signals={},
                chart=None
            )

        # ==================================================
        # OUTPUT CONTAINERS
        # ==================================================
        signals_out = {}
        vol_out = {}

        latest_vols = []

        # ==================================================
        # PER SYMBOL LOOP
        # ==================================================
        for sym, df in cleaned.items():

            if len(df) < vol_window:
                continue

            df = df.copy()

            # ----------------------------------------------
            # RETURNS
            # ----------------------------------------------
            df["ret"] = (
                df["close"]
                .pct_change()
            )

            # ----------------------------------------------
            # ANNUALIZED REALIZED VOL
            # ----------------------------------------------
            df["volatility"] = (
                df["ret"]
                .rolling(
                    vol_window,
                    min_periods=1
                )
                .std()
                * np.sqrt(252)
            )

            # ----------------------------------------------
            # TARGET VOL
            # ----------------------------------------------
            df["target_vol"] = target_vol

            # ----------------------------------------------
            # POSITION SCALING
            # ----------------------------------------------
            df["vol_signal"] = (
                target_vol
                / (
                    df["volatility"]
                    + 1e-8
                )
            )

            # optional cap
            df["vol_signal"] = (
                df["vol_signal"]
                .clip(
                    lower=0,
                    upper=5
                )
            )

            # ----------------------------------------------
            # STORE SIGNAL
            # ----------------------------------------------
            signals_out[sym] = pd.Series(
                df["vol_signal"]
                .fillna(0)
                .values,
                index=df.index,
                name=f"{sym}_vol_signal"
            )

            # ----------------------------------------------
            # STORE VOL CURVES
            # ----------------------------------------------
            vol_out[sym] = {

                "volatility": pd.Series(
                    df["volatility"]
                    .values,
                    index=df.index,
                    name=f"{sym}_volatility"
                ),

                "target_vol": pd.Series(
                    df["target_vol"]
                    .values,
                    index=df.index,
                    name=f"{sym}_target_vol"
                )
            }

            # ----------------------------------------------
            # METRICS
            # ----------------------------------------------
            latest_vol = (
                df["volatility"]
                .dropna()
            )

            if len(latest_vol):

                latest_vols.append(
                    float(
                        latest_vol.iloc[-1]
                    )
                )

        # ==================================================
        # CHART
        # ==================================================
        chart = self.build_chart(
            series=chart_cfg.get(
                "series"
            ),
            title=chart_cfg.get(
                "title"
            ),
            charttype=chart_cfg.get(
                "type"
            ),
            chartmode=chart_cfg.get(
                "mode"
            ),
        )

        # ==================================================
        # METRICS
        # ==================================================
        metrics = {

            "lookback": lookback,

            "vol_window": vol_window,

            "target_vol": target_vol,

            "symbols": len(
                self.data
            ),

            "valid_symbols": len(
                signals_out
            ),

            "mean_realized_vol": (
                float(
                    np.mean(
                        latest_vols
                    )
                )
                if latest_vols
                else 0.0
            ),

            "max_realized_vol": (
                float(
                    np.max(
                        latest_vols
                    )
                )
                if latest_vols
                else 0.0
            ),

            "min_realized_vol": (
                float(
                    np.min(
                        latest_vols
                    )
                )
                if latest_vols
                else 0.0
            )
        }

        # ==================================================
        # SIGNALS
        # ==================================================
        signals = {

            "vol_signal": signals_out,

            "vol_series": vol_out
        }

        # ==================================================
        # SUCCESS
        # ==================================================
        return StrategyResult(
            name="VolatilityStrategy",
            data=self.data,
            metrics=metrics,
            signals=signals,
            chart=chart
        )