# cross sectional risk signal timestamp based spread out returns across assets  aka time series dispersion index  optionally per asset 
import pandas as pd
import numpy as np

from .BaseStrategy import BaseStrategy
from ..StrategyResult import StrategyResult

class DispersionStrategy(BaseStrategy):

    def run(self):

        cfg = self.cfg

        lookback = cfg.get("lookback", 63)
        cs_window = cfg.get("cross_sectional_window", 63)

        # ==================================================
        # FIX INPUT DATA ONLY
        # ==================================================
        cleaned = {}

        for sym, df in self.data.items():

            if not isinstance(df, pd.DataFrame):
                continue

            if "close" not in df.columns:
                continue

            tmp = df.copy()

            if "date" in tmp.columns:

                tmp["date"] = pd.to_datetime(
                    tmp["date"],
                    errors="coerce"
                )

                tmp = tmp.dropna(subset=["date"])
                tmp = tmp.sort_values("date")
                tmp = tmp.set_index("date")

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

            cleaned[sym] = tmp

        # ==================================================
        # PRICE MATRIX
        # ==================================================
        prices = pd.DataFrame({
            sym: df["close"]
            for sym, df in cleaned.items()
        })

        prices = prices.sort_index()

        # keep full history
        prices = prices.ffill()

        # only remove rows where all assets are missing
        prices = prices.dropna(how="all")

        # ==================================================
        # VALIDATION
        # ==================================================
        if prices.empty or prices.shape[1] < 2:

            return StrategyResult(
                name="DispersionStrategy",
                data=self.data,
                metrics={
                    "error": "insufficient price data"
                },
                signals={},
                chart=None
            )

        # ==================================================
        # RETURNS
        # ==================================================
        rets = prices.pct_change().dropna()

        if rets.empty:

            return StrategyResult(
                name="DispersionStrategy",
                data=self.data,
                metrics={
                    "error": "insufficient return history"
                },
                signals={},
                chart=None
            )

        # ==================================================
        # CROSS-SECTIONAL DISPERSION
        # std across assets each date
        # ==================================================
        dispersion = pd.Series(
            rets.std(axis=1),
            index=rets.index,
            name="dispersion"
        )

        # ==================================================
        # MOVING AVERAGE
        # ==================================================
        ma_63 = (
            dispersion
            .rolling(
                lookback,
                min_periods=1
            )
            .mean()
        )

        # ==================================================
        # REGIME
        # ==================================================
        regime_switches = pd.Series(
            np.where(
                dispersion > ma_63,
                1,
                -1
            ),
            index=dispersion.index,
            name="regime_switches"
        )

        # ==================================================
        # OPTIONAL Z-SCORE
        # ==================================================
        dispersion_vol = (
            dispersion
            .rolling(
                lookback,
                min_periods=1
            )
            .std()
            .fillna(0)
        )

        zscore = (
            (dispersion - ma_63)
            / (dispersion_vol + 1e-8)
        )

        zscore = (
            zscore
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .fillna(0)
        )

        # ==================================================
        # MOMENTUM
        # ==================================================
        momentum = (
            dispersion
            .diff()
            .fillna(0)
        )

        # ==================================================
        # METRICS
        # ==================================================
        metrics = {

            "lookback": lookback,
            "cross_sectional_window": cs_window,

            "assets": prices.shape[1],

            "mean_dispersion": float(
                dispersion.mean()
            ),

            "dispersion_volatility": float(
                dispersion.std()
            ),

            "latest_dispersion": float(
                dispersion.iloc[-1]
            ),

            "latest_zscore": float(
                zscore.iloc[-1]
            )
        }

        # ==================================================
        # SIGNALS
        # ==================================================
        signals = {

            "dispersion": dispersion,

            "ma_63": ma_63,

            "dispersion_volatility": dispersion_vol,

            "zscore": zscore,

            "momentum": momentum,

            "regime_switches": regime_switches
        }

        # ==================================================
        # CHART
        # ==================================================
        chart = self.build_chart(
            series=self.cfg.get(
                "chart",
                {}
            ).get(
                "series"
            ),
            title=self.cfg.get(
                "chart",
                {}
            ).get(
                "title"
            ),
            charttype=self.cfg.get(
                "chart",
                {}
            ).get(
                "type"
            ),
            chartmode=self.cfg.get(
                "chart",
                {}
            ).get(
                "mode"
            ),
        )

        # ==================================================
        # SUCCESS
        # ==================================================
        return StrategyResult(
            name="DispersionStrategy",
            data=self.data,
            metrics=metrics,
            signals=signals,
            chart=chart
        )