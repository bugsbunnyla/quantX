from ..StrategyResult import StrategyResult
from ..StrategyCharts import StrategyChart

import numpy as np
import pandas as pd

class BaseStrategy:

    requires_factor_engine = False

    def __init__(self, context):

        if context is None:
            raise ValueError(
                "Context must be provided by QuantXEngine"
            )

        self.context = context

        self.data = context.data
        self.cfg = context.cfg
        self.runtime_cfg = context.runtime_cfg
        self.factor_engine = context.factor_engine
        self.logger = context.logger

    # =====================================================
    # CONFIG HELPERS
    # =====================================================

    def get_cfg(self, key, default=None):
        return self.cfg.get(key, default)

    def lookback(self):
        return self.cfg.get("lookback", 20)

    def formation(self):
        return self.cfg.get("formation", 252)

    def window(self):
        return self.cfg.get("window", 60)

    # =====================================================
    # COMMON FACTOR HELPERS
    # =====================================================

    @property
    def spy(self):
        return self.data["SPY"]

    def momentum(self, series, lookback=20):
        return series.pct_change(lookback)

    def reversal(self, series, lookback=5):
        return -series.pct_change(lookback)

    def volatility(self, series, window=20):
        return series.pct_change().rolling(window).std()

    def t_stat(self, x):
        if self.factor_engine is None:
            return 0.0

        return self.factor_engine.t_stat(x)

    # =====================================================
    # VALIDATION
    # =====================================================

    def validate(self):
        return True

    # =====================================================
    # CHART BUILDER
    #
    # Optional helper.
    # Derived strategy may completely ignore this.
    # =====================================================

    def build_chart(
        self,
        chartdata=None,
        charttype=None,
        chartmode=None,
        title=None,
        xaxis=None,
        yaxis=None,
        series=None
    ):

        chart_cfg = self.cfg.get("chart", {})

        return StrategyChart(

            charttype=
                charttype
                or chart_cfg.get("type", "line"),

            chartmode=
                chartmode
                or chart_cfg.get("mode", "lines"),

            title=
                title
                or chart_cfg.get(
                    "title",
                    self.__class__.__name__
                ),

            xaxis=
                xaxis
                or [chart_cfg.get("axes", {}).get("x", "date")],

            yaxis=
                yaxis
                or [chart_cfg.get("axes", {}).get("y", "value")],

            chartdata=
                 (lambda x:
    x if isinstance(x, pd.DataFrame)
    else (x if isinstance(x, dict) else {})
)(chartdata),

            series=
                chart_cfg.get("series", [])
        )

    # =====================================================
    # RESULT BUILDER
    #
    # Derived strategy supplies everything.
    # BaseStrategy simply wraps it.
    # =====================================================

    def build_result(
        self,
        signal,
        metrics=None,
        chart=None
    ):

        sr = StrategyResult(

            name=self.cfg.get(
                "title",
                self.__class__.__name__
            ),

            data=self.data,

            metrics=metrics or {},

            signals=signal,

            chart=chart
        )

        plot_enabled = str(
            self.cfg.get(
                "plot_enabled",
                False
            )
        ).lower() == "true"

        if plot_enabled:

            tab = self.cfg.get("tab")

            print(
                "sr add",
                self.cfg.get(
                    "title",
                    self.__class__.__name__
                )
            )

            sr.add(tab)

        return sr

    # =====================================================
    # REQUIRED OVERRIDE
    # =====================================================

    def run(self):
        raise NotImplementedError(
            f"{self.__class__.__name__}.run() not implemented"
        )

