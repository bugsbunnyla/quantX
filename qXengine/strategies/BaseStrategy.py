from ..StrategyResult import StrategyResult
from ..StrategyCharts import StrategyChart
from .FormulaOutput import FormulaOutput
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
        ##### qX base data structure #####
        self.formulaOutput = FormulaOutput(self.data)
        print("[BASE] structured data ", self.formulaOutput.assemble())
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

            #print("sr add", self.cfg.get("title", self.__class__.__name__))

            sr.add(tab)

        return sr

    def _clip_4y(self, df: pd.DataFrame):
       df = df.copy()

       df.index = pd.to_datetime(df.index, errors="coerce")
       df = df[~df.index.isna()]

       df = df.sort_index()

       end = df.index.max()
       start = end - pd.DateOffset(years=4)

       return df.loc[df.index >= start]

    def _build_chartdata(self, data_dict: dict):
             """
                Enforces:
               index = datetime (x-axis)
               columns = y-series only
               no 'date' column allowed
               no 1970 fallback risk
               see if we need to replace with this 
               def _build_chartdata(self, data_dict: dict):

                df = pd.DataFrame(data_dict)

               # -----------------------------
               #  1. FORCE datetime index (X-AXIS)
               # -----------------------------
               df.index = pd.to_datetime(df.index, errors="coerce")
               df = df[~df.index.isna()]
               df = df.sort_index()

               # -----------------------------
               # 2. REMOVE invalid columns
               # -----------------------------
               df = df.drop(columns=["date"], errors="ignore")

               # -----------------------------
               # 3. CLIP TO LAST 4 YEARS (CRITICAL)
               # -----------------------------
                end = df.index.max()
                start = end - pd.DateOffset(years=4)

                df = df.loc[df.index >= start]

                 return df


             """

             df = pd.DataFrame(data_dict)

             # FORCE datetime index safety
             df.index = pd.to_datetime(df.index, errors="coerce")
             df = df[~df.index.isna()]
             df = df.sort_index()

             # REMOVE accidental date columns if any
             df = df.drop(columns=["date"], errors="ignore")

             return df

    # =====================================================
    # REQUIRED OVERRIDE
    # =====================================================

    def run(self):
        raise NotImplementedError(
            f"{self.__class__.__name__}.run() not implemented"
        )

