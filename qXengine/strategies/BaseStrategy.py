# ===============================================================
# BaseStrategy : base strategy in Quant Xpert
# Date: 2026/06/24 
# Author : bugsbunnyla
# Comment : Processing of base class of strategy
# ===============================================================
from ..StrategyResult import StrategyResult
from ..StrategyCharts import StrategyChart
from .FormulaInfo import FormulaInfo
import numpy as np
import pandas as pd

class BaseStrategy:

    requires_factor_engine = False

    def __init__(self, context):

        if context is None:
            raise ValueError(
                "Context must be provided by QuantXEngine"
            )
        self.rawdata = context.data
        self.context = context

        self.data = context.data.copy()
        cleaned_data = {}
        for symbol, df in self.data.items():
            if df is None:
                continue
            df = df.copy()
            # ---------------------------------
            # REMOVE FULL NaN ROWS FIRST
            # ---------------------------------
            df = df.dropna( how="all" )
            # skip empty dataframe
            if df.empty:
                continue
            cleaned_data[symbol] = df
        # ---------------------------------
        # NOW ASSIGN DATA
        # ---------------------------------
        self.data = cleaned_data
        # -----------------------------------------
        # FIX INDEXES BEFORE FORMULA ENGINE
        # -----------------------------------------
        for symbol, df in self.data.items():
            self.data[symbol] = self._ensure_datetime_index(df)

        self.cfg = context.cfg
        self.runtime_cfg = context.runtime_cfg
        self.factor_engine = context.factor_engine
        self.logger = context.logger
        ##### qX base data structure #####
        self.formulaOutput = FormulaInfo(self.data)

    # -------------------------------------------------
    # INDEX HANDLING
    # -------------------------------------------------
    def _ensure_datetime_index(self, df):
        df = df.copy()
        # remove only fully empty rows
        df = df.dropna( how="all")
        if "date" in df.columns:
            df["date"] = pd.to_datetime( df["date"], errors="coerce")
            df = df.dropna( subset=["date"] )
            df = df[ df["date"] >  pd.Timestamp("2000-01-01")  ]
            df = df.sort_values( "date" )
            df = df.set_index(  "date"  )
        else:
          if not isinstance( df.index,  pd.DatetimeIndex  ):
            df.index = pd.to_datetime( df.index,  errors="coerce" )
            df = df[  df.index.notna() ]
            df = df[ df.index >  pd.Timestamp("2000-01-01")  ]
            df = df.sort_index()
        return df

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

    # ====================================================
    # Get value
    # ====================================================
    def _extract_metric(self, fo, symbol, name, report_key=None):
      def clean(value):
        # DataFrame
        if isinstance(value, pd.DataFrame):
            if symbol in value.columns:
                value = value[symbol].iloc[-1]
            elif len(value):
                value = value.iloc[-1, 0]
        # Series
        elif isinstance(value, pd.Series):
            if symbol in value.index:
                value = value.loc[symbol]
            elif len(value):
                value = value.iloc[-1]
        # Dictionary
        elif isinstance(value, dict):
            value = value.get(symbol)
        try:
            value = float(value)
            return value if np.isfinite(value) else None
        except (TypeError, ValueError):
            return None
      try:
        value = fo.get(name)
        return clean(value)
      except KeyError:
        return None
      except Exception as e:
        print(f"[WARN] {name} {symbol}: {e}")
        return None


# =============================================================================
# END OF BASE STRATEGY
# =============================================================================