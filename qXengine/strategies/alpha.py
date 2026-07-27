# ===============================================================
# AlphaStrategy : alpha strategy in Quant Xpert
# Date: 2026/06/24 
# Author : bugsbunnyla
# Comment : Processing of alpha strategy
# ===============================================================
import numpy as np
import pandas as pd

from ..StrategyResult import StrategyResult
from ..strategies.BaseStrategy import BaseStrategy
from .FormulaInfo import FormulaInfo

"""
===========================================================
AlphaStrategy CONFIG CONTRACT (REQUIRED FORMAT)
===========================================================

This strategy expects the following configuration shape:

-----------------------------------------------------------
1. CORE PARAMETERS
-----------------------------------------------------------
lookback:
    int
    Rolling window used for alpha/beta estimation.
    Example: 252

rebalance:
    int
    Portfolio rebalance frequency (used externally or by engine).
    Example: 21

top_quantile:
    float (0-1)
    Long selection percentile.
    Example: 0.2

bottom_quantile (optional but supported):
    float (0-1)
    Short selection percentile.

benchmark:
    string (optional override via runtime_cfg)
    Default fallback: "SPY"

-----------------------------------------------------------
2. CHART CONTRACT (IMPORTANT FOR DASHBOARD)
-----------------------------------------------------------

chart.type:
    string
    Determines renderer selection in dashboard.

    Supported values for AlphaStrategy:
        - "time_series_multi"   (recommended for extended signal evolution)
        - "bar"                 (recommended for cross-sectional snapshot view)

chart.series:
    list of dicts describing logical series mapping.

    Expected format:

    [
        {
            "key": "price",
            "source": "close",
            "label": "Price"
        },
        {
            "key": "signal",
            "source": "signal",
            "label": "Alpha Signal"
        },
        {
            "key": "benchmark",
            "source": "SPY",
            "label": "Benchmark"
        }
    ]

    NOTE:
    - "source" is resolved from StrategyResult.data or chartdata
    - "key" is the dashboard mapping identifier
    - AlphaStrategy may not always fully use all series (especially benchmark alignment)

chart.axes:
    {
        "x": "date",
        "y": "normalized_value"
    }

    Defines default axis labels for renderer.
    AlphaStrategy may override internally if charttype != time_series_multi.

chart.title:
    string
    Human-readable chart title.

-----------------------------------------------------------
3. STRATEGY OUTPUT ASSUMPTIONS
-----------------------------------------------------------

This strategy produces:

signal:
    dict[str, float]
    Cross-sectional portfolio weights:
        + positive = long
        - negative = short

metrics:
    dict containing:
        - score
        - alpha_scores
        - beta_scores
        - lookback
        - benchmark
        - assets

chart:
    StrategyChart object with:
        - charttype derived from cfg.chart.type
        - chartdata constructed from:
            * benchmark series
            * alpha_scores (cross-sectional snapshot)
            * optional signal mapping

-----------------------------------------------------------
4. IMPORTANT BEHAVIOR NOTES
-----------------------------------------------------------

- This is NOT a pure time-series forecasting model.
- It is primarily a cross-sectional ranking engine.
- "time_series_multi" is used only for visualization expansion.
- "bar" is the most accurate representation of alpha scores.

- No normalization is performed in BaseStrategy.
- No engine (qx_engine / institution_engine) should modify signals.

===========================================================
"""

class AlphaStrategy(BaseStrategy):

    # -------------------------------------------------
    # METRIC SERIES
    # -------------------------------------------------
    def _build_metric_series( self, fo, symbols,  metric, report_key=None ):
        values = []
        for symbol in symbols:
            try:
                value = self._extract_metric( fo, symbol,  metric,  report_key  )
            except Exception:
                value = np.nan
            values.append( value  )
        return {
            "x": symbols,
            "y": values
        }
    # -------------------------------------------------
    # MAIN RUN
    # -------------------------------------------------
    def run(self):
        print( "[DEBUG] starting run" )
        # -------------------------------------------------
        # FORMULA ENGINE FIX
        # -------------------------------------------------
        # Do not call assemble().
        # assemble() executes every formula and can trigger
        # unrelated failures (example: resample on RangeIndex).
        self.formulaOutput.assemble()
        fo = self.formulaOutput
        required_formulas = [
            "ret",
            "benchmark",
            "alpha",
            "beta",
            "future_ret",
            "ic",
            "hit_ratio",
            "sharpe",
            "tstat_alpha",
            "drawdown",
            "corr_rm",
            "cvar",
            "volatility",
            "alpha",
            "beta",
            "turnover",
            "slippage",
            "kelly",
            "max_drawdown",
            "transaction_cost"
        ]
        # -------------------------------------------------
        # BENCHMARK SELECTION
        # -------------------------------------------------
        crypto_universe = any(
            str(symbol).upper().endswith("USDT")
            for symbol in self.data.keys()
        )
        if crypto_universe:
           benchmark = "BTCUSDT"
        else:
           benchmark = "SPY"

        for formula in required_formulas:
            try:
                fo.compute(  formula  )
            except Exception as e:
                print("[WARN] formula failed:",  formula,   e  )
        window = self.lookback()
        top_q = self.get_cfg( "top_quantile", 0.2 )
        bottom_q = self.get_cfg( "bottom_quantile", 0.2  )
        chart_cfg = self.get_cfg( "chart", {}  )
        series_cfg = chart_cfg.get( "series",  [] )
        # -------------------------------------------------
        # BENCHMARK RETURNS + PRICE
        # -------------------------------------------------
        ret_df = fo.get("ret")
        close_df = fo.get("mkt_price")
        benchmark_ret = pd.Series(dtype=float)
        price = {"x": [], "y": []}
        stored_benchmark = fo.get("benchmark")
        if stored_benchmark is not None:
           # benchmark already exists as a return series
           if isinstance(stored_benchmark, pd.Series):
              benchmark_ret = ( stored_benchmark.replace([np.inf, -np.inf], np.nan).dropna() )
           # benchmark exists as a column name/symbol
           elif isinstance(stored_benchmark, str) and ret_df is not None:
             if stored_benchmark in ret_df.columns:
                benchmark_ret = ( ret_df[stored_benchmark].replace([np.inf, -np.inf], np.nan).dropna() )
        # Fallback: use passed benchmark symbol
        if benchmark_ret.empty and ret_df is not None and benchmark in ret_df.columns:
              benchmark_ret = ( ret_df[benchmark].replace([np.inf, -np.inf], np.nan).dropna()    )
        # Fallback: equal-weight benchmark
        if benchmark_ret.empty and ret_df is not None:
              benchmark_ret = ( ret_df.replace([np.inf, -np.inf], np.nan).mean(axis=1).dropna()  )
        # Price series for charting
        if close_df is not None:
           price_symbol = None
           if isinstance(stored_benchmark, str) and stored_benchmark in close_df.columns:
              price_symbol = stored_benchmark
           elif benchmark in close_df.columns:
              price_symbol = benchmark

           if price_symbol is not None:
              price_series = close_df[price_symbol].dropna()
           else:
              price_series = close_df.mean(axis=1).dropna()
           price = {
             "x": price_series.index.to_pydatetime().tolist(),
             "y": price_series.tolist(),
           }


        # -------------------------------------------------
        # ALPHA / BETA
        # -------------------------------------------------
        alpha_scores = {}
        beta_scores = {}
        ret_df = fo.get("ret")        
        alpha_bar = fo.get("alpha")
        beta_bar = fo.get("beta")
        #print("[DEBUG] alpha_bar =", alpha_bar)
        #print("[DEBUG] beta_bar =", beta_bar)
        for symbol in self.data:
          try:
            # -----------------------------
            # Alpha
            # -----------------------------
            if alpha_bar is not None:
               if isinstance(alpha_bar, pd.DataFrame):
                  # current bar row
                  alpha_value = alpha_bar[symbol].iloc[-1]
               elif isinstance(alpha_bar, pd.Series):
                  # already symbol indexed
                  alpha_value = alpha_bar.get(symbol, np.nan)
               else:
                  alpha_value = np.nan
               if np.isfinite(alpha_value):
                  alpha_scores[symbol] = float(alpha_value)
            # -----------------------------
            # Beta
            # -----------------------------
            if beta_bar is not None:
               if isinstance(beta_bar, pd.DataFrame):
                  beta_value = beta_bar[symbol].iloc[-1]
               elif isinstance(beta_bar, pd.Series):
                  beta_value = beta_bar.get(symbol, np.nan)
               else:
                  beta_value = np.nan
               if np.isfinite(beta_value):
                  beta_scores[symbol] = float(beta_value)
          except Exception as e:
             print("[WARN] failed alpha/beta extraction", symbol, e  )
        #print("[DEBUG] alpha_scores = ", alpha_scores, " beta_scores" , beta_scores)
        vals = np.array(list(alpha_scores.values()))
        for symbol, beta in beta_scores.items():
           if beta <= 0:
              #print("\nBAD BETA:", symbol, beta)
              r = fo.get("ret")[symbol].dropna()
              pair = pd.concat( [ r.rename("asset"),  benchmark_ret.rename("benchmark")    ],   axis=1  ).dropna()
        # -------------------------------------------------
        # SIGNAL
        # -------------------------------------------------
        ordered = sorted(  alpha_scores.items(),     key=lambda x:x[1]    )
        symbols = list( alpha_scores.keys()    )
        scores = np.array( list(alpha_scores.values())  )
        signal = {
            "x":
                [x[0] for x in ordered],
            "y":
                [x[1] for x in ordered]
        }
        signal_map = {}
        if len(symbols):
            idx = np.argsort(  scores   )
            n = len(symbols)
            n_long = max( 1,  int(n * top_q)  )
            n_short = max( 1, int(n * bottom_q)  )
            for i in idx[-n_long:]:
                signal_map[ symbols[i] ] = 1.0 / n_long
            for i in idx[:n_short]:
                signal_map[ symbols[i] ] = -1.0 / n_short

    # -------------------------------------------------
    # METRICS
    # -------------------------------------------------

        metrics_by_symbol = {
            s: {
                "ic":  self._extract_metric( fo, s, "ic",  ("intel","ic")   ),
                "hit_ratio":  self._extract_metric(  fo,    s,  "hit_ratio",   ("basic","hit_ratio")    ),
                "sharpe": self._extract_metric(  fo,  s,  "sharpe",  ("risk","sharpe") ),
                "tstat_alpha":  self._extract_metric( fo,  s, "tstat_alpha",  ("risk","tstat_alpha")    ),
                "drawdown": self._extract_metric( fo, s, "drawdown", ("risk","drawdown")  ),
                "corr": self._extract_metric(fo, s, "corr_rm", ("basic","corr")  ),
                "cvar": self._extract_metric(fo,  s, "cvar",  ("risk","cvar") ),
                "volatility":  self._extract_metric( fo, s, "volatility", ("risk","volatility") ),                
                "alpha" :  self._extract_metric( fo, s, "alpha", ("alpha","alpha") ),              
                "beta" :  self._extract_metric( fo, s, "beta", ("alpha","beta") ),
                "turnover":  self._extract_metric( fo, s, "turnover", ("execution","turnover") ),
                "slippage":  self._extract_metric( fo, s, "slippage", ("execution","slippage") ),
                "kelly":  self._extract_metric( fo, s, "kelly", ("portfolio","kelly") ),
                "max_drawdown":  self._extract_metric( fo, s, "max_drawdown", ("risk","max_drawdown") ),
                "transaction_cost":  self._extract_metric( fo, s, "transaction_cost", ("execution","transaction_cost") )
            }

            for s in symbols

        }

        metrics = {
            "lookback":  window,
            "assets":    len(symbols),
            "benchmark": benchmark,
            "by_symbol": metrics_by_symbol
        }
        # -------------------------------------------------
        # CHART METRICS
        # -------------------------------------------------
        metric_series = {
            "ic": self._build_metric_series( fo, symbols,  "ic",  ("intel","ic")  ),
            "hit_ratio": self._build_metric_series( fo, symbols, "hit_ratio", ("basic","hit_ratio") ),
            "sharpe": self._build_metric_series( fo, symbols, "sharpe", ("risk","sharpe") ),
            "tstat_alpha": self._build_metric_series( fo, symbols, "tstat_alpha", ("risk","tstat_alpha")  ),
            "drawdown": self._build_metric_series( fo, symbols, "drawdown",  ("risk","drawdown")  ),
            "corr": self._build_metric_series( fo, symbols, "corr_rm",  ("basic","corr")   ),
            "cvar": self._build_metric_series( fo,symbols,"cvar",   ("risk","cvar")   ),
            "volatility":  self._build_metric_series( fo,  symbols,  "volatility",  ("risk","volatility")  ),
            "alpha" :  self._build_metric_series( fo, symbols, "alpha", ("alpha","alpha") ),              
            "beta" :  self._build_metric_series( fo, symbols, "beta", ("alpha","beta") ),
            "turnover":  self._build_metric_series( fo, symbols, "turnover", ("execution","turnover") ),
            "slippage":  self._build_metric_series( fo, symbols, "slippage", ("execution","slippage") ),
            "kelly": self._build_metric_series( fo, symbols, "kelly", ("portfolio","kelly") ),
            "max_drawdown":  self._build_metric_series( fo, symbols, "max_drawdown", ("risk","max_drawdown") ),
            "transaction_cost": self._build_metric_series( fo, symbols, "transaction_cost", ("execution","transaction_cost") )
        }

        #for name, series in metric_series.items():
        #    print(name, len(series["x"]), len(series["y"]), series["x"],series["y"] )
        # -------------------------------------------------
        # CHART DATA
        # -------------------------------------------------
        chartdata = {
            "price":  price,
            "benchmark":   benchmark,
            "signal":  signal,
            "alpha_scores":  {
                "x":  list(alpha_scores.keys()),
                "y":  list(alpha_scores.values())
            },
            "beta_scores": {
                "x":  list(beta_scores.keys()),
                "y":  list(beta_scores.values())
            },
            **metric_series,
            "assets":  symbols
        }
        chart = self.build_chart(
            charttype= chart_cfg.get("type", "bar" ),
            chartmode= "lines",
            title=     chart_cfg.get( "title", "Cross-sectional Alpha"  ),
            chartdata= chartdata, 
            series= series_cfg   )

        return StrategyResult(
            name= "AlphaStrategy",
            data=  chartdata,
            signals= signal_map,
            metrics= metrics,
            chart=   chart )


# ===========================================================
# END OF ALPHA STRATEGY
# ===========================================================
