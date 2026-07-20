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
class AlphaStrategy0(BaseStrategy):

    # -------------------------------------------------
    # INDEX HANDLING
    # -------------------------------------------------
    def _ensure_datetime_index(self, df):

        df = df.copy()

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
            df = df[df["date"] > pd.Timestamp("2000-01-01")]
            df = df.sort_values("date")
            df = df.set_index("date")
        else:
            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df[df.index.notna()]
            df = df[df.index > pd.Timestamp("2000-01-01")]
            df = df.sort_index()

        return df

    # -------------------------------------------------
    # METRICS EXTRACTION (FIXED MULTIINDEX SAFE)
    # -------------------------------------------------
    def _extract_metrics(self, fo, s):

        def get(cat, met):
            try:
                v = fo.loc[(cat, met), s]
                return float(v) if v is not None else None
            except Exception:
                return None

        return {
            "r_squared": get("basic", "r_squared"),
            "tstat": get("basic", "tstat"),
            "hit_ratio": get("basic", "hit_ratio"),
            "corr": get("basic", "corr"),
            "sharpe_ratio": get("risk", "sharpe"),
            "cvar": get("risk", "cvar"),
        }

    # -------------------------------------------------
    # CHART NORMALIZATION (IMPORTANT FIX)
    # -------------------------------------------------
    def _build_metric_series(self, fo, symbols, cat, metric):

        y = []
        for s in symbols:
            try:
                v = fo.loc[(cat, metric), s]
                y.append(float(v))
            except Exception:
                y.append(None)

        return {
            "x": symbols,
            "y": y
        }

    # -------------------------------------------------
    # MAIN RUN
    # -------------------------------------------------
    def run(self):

        window = self.lookback()

        benchmark = self.get_cfg(
            "benchmark",
            self.runtime_cfg.get("benchmark", "SPY")
        )

        top_q = self.get_cfg("top_quantile", 0.2)
        bottom_q = self.get_cfg("bottom_quantile", 0.2)

        chart_cfg = self.get_cfg("chart", {})
        series_cfg = chart_cfg.get("series", [])

        # ==================================================
        # VALIDATION
        # ==================================================
        if benchmark not in self.data:
            return StrategyResult(
                name="AlphaStrategy",
                data={},
                signals={},
                metrics={
                    "lookback": window,
                    "assets": 0,
                    "by_symbol": {}
                },
                chart=self.build_chart(charttype="bar", chartdata={})
            )

        # ==================================================
        # BENCHMARK DATA
        # ==================================================
        benchmark_df = self._ensure_datetime_index(self.data[benchmark])
        benchmark_ret = benchmark_df["ret"].dropna()

        price_series = benchmark_df["close"].dropna()

        price = {
            "x": price_series.index.to_pydatetime().tolist(),
            "y": price_series.values.tolist()
        }

        # ==================================================
        # FACTORS
        # ==================================================
        alpha_scores = {}
        beta_scores = {}

        # ==================================================
        # CROSS SECTION LOOP
        # ==================================================
        for symbol, df in self.data.items():

            if symbol == benchmark:
                continue

            if "ret" not in df.columns:
                continue

            df = self._ensure_datetime_index(df)
            asset_ret = df["ret"].dropna()

            if len(asset_ret) < window + 10:
                continue

            n = min(len(asset_ret), len(benchmark_ret))

            y = asset_ret.iloc[-n:].values
            x = benchmark_ret.iloc[-n:].values

            alpha_series = []
            beta_series = []

            for i in range(window, n):

                y_win = y[i - window:i]
                x_win = x[i - window:i]

                x_var = np.var(x_win)

                if x_var <= 1e-12:
                    continue

                beta = np.cov(y_win, x_win)[0, 1] / x_var
                alpha = np.mean(y_win) - beta * np.mean(x_win)

                if np.isfinite(alpha):
                    alpha_series.append(alpha)
                    beta_series.append(beta)

            if not alpha_series:
                continue

            alpha_scores[symbol] = float(np.mean(alpha_series))
            beta_scores[symbol] = float(np.mean(beta_series))

        # ==================================================
        # SIGNAL
        # ==================================================
        ordered = sorted(alpha_scores.items(), key=lambda x: x[1])

        symbols = list(alpha_scores.keys())
        scores = np.array(list(alpha_scores.values())) if alpha_scores else np.array([])

        signal = {
            "x": [k for k, _ in ordered],
            "y": [v for _, v in ordered]
        }

        signal_map = {}

        if len(symbols) > 0:

            sorted_idx = np.argsort(scores)

            n = len(symbols)
            n_long = max(1, int(n * top_q))
            n_short = max(1, int(n * bottom_q))

            for i in sorted_idx[-n_long:]:
                signal_map[symbols[i]] = 1.0 / n_long

            for i in sorted_idx[:n_short]:
                signal_map[symbols[i]] = -1.0 / n_short

        # ==================================================
        # CHART DATA (CLEAN + CONSISTENT)
        # ==================================================
        chartdata = {
            "price": price,
            "signal": signal,
            "alpha_scores": {
                "x": list(alpha_scores.keys()),
                "y": list(alpha_scores.values())
            },
            "beta_scores": {
                "x": list(beta_scores.keys()),
                "y": list(beta_scores.values())
            },
            "assets": symbols
        }

        # ==================================================
        # FORMULA OUTPUT
        # ==================================================
        fo = self.formulaOutput.assemble()

        # ==================================================
        # METRICS (FIXED STRUCTURE)
        # ==================================================
        metrics = {
            "lookback": window,
            "assets": len(symbols),
            "by_symbol": {
                s: self._extract_metrics(fo, s)
                for s in symbols
            }
        }

        # ==================================================
        # CHART BUILD (NO DIRECT FO ACCESS HERE)
        # ==================================================
        chart = self.build_chart(
            charttype=chart_cfg.get("type", "bar"),
            chartmode="lines",
            title=chart_cfg.get("title", "Cross-sectional Alpha"),
            chartdata=chartdata,
            series=series_cfg
        )

        return StrategyResult(
            name="AlphaStrategy",
            data=chartdata,
            signals=signal_map,
            metrics=metrics,
            chart=chart
        )
class AlphaStrategy1(BaseStrategy):

    # -------------------------------------------------
    # INDEX HANDLING
    # -------------------------------------------------
    def _ensure_datetime_index(self, df):
        df = df.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
            df = df[df["date"] > pd.Timestamp("2000-01-01")]
            df = df.sort_values("date")
            df = df.set_index("date")
        else:
            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df[df.index.notna()]
            df = df[df.index > pd.Timestamp("2000-01-01")]
            df = df.sort_index()
        return df


    # -------------------------------------------------
    # METRIC SERIES
    # -------------------------------------------------
    def _build_metric_series(self, fo, symbols, metric, report_key=None):
        values = []
        for symbol in symbols:
          try:
            value = self._extract_metric(
                fo,
                symbol,
                metric,
                report_key
            )
          except Exception as e:
            print(f"[WARN] {report_key} {symbol}: {e}")
            value = np.nan

          values.append(value)

        return {
        "x": symbols,
        "y": values,
        }


    # -------------------------------------------------
    # MAIN RUN
    # -------------------------------------------------
    def run(self):
        print("[DEBUG] starting run")

        # -------------------------------------------------
        # FORMULA ENGINE
        # -------------------------------------------------

        self.formulaOutput.assemble()
        fo = self.formulaOutput
        window = self.lookback()
        benchmark = self.get_cfg(
            "benchmark",
            self.runtime_cfg.get("benchmark", "SPY")
        )
        top_q = self.get_cfg(
            "top_quantile",
            0.2
        )
        bottom_q = self.get_cfg(
            "bottom_quantile",
            0.2
        )
        chart_cfg = self.get_cfg(
            "chart",
            {}
        )
        series_cfg = chart_cfg.get(
            "series",
            []
        )


        # -------------------------------------------------
        # VALIDATION
        # -------------------------------------------------
        #print("[DEBUG] benchmark =", benchmark)
        #print("[DEBUG] data keys =", list(self.data.keys()))
        # -------------------------------------------------
        # BENCHMARK
        # -------------------------------------------------
        if benchmark in self.data:
          benchmark_df = self._ensure_datetime_index(self.data[benchmark])
        else:
          print(f"[INFO] Benchmark '{benchmark}' not found. Using equal-weight benchmark.")
          returns = []
          closes = []
          for symbol, df in self.data.items():
            df = self._ensure_datetime_index(df)
            if "ret" in df:
              returns.append(df["ret"].rename(symbol))
            if "close" in df:
              closes.append(df["close"].rename(symbol))

          benchmark_ret = ( pd.concat(returns, axis=1).mean(axis=1).dropna() )
          price_series = ( pd.concat(closes, axis=1).mean(axis=1) )
          price = {
           "x": price_series.index.to_pydatetime().tolist(),
           "y": price_series.tolist()
          }


        # -------------------------------------------------
        # ALPHA / BETA
        # -------------------------------------------------
        alpha_scores = {}
        beta_scores = {}
        for symbol, df in self.data.items():
            if symbol == benchmark:
                continue
            if "ret" not in df.columns:
                continue
            df = self._ensure_datetime_index(df)
            asset_ret = ( df["ret"].dropna()  )
            if len(asset_ret) < window + 10:
                continue
            n = min( len(asset_ret), len(benchmark_ret) )
            y = asset_ret.iloc[-n:].values
            x = benchmark_ret.iloc[-n:].values

            alpha_series = []
            beta_series = []
            for i in range(window, n):
                y_win = y[i-window:i]
                x_win = x[i-window:i]
                x_var = np.var(x_win)
                if x_var <= 1e-12:
                    continue
                beta = ( np.cov( y_win, x_win )[0,1] / x_var )
                alpha = ( np.mean(y_win) -  beta * np.mean(x_win) )
                if np.isfinite(alpha):
                    alpha_series.append(alpha)
                    beta_series.append(beta)
            if alpha_series:
                alpha_scores[symbol] = float(
                    np.mean(alpha_series)
                )
                beta_scores[symbol] = float(
                    np.mean(beta_series)
                )

        # -------------------------------------------------
        # SIGNAL
        # -------------------------------------------------
        ordered = sorted( alpha_scores.items(), key=lambda x:x[1]      )
        symbols = list(  alpha_scores.keys()    )
        scores = np.array(  list(alpha_scores.values())       )
        signal = {
           "x":  [x[0] for x in ordered],
           "y":  [x[1] for x in ordered]
        }
        signal_map = {}
        if len(symbols):
            idx = np.argsort(scores)
            n = len(symbols)
            n_long = max( 1,   int(n * top_q)  )
            n_short = max( 1,  int(n * bottom_q)  )
            for i in idx[-n_long:]:
                signal_map[ symbols[i]  ] = 1.0/n_long
            for i in idx[:n_short]:
                signal_map[
                    symbols[i]
                ] = -1.0/n_short

        # -------------------------------------------------
        # METRICS
        # -------------------------------------------------

        metrics_by_symbol = {
            s: {
                "ic": self._extract_metric(  fo, s,  "ic",  ("basic","ic")  ),
                "hit_ratio": self._extract_metric( fo, s, "hit_ratio",  ("basic","hit_ratio") ),
                "sharpe":self._extract_metric( fo, s,"sharpe", ("risk","sharpe")  ),
                "tstat_alpha":  self._extract_metric( fo, s, "tstat_alpha", ("basic","tstat") ),
                "drawdown": self._extract_metric(  fo, s,"drawdown", ("risk","drawdown") ),
                "corr":  self._extract_metric( fo,  s, "corr",  ("basic","corr")  ),
                "cvar":  self._extract_metric( fo,  s, "cvar", ("risk","cvar")  ),
                "volatility":  self._extract_metric(   fo,  s,  "volatility",  ("risk","volatility")    )
            }
            for s in symbols
        }
        #print("[DEBUG] metrics_by_symbol ",  metrics_by_symbol )

        metrics = {
            "lookback": window,
            "assets": len(symbols),
            "by_symbol": metrics_by_symbol
        }

        # -------------------------------------------------
        # CHART METRICS
        # -------------------------------------------------
        metric_series = {
            "ic":  self._build_metric_series( fo, symbols,  "ic",  ("basic","ic") ),
            "hit_ratio": self._build_metric_series( fo, symbols, "hit_ratio",  ("basic","hit_ratio")  ),
            "sharpe":  self._build_metric_series( fo, symbols, "sharpe", ("risk","sharpe")  ),
            "tstat_alpha": self._build_metric_series( fo, symbols, "tstat_alpha", ("basic","tstat") ),
            "drawdown":  self._build_metric_series( fo,  symbols,  "drawdown",  ("risk","drawdown")   ),
            "corr":  self._build_metric_series( fo,  symbols,  "corr",   ("basic","corr")   ),
            "cvar":  self._build_metric_series( fo,  symbols,  "cvar",  ("risk","cvar")  ),
            "volatility":  self._build_metric_series( fo,  symbols,  "volatility",  ("risk","volatility")   )
        }
        #print("[DEBUG] metric_series ", metric_series)
        # -------------------------------------------------
        # CHART DATA
        # -------------------------------------------------
        chartdata = {
            "price": price,
            "signal": signal,
            "alpha_scores": {
                "x": list(alpha_scores.keys()),
                "y": list(alpha_scores.values())
            },
            "beta_scores": {
                "x": list(beta_scores.keys()),
                "y": list(beta_scores.values())
            },
            **metric_series,
            "assets": symbols
        }

        chart = self.build_chart(
            charttype=chart_cfg.get(
                "type",
                "bar"
            ),
            chartmode="lines",
            title=chart_cfg.get(
                "title",
                "Cross-sectional Alpha"
            ),
            chartdata=chartdata,
            series=series_cfg

        )

        return StrategyResult(
            name="AlphaStrategy",
            data=chartdata,
            signals=signal_map,
            metrics=metrics,
            chart=chart
        )
# ===========================================================
# END OF ALPHA STRATEGY
# ===========================================================
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