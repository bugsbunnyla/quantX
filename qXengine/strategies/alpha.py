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

# ===========================================================
# END OF ALPHA STRATEGY
# ===========================================================