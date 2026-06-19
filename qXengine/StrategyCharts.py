
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import math


# ==================================================
# BASE DATA CLASS (NO RENDER LOGIC)
# ==================================================
class StrategyChart:

    def __init__(
        self,
        charttype="line",
        chartmode="lines",
        title="",
        xaxis=None,
        yaxis=None,
        chartdata=None,
        series=None
    ):
        self.charttype = charttype
        self.chartmode = chartmode
        self.title = title
        self.xaxis = xaxis or []
        self.yaxis = yaxis or []
        self.chartdata = (lambda x:
    x if isinstance(x, pd.DataFrame)
    else (x if isinstance(x, dict) else {})
)(chartdata)
        self.series = series or {}


# ==================================================
# ALPHA STRATEGY VISUALIZER (PURE FUNCTION STYLE)
# ==================================================
class AlphaStrategyChart(StrategyChart):

    STATIC_MAP = {
        "series": ["signals", "returns", "metrics"],
        "chart": ["alpha_scores", "beta_scores"]
    }

    @staticmethod
    def _plot_dict(fig, d, name):
      if d is None:
        return False

      if not isinstance(d, dict):
        return False

      if len(d) == 0:
        return False

      fig.add_trace(
        go.Bar(
            x=list(d.keys()),
            y=list(d.values()),
            name=name
        )
      )

      return True
    @staticmethod
    def _plot_series(fig, v, name):
        if v is None:
            return False

        try:
            v = list(v)
        except Exception:
            return False

        fig.add_trace(
            go.Scatter(
                y=v,
                mode="lines",
                name=name
            )
        )
        return True

    @staticmethod
    def get_attr_or_key(obj, name, default=None):
        if isinstance(obj, dict):
           return obj.get(name, default)
        return getattr(obj, name, default)
    # ==================================================
    # MAIN RENDER (FIXED CONTRACT)
    # ==================================================
    @staticmethod
    def render(item):

      fig = go.Figure()
      metrics = getattr(item, "metrics", None) or {}
      series = getattr(item, "series", None) or {}

      # normalize safely
      if hasattr(metrics, "__dict__"):
          metrics = vars(metrics)

      if hasattr(series, "__dict__"):
          series = vars(series)

      # -----------------------
      # SERIES
      # -----------------------
      metrics_series = item.metrics
            
      AlphaStrategyChart._plot_series(fig, item.signals, "signals")
      AlphaStrategyChart._plot_series(fig, metrics.get("returns"), "returns")
      #f isinstance(metrics_series, dict):
      #   AlphaStrategyChart._plot_dict(fig, metrics_series, "metrics")
      # -----------------------
      # CHART METRICS
      # -----------------------
      alpha = metrics.get("alpha_scores")
      beta = metrics.get("beta_scores")

      #print("[DEBUG] alpha:", type(alpha), alpha is not None)
      #print("[DEBUG] beta:", type(beta), beta is not None)

      # FIX: normalize before plotting
      if isinstance(alpha, dict):
          AlphaStrategyChart._plot_dict(fig, alpha, "alpha_scores")
      elif isinstance(alpha, (list, tuple, np.ndarray, pd.Series)):
          AlphaStrategyChart._plot_series(fig, alpha, "alpha_scores")

      if isinstance(beta, dict):
          AlphaStrategyChart._plot_dict(fig, beta, "beta_scores")
      elif isinstance(beta, (list, tuple, np.ndarray, pd.Series)):
          AlphaStrategyChart._plot_series(fig, beta, "beta_scores")

      fig.update_layout(
          template="plotly_dark",
          barmode="group",
          title=getattr(item, "name", "AlphaStrategy")
      )

      return fig



# ==================================================
# STRATEGY CHARTS (UNCHANGED BUT SAFE)
# ==================================================
class StrategyCharts:

    @staticmethod
    def _extract_series(obj):

        out = {}

        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (list, tuple, np.ndarray, pd.Series)):
                    out[k] = pd.Series(v)
            return out

        if isinstance(obj, (list, tuple, np.ndarray, pd.Series)):
            return {"value": pd.Series(obj)}

        return out

    @staticmethod
    def _build_plot_bundle(result):

        plot = {}

        plot.update(StrategyCharts._extract_series(result.signals))
        plot.update(StrategyCharts._extract_series(result.metrics))

        data_plot = StrategyCharts._extract_series(result.data)

        for k, v in data_plot.items():
            if k not in plot:
                plot[k] = v

        return plot

    @staticmethod
    def render(tabs):

        figs = {}

        for tab_name, items in tabs.items():

            if not items:
                continue

            n = len(items)
            cols = 2 if n <= 4 else 3
            rows = math.ceil(n / cols)

            fig = make_subplots(
                rows=rows,
                cols=cols,
                subplot_titles=[i.cfg.get("title", i.result.name) for i in items]
            )

            for idx, s in enumerate(items):

                r = idx // cols + 1
                c = idx % cols + 1

                result = s.result
                plot_data = StrategyCharts._build_plot_bundle(result)

                if not plot_data:
                    continue

                for name, series in plot_data.items():

                    if series is None or len(series) == 0:
                        continue

                    fig.add_trace(
                        go.Scatter(y=series, name=name),
                        row=r,
                        col=c
                    )

            fig.update_layout(
                title=f"{tab_name.upper()} DASHBOARD",
                template="plotly_dark",
                height=max(400, 350 * rows),
                showlegend=False
            )

            figs[tab_name] = fig

        return figs

# =====================================================
# IntradayStrategyChart
# =====================================================
import pandas as pd
import plotly.graph_objects as go


class IntradayStrategyChart(StrategyChart):

    # -------------------------------------------------
    @staticmethod
    def _clean(series):

        if series is None:
            return None

        try:
            s = pd.Series(series).copy()
        except:
            return None

        s.index = pd.to_datetime(s.index, errors="coerce")
        s = s[s.index.notna()]
        s = s[s.index > pd.Timestamp("2000-01-01")]
        s = s.sort_index()

        if s.empty:
            return None

        return s

    # -------------------------------------------------
    @staticmethod
    def _plot_series(fig, series, name, color=None):

        series = IntradayStrategyChart._clean(series)
        if series is None:
            return False

        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                name=name,
                line=dict(color=color) if color else None
            )
        )
        return True

    # -------------------------------------------------
    @staticmethod
    def _plot_signal(fig, series, name, color=None):

        series = IntradayStrategyChart._clean(series)
        if series is None:
            return False

        series = series.fillna(0)

        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                line_shape="hv",
                name=name,
                opacity=0.7,
                line=dict(color=color) if color else None
            )
        )
        return True

    # -------------------------------------------------
    @staticmethod
    def _plot_markers(fig, series, name):

        series = IntradayStrategyChart._clean(series)
        if series is None:
            return False

        points = series[series > 0]
        if points.empty:
            return False

        fig.add_trace(
            go.Scatter(
                x=points.index,
                y=points.values,
                mode="markers",
                name=name,
                marker=dict(size=6, color="red")
            )
        )
        return True

    # -------------------------------------------------
    @staticmethod
    def render(item):

        fig = go.Figure()

        signals = getattr(item, "signals", {}) or {}

        signal_map = signals.get("signal", {})
        volume_map = signals.get("volume_stress", {})
        event_map = signals.get("dislocation_events", {})
        score_map = signals.get("score", {})

        for sym in signal_map.keys():

            IntradayStrategyChart._plot_series(fig, signal_map.get(sym), f"{sym} Signal", "steelblue")
            IntradayStrategyChart._plot_series(fig, volume_map.get(sym), f"{sym} Volume", "orange")
            IntradayStrategyChart._plot_series(fig, score_map.get(sym), f"{sym} Score", "gray")

        for sym, events in event_map.items():
            IntradayStrategyChart._plot_markers(fig, events, f"{sym} Events")

        fig.add_hline(y=0, line_color="gray", opacity=0.4)

        fig.update_layout(
            template="plotly_dark",
            title=getattr(item, "name", "Intraday Strategy"),
            xaxis_title="Time",
            yaxis_title="Signal",
            hovermode="x unified"
        )

        return fig
# ===================================================
# Correlation Fallback Chart
# ===================================================
import pandas as pd
import plotly.graph_objects as go


class CorrelationFallbackChart(StrategyChart):

    # ==================================================
    # STATIC MAP (MATCHES CorrelationFallback OUTPUT)
    # ==================================================
    STATIC_MAP = {
        "metrics": [
            "lookback",
            "corr_window",
            "signal_threshold",
            "low_corr_threshold",
            "dispersion_window",
            "avg_corr",
            "dispersion",
            "regime"
        ],
        "signals": [
            "signal",
            "correlation",
            "dispersion",
            "regime",
            "equity_curve",
            "threshold_high",
            "threshold_low"
        ],
        "chart": []
    }

    # ==================================================
    # CORE CLEANER (FIX 1970 / BAD INDEX)
    # ==================================================
    @staticmethod
    def _clean(series):

        if series is None:
            return None

        try:
            s = pd.Series(series).copy()
        except:
            return None

        # numeric cleanup (safe for all signal types)
        s = pd.to_numeric(s, errors="coerce")

        # force datetime index
        s.index = pd.to_datetime(s.index, errors="coerce")

        # remove bad timestamps (NaT + 1970 garbage)
        s = s[s.index.notna()]
        s = s[s.index > pd.Timestamp("2000-01-01")]

        # sort + deduplicate
        s = s.sort_index()
        s = s[~s.index.duplicated(keep="last")]

        if len(s) < 2:
            return None

        return s

    # ==================================================
    # PLOT HELPER
    # ==================================================
    @staticmethod
    def _plot_series(fig, series, name, color=None):

        s = CorrelationFallbackChart._clean(series)

        if s is None:
            return False

        fig.add_trace(
            go.Scatter(
                x=s.index,
                y=s.values,
                mode="lines",
                name=name,
                line=dict(color=color) if color else None
            )
        )

        return True

    # ==================================================
    # STEP STYLE (REGIME)
    # ==================================================
    @staticmethod
    def _plot_step(fig, series, name, color=None):

        s = CorrelationFallbackChart._clean(series)

        if s is None:
            return False

        fig.add_trace(
            go.Scatter(
                x=s.index,
                y=s.values,
                mode="lines",
                line_shape="hv",
                name=name,
                opacity=0.7,
                line=dict(color=color) if color else None
            )
        )

        return True

    # ==================================================
    # MAIN RENDER
    # ==================================================
    @staticmethod
    def render(item):

        fig = go.Figure()

        signals = getattr(item, "signals", {}) or {}
        metrics = getattr(item, "metrics", {}) or {}

        signal = signals.get("signal")
        correlation = signals.get("correlation")
        dispersion = signals.get("dispersion")
        regime = signals.get("regime")
        equity = signals.get("equity_curve")

        threshold_high = signals.get("threshold_high")
        threshold_low = signals.get("threshold_low")

        # ==================================================
        # 1. CORRELATION
        # ==================================================
        CorrelationFallbackChart._plot_series(
            fig,
            correlation,
            "Rolling Correlation",
            "orange"
        )

        # ==================================================
        # 2. SIGNAL
        # ==================================================
        CorrelationFallbackChart._plot_series(
            fig,
            signal,
            "Smoothed Regime Signal",
            "cyan"
        )

        # ==================================================
        # 3. DISPERSION
        # ==================================================
        CorrelationFallbackChart._plot_series(
            fig,
            dispersion,
            "Market Dispersion",
            "red"
        )

        # ==================================================
        # 4. EQUITY CURVE
        # ==================================================
        CorrelationFallbackChart._plot_series(
            fig,
            equity,
            "Equity Curve (Proxy)",
            "white"
        )

        # ==================================================
        # 5. THRESHOLD LINES
        # ==================================================
        CorrelationFallbackChart._plot_series(
            fig,
            threshold_high,
            "High Correlation Threshold",
            "red"
        )

        CorrelationFallbackChart._plot_series(
            fig,
            threshold_low,
            "Low Correlation Threshold",
            "green"
        )

        # ==================================================
        # 6. REGIME (STEP)
        # ==================================================
        CorrelationFallbackChart._plot_step(
            fig,
            regime,
            "Regime State",
            "yellow"
        )

        # ==================================================
        # LAYOUT
        # ==================================================
        fig.update_layout(
            template="plotly_dark",
            title=getattr(item, "name", "Correlation Fallback Regime"),
            xaxis_title="Date",
            yaxis_title="Value",
            legend_title="Series",
            hovermode="x unified"
        )

        return fig

# ===================================================
# Pair Trading Fallback Chart
# ===================================================
import pandas as pd
import plotly.graph_objects as go

class PairTradingFallbackChart(StrategyChart):

    STATIC_MAP = {
        "metrics": [
            "lookback",
            "corr_window",
            "min_corr",
            "entry_zscore",
            "exit_zscore",
            "max_pairs",
            "min_half_life",
            "pairs_used",
            "avg_half_life"
        ],
        "signals": [
            "spread",
            "signal"
        ],
        "chart": []
    }

    # ==================================================
    # CORE FIX (1970 REMOVAL)
    # ==================================================
    @staticmethod
    def _clean(series):

        if series is None:
            return None

        try:
            s = pd.Series(series).copy()
        except:
            return None

        s = pd.to_numeric(s, errors="coerce").dropna()

        s.index = pd.to_datetime(s.index, errors="coerce")

        s = s[s.index.notna()]

        s = s[s.index > pd.Timestamp("2000-01-01")]

        s = s.sort_index()

        s = s[~s.index.duplicated(keep="last")]

        if len(s) < 2:
            return None

        return s

    # ==================================================
    # PLOT HELPERS
    # ==================================================
    @staticmethod
    def _plot_series(fig, series, name, color=None):

        s = PairTradingFallbackChart._clean(series)

        if s is None:
            return False

        fig.add_trace(
            go.Scatter(
                x=s.index,
                y=s.values,
                mode="lines",
                name=name,
                line=dict(color=color) if color else None
            )
        )
        return True

    @staticmethod
    def _plot_step(fig, series, name):

        s = PairTradingFallbackChart._clean(series)

        if s is None:
            return False

        fig.add_trace(
            go.Scatter(
                x=s.index,
                y=s.values,
                mode="lines",
                line_shape="hv",
                name=name,
                opacity=0.65
            )
        )
        return True

    # ==================================================
    # MAIN RENDER
    # ==================================================
    @staticmethod
    def render(item):

        fig = go.Figure()

        signals = getattr(item, "signals", {}) or {}
        metrics = getattr(item, "metrics", {}) or {}

        spreads = signals.get("spread", {})
        trade_signals = signals.get("signal", {})

        entry_z = metrics.get("entry_zscore")
        exit_z = metrics.get("exit_zscore")

        # ==================================================
        # SPREAD + ZSCORE
        # ==================================================
        for pair, pair_data in spreads.items():

            if not isinstance(pair_data, dict):
                continue

            PairTradingFallbackChart._plot_series(
                fig,
                pair_data.get("spread"),
                f"{pair} Spread",
                "steelblue"
            )

            PairTradingFallbackChart._plot_series(
                fig,
                pair_data.get("zscore"),
                f"{pair} Z-Score",
                "orange"
            )

        # ==================================================
        # SIGNALS
        # ==================================================
        for pair, signal in trade_signals.items():

            PairTradingFallbackChart._plot_step(
                fig,
                signal,
                f"{pair} Signal"
            )

        # ==================================================
        # ENTRY / EXIT LINES
        # ==================================================
        if entry_z is not None:

            fig.add_hline(
                y=entry_z,
                line_dash="dash",
                line_color="red"
            )
            fig.add_hline(
                y=-entry_z,
                line_dash="dash",
                line_color="red"
            )

        if exit_z is not None:

            fig.add_hline(
                y=exit_z,
                line_dash="dot",
                line_color="green"
            )
            fig.add_hline(
                y=-exit_z,
                line_dash="dot",
                line_color="green"
            )

        # ==================================================
        # LAYOUT
        # ==================================================
        fig.update_layout(
            template="plotly_dark",
            title=getattr(item, "name", "Pair Trading Fallback"),
            xaxis_title="Date",
            yaxis_title="Spread / Z-Score / Signal",
            legend_title="Pairs",
            hovermode="x unified"
        )

        return fig

# ===================================================
# ForecastStrategyChart
#====================================================
import pandas as pd
import plotly.graph_objects as go

class ForecastStrategyChart(StrategyChart):

    STATIC_MAP = {
        "metrics": [
            "lookback",
            "forecast_horizon",
            "train_window",
            "assets"
        ],
        "chart": [
            "forecast",
            "actual"
        ]
    }

    # ==================================================
    # CORE FIX: REMOVE 1970 / BAD INDEXES
    # ==================================================
    @staticmethod
    def _clean(series):

        if series is None:
            return None

        try:
            s = pd.Series(series).copy()
        except Exception:
            return None

        s = pd.to_numeric(s, errors="coerce").dropna()

        s.index = pd.to_datetime(s.index, errors="coerce")

        s = s[s.index.notna()]

        s = s[s.index > pd.Timestamp("2000-01-01")]

        s = s.sort_index()

        s = s[~s.index.duplicated(keep="last")]

        if len(s) < 2:
            return None

        return s

    # ==================================================
    # PLOT HELPER
    # ==================================================
    @staticmethod
    def _plot_series(fig, series, name, color=None, dashed=False, opacity=1.0):

        s = ForecastStrategyChart._clean(series)

        if s is None:
            return False

        fig.add_trace(
            go.Scatter(
                x=s.index,
                y=s.values,
                mode="lines",
                name=name,
                opacity=opacity,
                line=dict(
                    color=color,
                    dash="dash" if dashed else None
                ) if color else dict(
                    dash="dash" if dashed else None
                )
            )
        )

        return True

    # ==================================================
    # MAIN RENDER
    # ==================================================
    @staticmethod
    def render(item):

        fig = go.Figure()

        signals = getattr(item, "signals", {}) or {}
        metrics = getattr(item, "metrics", {}) or {}

        forecast_data = signals.get("forecast", {})
        actual_data = signals.get("actual", {})

        # ==================================================
        # PLOTS
        # ==================================================
        for sym, forecast in forecast_data.items():

            actual = actual_data.get(sym)

            ForecastStrategyChart._plot_series(
                fig,
                forecast,
                f"{sym} Forecast Return",
                color="orange"
            )

            ForecastStrategyChart._plot_series(
                fig,
                actual,
                f"{sym} Actual Return",
                color="steelblue"
            )

        # ==================================================
        # ZERO LINE
        # ==================================================
        fig.add_hline(
            y=0,
            line_dash="dot",
            line_color="gray"
        )

        # ==================================================
        # ANNOTATION
        # ==================================================
        fig.add_annotation(
            text=(
                f"Assets: {metrics.get('assets', 0)} | "
                f"Horizon: {metrics.get('forecast_horizon', 0)}"
            ),
            x=0.01,
            y=0.99,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color="white", size=11)
        )

        # ==================================================
        # LAYOUT
        # ==================================================
        fig.update_layout(
            template="plotly_dark",
            title=getattr(item, "name", "Forecast Strategy"),
            xaxis_title="Date",
            yaxis_title="Return",
            legend_title="Series",
            hovermode="x unified"
        )

        return fig
# ===================================================
# Industry Momentum Chart
# ===================================================
import pandas as pd
import plotly.graph_objects as go


class IndustryMomentumChart(StrategyChart):

    STATIC_MAP = {
        "metrics": [
            "formation",
            "industry_window",
            "top_quantile",
            "holding",
            "stocks",
            "industries",
            "avg_stock_momentum",
            "avg_industry_momentum",
            "best_industry",
            "worst_industry"
        ],
        "signals": [
            "stock_signal",
            "industry_signal",
            "rebalance_events"
        ],
        "chart": [
            "portfolio_momentum",
            "industry_strength",
            "momentum_ma",
            "stock_momentum",
            "benchmark"
        ]
    }

    # ==================================================
    # STANDARD SERIES
    # ==================================================
    @staticmethod
    def _plot_series(fig, series, name, color=None):

        if series is None:
            return False

        try:
            series = pd.Series(series).dropna()
        except Exception:
            return False

        if series.empty:
            return False

        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                name=name,
                line=dict(color=color, width=2)
                if color else dict(width=2)
            )
        )

        return True

    @staticmethod
    def _to_series(x):

        if x is None:
            return None

        if isinstance(x, pd.DataFrame):
            x = x.iloc[:, 0]

        if isinstance(x, pd.Series):
            return x.copy()

        return pd.Series(x)

    # ==================================================
    # SAFE MARKERS 
    # ==================================================
    @staticmethod
    def _plot_markers(fig, signal, label, ref, color, symbol):

        signal = IndustryMomentumChart._to_series(signal)
        ref = IndustryMomentumChart._to_series(ref)

        if signal is None or ref is None:
            return False

        signal = signal.astype(float).fillna(0)
        ref = ref.astype(float)

        active = signal[signal > 0]

        if active.empty:
            return False

        # safe alignment (NEVER produces 2D)
        ref_aligned = ref.reindex(active.index)

        # FINAL HARD SAFETY
        x = ref_aligned.index.astype(str).tolist()
        y = np.asarray(ref_aligned.values).reshape(-1)

        if len(x) == 0 or len(y) == 0:
            return False

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name=label,
                marker=dict(
                    size=12,
                    color=color,
                    symbol=symbol,
                    line=dict(width=1, color="white")
                )
            )
        )

        return True


    # ==================================================
    # REBALANCE EVENTS
    # ==================================================
    @staticmethod
    def _plot_rebalance(fig, rebalance, ref):

        rebalance = IndustryMomentumChart._to_series(rebalance)
        ref = IndustryMomentumChart._to_series(ref)

        if rebalance is None or ref is None:
            return False

        rebalance = rebalance.fillna(0)

        active = rebalance[rebalance > 0]

        if active.empty:
            return False

        ref_aligned = ref.reindex(active.index)

        x = ref_aligned.index.astype(str).tolist()
        y = np.asarray(ref_aligned.values).reshape(-1)

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name="Rebalance Events",
                marker=dict(
                    size=12,
                    color="hotpink",
                    symbol="x",
                    line=dict(width=1, color="white")
                )
            )
        )

        return True



    # ==================================================
    # MAIN RENDER
    # ==================================================
    @staticmethod
    def render(item):

        fig = go.Figure()

        signals = getattr(item, "signals", None)
        metrics = getattr(item, "metrics", None)
        chart = getattr(item, "chart", None)

        chartdata = getattr(chart, "chartdata", None)

        if chartdata is None:
            chartdata = pd.DataFrame()

        if signals is None:
            signals = pd.DataFrame()

        if metrics is None:
            metrics = pd.DataFrame()

        # ==================================================
        # SIGNALS
        # ==================================================
        stock_signal = signals.get("stock_signal")
        industry_signal = signals.get("industry_signal")
        rebalance = signals.get("rebalance_events")

        # ==================================================
        # CHART DATA
        # ==================================================
        portfolio_momentum = chartdata.get("portfolio_momentum")
        industry_strength = chartdata.get("industry_strength")

        # FIXED TYPO
        momentum_ma = chartdata.get("momentum_ma")

        stock_momentum = chartdata.get("stock_momentum")
        benchmark = chartdata.get("benchmark")
        
        # ==================================================
        # METADATA
        # ==================================================
        formation = metrics.get("formation")
        holding = metrics.get("holding")
        top_quantile = metrics.get("top_quantile")

        stocks = metrics.get("stocks")
        industries = metrics.get("industries")

        best = metrics.get("best_industry")
        worst = metrics.get("worst_industry")

        # ==================================================
        # LEGEND LABELS
        # ==================================================
        portfolio_label = (
            f"Portfolio "
            f"(F={formation}, H={holding}, Q={top_quantile})"
        )

        industry_label = (
            f"Industry Strength "
            f"({industries} Industries)"
        )

        stock_label = (
            f"Stock Momentum "
            f"({stocks} Stocks)"
        )
        momentum_ma_label = (
            f"Smooth Momentum MA "
            f"({stocks} Stocks)"
        )

        industry_label = (
            f"Benchmark "
            f"SPY"
        )

        # ==================================================
        # PORTFOLIO
        # ==================================================
        IndustryMomentumChart._plot_series(
            fig,
            portfolio_momentum,
            portfolio_label,
            "cyan"
        )

        # ==================================================
        # INDUSTRY STRENGTH
        # ==================================================
        IndustryMomentumChart._plot_series(
            fig,
            industry_strength,
            industry_label,
            "orange"
        )

        # ==================================================
        # MOMENTUM MA
        # ==================================================
        IndustryMomentumChart._plot_series(
            fig,
            momentum_ma,
            "Momentum MA",
            "yellow"
        )

        # ==================================================
        # STOCK MOMENTUM
        # ==================================================
        IndustryMomentumChart._plot_series(
            fig,
            stock_momentum,
            stock_label,
            "crimson"
        )

        # ==================================================
        # BENCHMARK
        # ==================================================
        IndustryMomentumChart._plot_series(
            fig,
            benchmark,
            "Benchmark (SPY)",
            "white"
        )

        # ==================================================
        # SIGNALS
        # ==================================================
        IndustryMomentumChart._plot_markers(
            fig,
            industry_signal,
            "Industry Signal",
            portfolio_momentum,
            color="violet",
            symbol="triangle-up"
        )

        IndustryMomentumChart._plot_markers(
            fig,
            stock_signal,
            "Stock Signal",
            portfolio_momentum,
            color="lime",
            symbol="circle"
        )

        # ==================================================
        # REBALANCES
        # ==================================================
        IndustryMomentumChart._plot_rebalance(

            fig,
            rebalance,
            portfolio_momentum
        )

        # ==================================================
        # ANNOTATION
        # ==================================================
        if best is not None or worst is not None:

            fig.add_annotation(
                text=(
                    f"<b>Best Industry:</b> {best}"
                    f"<br>"
                    f"<b>Worst Industry:</b> {worst}"
                ),
                x=0.01,
                y=1.10,
                xref="paper",
                yref="paper",
                showarrow=False,
                align="left",
                bgcolor="rgba(30,30,30,0.8)",
                bordercolor="gray",
                borderwidth=1
            )

        # ==================================================
        # LAYOUT
        # ==================================================
        fig.update_layout(
            template="plotly_dark",
            title=getattr(
                item,
                "name",
                "Industry Momentum Strategy"
            ),
            xaxis_title="Date",
            yaxis_title="Normalized Performance",
            legend_title="Series / Signals",
            hovermode="x unified",
            showlegend=True,
            margin=dict(r=220)
        )

        return fig
# ===================================================
# DispersionStrategy Chart 
# ===================================================

import pandas as pd
import plotly.graph_objects as go
class DispersionStrategyChart(StrategyChart):

    STATIC_MAP = {
        "metrics": [
            "lookback",
            "cross_sectional_window",
            "mean_dispersion",
            "dispersion_volatility",
            "assets",
            "latest_dispersion",
            "latest_zscore"
        ],
        "chart": [
            "dispersion",
            "ma_63",
            "dispersion_volatility",
            "zscore",
            "momentum",
            "regime_switches"
        ]
    }

    # ==================================================
    # CORE FIX: CLEAN TIME INDEX (1970 REMOVAL)
    # ==================================================
    @staticmethod
    def _clean(series):

        if series is None:
            return None

        try:
            s = pd.Series(series).copy()
        except Exception:
            return None

        s = pd.to_numeric(
            s,
            errors="coerce"
        ).dropna()

        # Force datetime index
        s.index = pd.to_datetime(
            s.index,
            errors="coerce"
        )

        # Remove invalid timestamps
        s = s[s.index.notna()]

        # Remove epoch/garbage dates
        s = s[
            s.index > pd.Timestamp("2000-01-01")
        ]

        # Sort and deduplicate
        s = s.sort_index()

        s = s[
            ~s.index.duplicated(
                keep="last"
            )
        ]

        if len(s) < 2:
            return None

        return s

    # ==================================================
    # LINE PLOT HELPER
    # ==================================================
    @staticmethod
    def _plot_series(
        fig,
        series,
        name,
        color=None,
        dashed=False,
        opacity=1.0
    ):

        s = DispersionStrategyChart._clean(series)

        if s is None:
            return False

        fig.add_trace(
            go.Scatter(
                x=s.index,
                y=s.values,
                mode="lines",
                name=name,
                opacity=opacity,
                line=dict(
                    color=color,
                    dash="dash" if dashed else None
                ) if color else dict(
                    dash="dash" if dashed else None
                )
            )
        )

        return True

    # ==================================================
    # STEP REGIME HELPER
    # ==================================================
    @staticmethod
    def _plot_step(
        fig,
        series,
        name,
        color=None
    ):

        s = DispersionStrategyChart._clean(series)

        if s is None:
            return False

        fig.add_trace(
            go.Scatter(
                x=s.index,
                y=s.values,
                mode="lines",
                line_shape="hv",
                name=name,
                opacity=0.6,
                line=dict(
                    color=color
                ) if color else None
            )
        )

        return True

    # ==================================================
    # MAIN RENDER
    # ==================================================
    @staticmethod
    def render(item):

        fig = go.Figure()

        signals = getattr(
            item,
            "signals",
            {}
        ) or {}

        metrics = getattr(
            item,
            "metrics",
            {}
        ) or {}

        # ==================================================
        # SIGNALS
        # ==================================================
        dispersion = signals.get(
            "dispersion"
        )

        ma_63 = signals.get(
            "ma_63"
        )

        dispersion_vol = signals.get(
            "dispersion_volatility"
        )

        zscore = signals.get(
            "zscore"
        )

        momentum = signals.get(
            "momentum"
        )

        regime_switches = signals.get(
            "regime_switches"
        )

        # ==================================================
        # DISPERSION
        # ==================================================
        DispersionStrategyChart._plot_series(
            fig,
            dispersion,
            "Cross-Sectional Dispersion",
            color="steelblue"
        )

        # ==================================================
        # MOVING AVERAGE
        # ==================================================
        DispersionStrategyChart._plot_series(
            fig,
            ma_63,
            "Dispersion MA",
            color="orange",
            dashed=True
        )

        # ==================================================
        # VOLATILITY
        # ==================================================
        DispersionStrategyChart._plot_series(
            fig,
            dispersion_vol,
            "Dispersion Volatility",
            color="yellow"
        )

        # ==================================================
        # Z-SCORE
        # ==================================================
        DispersionStrategyChart._plot_series(
            fig,
            zscore,
            "Dispersion Z-Score",
            color="white"
        )

        # ==================================================
        # MOMENTUM
        # ==================================================
        DispersionStrategyChart._plot_series(
            fig,
            momentum,
            "Dispersion Momentum",
            color="cyan"
        )

        # ==================================================
        # REGIME
        # ==================================================
        DispersionStrategyChart._plot_step(
            fig,
            regime_switches,
            "Regime Switches",
            color="green"
        )

        # ==================================================
        # MEAN DISPERSION LINE
        # ==================================================
        mean_dispersion = metrics.get(
            "mean_dispersion"
        )

        if mean_dispersion is not None:

            fig.add_hline(
                y=mean_dispersion,
                line_dash="dot",
                line_color="red",
                annotation_text=(
                    f"Mean Dispersion "
                    f"{round(mean_dispersion, 4)}"
                )
            )

        # ==================================================
        # LATEST DISPERSION
        # ==================================================
        latest_dispersion = metrics.get(
            "latest_dispersion"
        )

        if latest_dispersion is not None:

            fig.add_annotation(
                text=(
                    f"Latest Dispersion: "
                    f"{latest_dispersion:.4f}"
                ),
                x=0.01,
                y=0.99,
                xref="paper",
                yref="paper",
                showarrow=False
            )

        # ==================================================
        # LATEST ZSCORE
        # ==================================================
        latest_zscore = metrics.get(
            "latest_zscore"
        )

        if latest_zscore is not None:

            fig.add_annotation(
                text=(
                    f"Latest Z-Score: "
                    f"{latest_zscore:.2f}"
                ),
                x=0.01,
                y=0.94,
                xref="paper",
                yref="paper",
                showarrow=False
            )

        # ==================================================
        # LAYOUT
        # ==================================================
        fig.update_layout(
            template="plotly_dark",
            title=getattr(
                item,
                "name",
                "Dispersion Strategy"
            ),
            xaxis_title="Date",
            yaxis_title="Dispersion",
            legend_title="Series",
            hovermode="x unified"
        )

        return fig

# ========================================================
# VolatilityChart
# ========================================================
import pandas as pd
import plotly.graph_objects as go

class VolatilityStrategyChart(StrategyChart):

    STATIC_MAP = {
        "metrics": [
            "lookback",
            "vol_window",
            "target_vol",
            "symbols",
            "valid_symbols"
        ],
        "chart": [
            "vol_series",
            "vol_signal"
        ]
    }

    # ==================================================
    # CORE FIX: REMOVE 1970 / BAD INDEXES
    # ==================================================
    @staticmethod
    def _clean(series):

        if series is None:
            return None

        try:
            s = pd.Series(series).copy()
        except Exception:
            return None

        s = pd.to_numeric(
            s,
            errors="coerce"
        ).dropna()

        try:

            s.index = pd.to_datetime(
                s.index,
                errors="coerce"
            )

        except Exception:
            return None

        s = s[
            s.index.notna()
        ]

        s = s[
            s.index >
            pd.Timestamp("2000-01-01")
        ]

        s = s.sort_index()

        s = s[
            ~s.index.duplicated(
                keep="last"
            )
        ]

        if len(s) < 2:
            return None

        return s

    # ==================================================
    # LINE HELPER
    # ==================================================
    @staticmethod
    def _plot_series(
        fig,
        series,
        name,
        color=None,
        dashed=False,
        opacity=1.0
    ):

        s = VolatilityStrategyChart._clean(
            series
        )

        if s is None:
            return False

        fig.add_trace(
            go.Scatter(
                x=s.index,
                y=s.values,
                mode="lines",
                name=name,
                opacity=opacity,
                line=dict(
                    color=color,
                    dash="dash" if dashed else None
                ) if color else dict(
                    dash="dash" if dashed else None
                )
            )
        )

        return True

    # ==================================================
    # MAIN RENDER
    # ==================================================
    @staticmethod
    def render(item):

        fig = go.Figure()

        signals = getattr(
            item,
            "signals",
            {}
        ) or {}

        metrics = getattr(
            item,
            "metrics",
            {}
        ) or {}

        vol_series = signals.get(
            "vol_series",
            {}
        )

        vol_signals = signals.get(
            "vol_signal",
            {}
        )

        # ==================================================
        # VOLATILITY
        # ==================================================
        for sym, data in vol_series.items():

            if not isinstance(
                data,
                dict
            ):
                continue

            volatility = data.get(
                "volatility"
            )

            target_vol = data.get(
                "target_vol"
            )

            VolatilityStrategyChart._plot_series(
                fig,
                volatility,
                f"{sym} Realized Vol",
                color="steelblue"
            )

            VolatilityStrategyChart._plot_series(
                fig,
                target_vol,
                f"{sym} Target Vol",
                color="orange",
                dashed=True
            )

        # ==================================================
        # VOL SIGNALS
        # ==================================================
        for sym, signal in vol_signals.items():

            VolatilityStrategyChart._plot_series(
                fig,
                signal,
                f"{sym} Vol Signal",
                color="green"
            )

        # ==================================================
        # GLOBAL TARGET VOL
        # ==================================================
        target_vol = metrics.get(
            "target_vol"
        )

        if target_vol is not None:

            fig.add_hline(
                y=target_vol,
                line_dash="dash",
                line_color="red",
                annotation_text=(
                    f"Target Vol "
                    f"{target_vol:.2f}"
                )
            )

        # ==================================================
        # SUMMARY
        # ==================================================
        fig.add_annotation(
            text=(
                f"Assets: "
                f"{metrics.get('valid_symbols',0)}"
            ),
            x=0.01,
            y=0.99,
            xref="paper",
            yref="paper",
            showarrow=False
        )

        # ==================================================
        # LAYOUT
        # ==================================================
        fig.update_layout(
            template="plotly_dark",
            title=getattr(
                item,
                "name",
                "Volatility Strategy"
            ),
            xaxis_title="Date",
            yaxis_title="Volatility / Signal",
            legend_title="Series",
            hovermode="x unified"
        )

        return fig# ====================================================
# Correlation Chart
# ====================================================
import numpy as np
import pandas as pd
import plotly.graph_objects as go


class CorrelationStrategyChart(StrategyChart):

    STATIC_MAP = {
        "metrics": [
            "lookback",
            "corr_window",
            "signal_threshold",
            "mean_corr",
            "std_corr",
            "latest_corr"
        ],
        "chart": [
            "avg_correlation",
            "rolling_mean_corr",
            "correlation_volatility",
            "spread",
            "zscore",
            "momentum",
            "regime_series"
        ]
    }

    # ==================================================
    #  CORE FIX: CLEAN TIME INDEX (1970 REMOVAL)
    # ==================================================
    @staticmethod
    def _clean(series):

        if series is None:
            return None

        try:
            s = pd.Series(series).copy()
        except:
            return None

        s = pd.to_numeric(s, errors="coerce").dropna()

        # FORCE DATETIME INDEX
        s.index = pd.to_datetime(s.index, errors="coerce")

        # REMOVE BAD DATES (1970 / NaT)
        s = s[s.index.notna()]
        s = s[s.index > pd.Timestamp("2000-01-01")]

        # SORT + CLEAN
        s = s.sort_index()
        s = s[~s.index.duplicated(keep="last")]

        if len(s) < 2:
            return None

        return s

    # ==================================================
    # LINE PLOT HELPER
    # ==================================================
    @staticmethod
    def _plot_series(fig, series, name, color=None, dashed=False, opacity=1.0):

        s = CorrelationStrategyChart._clean(series)
        if s is None:
            return False

        fig.add_trace(
            go.Scatter(
                x=s.index,
                y=s.values,
                mode="lines",
                name=name,
                opacity=opacity,
                line=dict(
                    color=color,
                    dash="dash" if dashed else None
                ) if color else dict(
                    dash="dash" if dashed else None
                )
            )
        )

        return True

    # ==================================================
    # STEP REGIME HELPER
    # ==================================================
    @staticmethod
    def _plot_regime(fig, series, name):

        s = CorrelationStrategyChart._clean(series)
        if s is None:
            return False

        fig.add_trace(
            go.Scatter(
                x=s.index,
                y=s.values,
                mode="lines",
                line_shape="hv",
                name=name,
                opacity=0.6
            )
        )

        return True

    # ==================================================
    # MAIN RENDER
    # ==================================================
    @staticmethod
    def render(item):

        fig = go.Figure()

        signals = getattr(item, "signals", {}) or {}
        metrics = getattr(item, "metrics", {}) or {}

        avg_corr = signals.get("avg_correlation")
        rolling_mean = signals.get("rolling_mean_corr")
        corr_vol = signals.get("correlation_volatility")
        spread = signals.get("spread")
        zscore = signals.get("zscore")
        momentum = signals.get("momentum")
        regime_series = signals.get("regime_series")

        # ==================================================
        # CORRELATION LAYER
        # ==================================================
        CorrelationStrategyChart._plot_series(
            fig,
            avg_corr,
            "Average Correlation",
            color="cyan"
        )

        CorrelationStrategyChart._plot_series(
            fig,
            rolling_mean,
            "Rolling Mean Correlation",
            color="orange",
            dashed=True
        )

        # ==================================================
        # VOLATILITY
        # ==================================================
        CorrelationStrategyChart._plot_series(
            fig,
            corr_vol,
            "Correlation Volatility",
            color="yellow"
        )

        # ==================================================
        # SPREAD
        # ==================================================
        CorrelationStrategyChart._plot_series(
            fig,
            spread,
            "Correlation Spread",
            color="magenta"
        )

        # ==================================================
        # Z-SCORE
        # ==================================================
        CorrelationStrategyChart._plot_series(
            fig,
            zscore,
            "Correlation Z-Score",
            color="white"
        )

        # ==================================================
        # MOMENTUM
        # ==================================================
        CorrelationStrategyChart._plot_series(
            fig,
            momentum,
            "Correlation Momentum",
            color="green"
        )

        # ==================================================
        # REGIME (STEP)
        # ==================================================
        CorrelationStrategyChart._plot_regime(
            fig,
            regime_series,
            "Regime"
        )

        # ==================================================
        # THRESHOLD LINE
        # ==================================================
        threshold = metrics.get("signal_threshold")

        if threshold is not None:
            fig.add_hline(
                y=threshold,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Threshold {threshold}"
            )

        # ==================================================
        # MEAN LINE
        # ==================================================
        mean_corr = metrics.get("mean_corr")

        if mean_corr is not None:
            fig.add_hline(
                y=mean_corr,
                line_dash="dot",
                line_color="gray",
                annotation_text=f"Mean Corr {round(mean_corr, 3)}"
            )

        # ==================================================
        # REGIME LABEL
        # ==================================================
        regime = signals.get("regime")

        if regime:
            fig.add_annotation(
                text=f"Regime: {regime}",
                x=0.01,
                y=0.99,
                xref="paper",
                yref="paper",
                showarrow=False
            )

        # ==================================================
        # LAYOUT (UNCHANGED)
        # ==================================================
        fig.update_layout(
            template="plotly_dark",
            title=getattr(item, "name", "Correlation Strategy"),
            xaxis_title="Date",
            yaxis_title="Correlation",
            legend_title="Series",
            hovermode="x unified"
        )

        return fig

# ==========================================================
# Breadth Chart
# ==========================================================
import numpy as np
import pandas as pd
import plotly.graph_objects as go

class BreadthStrategyChart(StrategyChart):

    STATIC_MAP = {
        "metrics": [
            "lookback",
            "ma_short",
            "ma_long",
            "assets",
            "avg_breadth"
        ],
        "chart": [
            "breadth",
            "ma_short",
            "ma_long",
            "breadth_spread"
        ]
    }

    # ==================================================
    #  HARD CLEAN (CRITICAL FIX - SAME AS PAIRTRADING)
    # ==================================================
    @staticmethod
    def _clean(series):

        if series is None:
            return None

        try:
            s = pd.Series(series).dropna()
        except:
            return None

        # FORCE NUMERIC SAFETY
        s = pd.to_numeric(s, errors="coerce").dropna()

        # FORCE DATETIME INDEX
        s.index = pd.to_datetime(s.index, errors="coerce")

        # REMOVE BAD DATES (THIS KILLS 1970 ISSUE)
        s = s[s.index.notna()]
        s = s[s.index > pd.Timestamp("2000-01-01")]

        # SORT + DEDUPE
        s = s.sort_index()
        s = s[~s.index.duplicated(keep="last")]

        if len(s) < 5:
            return None

        return s

    # ==================================================
    # SERIES PLOT
    # ==================================================
    @staticmethod
    def _plot_series(fig, series, name, color=None):

        s = BreadthStrategyChart._clean(series)
        if s is None:
            return False

        fig.add_trace(
            go.Scatter(
                x=s.index,
                y=s.values,
                mode="lines",
                name=name,
                line=dict(color=color) if color else None
            )
        )
        return True

    # ==================================================
    # RESOLVE SERIES (SAFE ONLY)
    # ==================================================
    @staticmethod
    def _resolve_series(item, key):

        signals = getattr(item, "signals", None) or {}
        chart = getattr(item, "chart", None)

        chartdata = None

        if isinstance(chart, dict):
            chartdata = chart.get("chartdata", None)
        else:
            chartdata = getattr(chart, "chartdata", None)

        # priority 1: signals
        if isinstance(signals, dict) and key in signals:
            return signals.get(key)

        # priority 2: chartdata dict
        if isinstance(chartdata, dict):
            return chartdata.get(key)

        # priority 3: DataFrame
        if isinstance(chartdata, pd.DataFrame):
            if key in chartdata.columns:
                return chartdata[key]

        return None

    # ==================================================
    # MARKERS (SAFE)
    # ==================================================
    @staticmethod
    def _plot_markers(fig, series, label):

        s = BreadthStrategyChart._clean(series)
        if s is None:
            return False

        z = (s - s.mean()) / (s.std() + 1e-8)
        spikes = np.where(np.abs(z) > 2.0)[0]

        if len(spikes) == 0:
            return False

        fig.add_trace(
            go.Scatter(
                x=s.index[spikes],
                y=s.iloc[spikes],
                mode="markers",
                name=label,
                marker=dict(size=6)
            )
        )

        return True

    # ==================================================
    # RENDER
    # ==================================================
    @staticmethod
    def render(item):

        fig = go.Figure()

        series_keys = BreadthStrategyChart.STATIC_MAP.get("chart", [])

        color_map = {
            "breadth": "blue",
            "ma_short": "green",
            "ma_long": "red",
            "breadth_spread": "purple"
        }

        # ==================================================
        # PLOT SERIES (1970-PROOF)
        # ==================================================
        for key in series_keys:

            series = BreadthStrategyChart._resolve_series(item, key)

            BreadthStrategyChart._plot_series(
                fig,
                series,
                key,
                color_map.get(key)
            )

        # ==================================================
        # EXTREME ZONES (SAFE)
        # ==================================================
        breadth = BreadthStrategyChart._resolve_series(item, "breadth")
        breadth = BreadthStrategyChart._clean(breadth)

        if breadth is not None:

            spikes_high = np.where(breadth.values > 0.8)[0]
            spikes_low = np.where(breadth.values < 0.2)[0]

            spikes = np.unique(np.concatenate([spikes_high, spikes_low]))

            if len(spikes) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=breadth.index[spikes],
                        y=breadth.iloc[spikes],
                        mode="markers",
                        name="Extreme Breadth Zones",
                        marker=dict(size=6, color="orange")
                    )
                )

        # ==================================================
        # LAYOUT (NO RANGE GUESSING = NO 1970)
        # ==================================================
        fig.update_layout(
            template="plotly_dark",
            title=getattr(item, "name", "Market Breadth & Participation Strength"),
            xaxis_title="Date",
            yaxis_title="Breadth (0–1)",
            hovermode="x unified"
        )

        return fig

# ===========================================================
# Pair Trading Chart
# ===========================================================
import pandas as pd
import plotly.graph_objects as go

class PairTradingChart(StrategyChart):

    STATIC_MAP = {
        "metrics": [
            "pairs",
            "lookback",
            "entry_z",
            "exit_z"
        ],
        "chart": [
            "spread",
            "signal"
        ]
    }

    # ==================================================
    # SAFE X-AXIS RESOLVER (CORE FIX)
    # ==================================================
    @staticmethod
    def _resolve_xaxis(index, length):

        # Case 1: proper datetime index
        if isinstance(index, pd.DatetimeIndex):
            return index

        # Case 2: pandas datetime dtype
        if pd.api.types.is_datetime64_any_dtype(index):
            return index

        # Case 3: try coercion safely (ONLY if strings)
        if pd.api.types.is_object_dtype(index):
            try:
                x = pd.to_datetime(index, errors="raise")
                return x
            except Exception:
                pass

        # Case 4: fallback numeric (IMPORTANT FIX)
        return list(range(length))

    # ==================================================
    # LINE SERIES PLOT
    # ==================================================
    @staticmethod
    def _plot_series(fig, series, name, color=None):

        if series is None:
            return False

        try:
            series = pd.Series(series).dropna()
        except Exception:
            return False

        if series.empty:
            return False

        x = PairTradingChart._resolve_xaxis(series.index, len(series))
        y = series.values

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=name,
                line=dict(color=color) if color else None
            )
        )

        return True

    # ==================================================
    # STEP SIGNAL PLOT
    # ==================================================
    @staticmethod
    def _plot_signal(fig, series, name):

        if series is None:
            return False

        try:
            series = pd.Series(series).dropna()
        except Exception:
            return False

        if series.empty:
            return False

        x = PairTradingChart._resolve_xaxis(series.index, len(series))
        y = series.values

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line_shape="hv",
                name=name,
                opacity=0.6
            )
        )

        return True

    # ==================================================
    # MAIN RENDER
    # ==================================================
    @staticmethod
    def render(item):

        fig = go.Figure()

        signals = getattr(item, "signals", {}) or {}
        metrics = getattr(item, "metrics", {}) or {}

        spreads = signals.get("spread", {})
        trade_signals = signals.get("signal", {})

        # ==================================================
        # SPREAD + ZSCORE
        # ==================================================
        for pair, pair_data in spreads.items():

            if not isinstance(pair_data, dict):
                continue

            PairTradingChart._plot_series(
                fig,
                pair_data.get("spread"),
                f"{pair} Spread",
                "steelblue"
            )

            PairTradingChart._plot_series(
                fig,
                pair_data.get("zscore"),
                f"{pair} Z-Score",
                "orange"
            )

        # ==================================================
        # SIGNALS
        # ==================================================
        for pair, signal in trade_signals.items():

            PairTradingChart._plot_signal(
                fig,
                signal,
                f"{pair} Signal"
            )

        # ==================================================
        # THRESHOLDS
        # ==================================================
        entry_z = metrics.get("entry_z")
        exit_z = metrics.get("exit_z")

        if entry_z is not None:

            fig.add_hline(
                y=entry_z,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Entry +{entry_z}"
            )

            fig.add_hline(
                y=-entry_z,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Entry -{entry_z}"
            )

        if exit_z is not None:

            fig.add_hline(
                y=exit_z,
                line_dash="dot",
                line_color="green",
                annotation_text=f"Exit +{exit_z}"
            )

            fig.add_hline(
                y=-exit_z,
                line_dash="dot",
                line_color="green",
                annotation_text=f"Exit -{exit_z}"
            )

        # ==================================================
        # LAYOUT
        # ==================================================
        fig.update_layout(
            template="plotly_dark",
            title=getattr(item, "name", "Pairs Trading"),
            xaxis_title="Time",
            yaxis_title="Spread / Z-Score",
            legend_title="Series",
            hovermode="x unified"
        )

        return fig

# ==================================================
# Intraday Reversal chart
# ==================================================
import numpy as np
import pandas as pd
import plotly.graph_objects as go
class IntradayReversalChart(StrategyChart):

    STATIC_MAP = {
        "chart": [
            "z_vol",
            "z_volume",
            "volatility",
            "reversal_event_vol",
            "reversal_event_volume"
        ],
        "metrics": [
            "lookback",
            "volume_window",
            "threshold"
        ]
    }

    # =========================================================
    # HELPERS
    # =========================================================

    @staticmethod
    def get_attr_or_key(obj, name, default=None):

        if isinstance(obj, dict):
            return obj.get(name, default)

        return getattr(obj, name, default)

    @staticmethod
    def _resolve_xaxis(index, length):

        if isinstance(index, pd.DatetimeIndex):
            return index

        if pd.api.types.is_datetime64_any_dtype(index):
            return index

        if pd.api.types.is_object_dtype(index):
            try:
                return pd.to_datetime(index, errors="raise")
            except Exception:
                pass

        return list(range(length))

    @staticmethod
    def _safe_series(v, index=None):

        if isinstance(v, pd.Series):
            return v

        if isinstance(v, (list, tuple, np.ndarray)):

            idx = (
                index
                if index is not None
                else range(len(v))
            )

            return pd.Series(v, index=idx)

        if isinstance(
            v,
            (
                int,
                float,
                np.integer,
                np.floating
            )
        ):

            idx = index if index is not None else [0]

            return pd.Series(
                [v] * len(idx),
                index=idx
            )

        return None

    # =========================================================
    # GENERIC PLOTTERS
    # =========================================================

    @staticmethod
    def _plot_dict(fig, d, name):

        if not isinstance(d, dict):
            return False

        if len(d) == 0:
            return False

        fig.add_trace(
            go.Bar(
                x=list(d.keys()),
                y=list(d.values()),
                name=name
            )
        )

        return True

    @staticmethod
    def _plot_series(fig, v, name):

        if v is None:
            return False

        try:
            v = list(v)
        except Exception:
            return False

        fig.add_trace(
            go.Scatter(
                y=v,
                mode="lines",
                name=name
            )
        )

        return True

    @staticmethod
    def _plot_dataframe_series(
        fig,
        x,
        y,
        name,
        color=None,
        dash=None
    ):

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=name,
                line=dict(
                    color=color,
                    dash=dash
                ) if color or dash else None
            )
        )

    # =========================================================
    # RENDER
    # =========================================================

    @staticmethod
    def render(item):

        fig = go.Figure()

        chart = getattr(item, "chart", None)

        if chart is None:
            return fig

        # =====================================================
        # PASS 1
        # STATIC CHART ATTRIBUTES
        # =====================================================

        #for field in IntradayReversalChart.STATIC_MAP.get(
        #    "chart",
        #    []
        #):

        #    value = IntradayReversalChart.get_attr_or_key(
        #        chart,
        #        field
        #    )

        #    if value is None:
        #        continue

        #    if isinstance(value, dict):

        #        IntradayReversalChart._plot_dict(
        #            fig,
        #            value,
        #            field
        #        )

        #        continue

        #    if isinstance(
        #        value,
        #        (
        #            list,
        #            tuple,
        #            np.ndarray,
        #            pd.Series
        #        )
        #    ):

        #        IntradayReversalChart._plot_series(
        #            fig,
        #            value,
        #            field
        #        )

        # =====================================================
        # PASS 2
        # DATAFRAME SERIES
        # =====================================================

        df = getattr(chart, "chartdata", None)

        if isinstance(df, pd.DataFrame) and not df.empty:

            xbase = (
                IntradayReversalChart
                ._resolve_xaxis(
                    df.index,
                    len(df)
                )
            )

            # ---------------------------------------------
            # Normal indicator series
            # ---------------------------------------------

            series_map = {
                "z_vol": "Z Volatility Reversal",
                "z_volume": "Z Volume Reversal",
                "volatility": "Rolling Volatility"
            }

            for col, label in series_map.items():

                if col not in df.columns:
                    continue

                fig.add_trace(
                    go.Scatter(
                        x=xbase,
                        y=df[col],
                        mode="lines",
                        name=label
                    )
                )

            # ---------------------------------------------
            # Event series as visible legend traces
            # ---------------------------------------------

            event_map = {
                "reversal_event_vol":
                    (
                        "Volatility Reversal Events",
                        "red"
                    ),
                "reversal_event_volume":
                    (
                        "Volume Reversal Events",
                        "orange"
                    )
            }

            for event_col, (
                label,
                color
            ) in event_map.items():

                if event_col not in df.columns:
                    continue

                event_values = (
                    df[event_col]
                    .fillna(False)
                    .astype(int)
                )

                fig.add_trace(
                    go.Scatter(
                        x=xbase,
                        y=event_values,
                        mode="markers",
                        line=dict(
                            dash="dot",
                            color=color
                        ),
                        name=label
                    )
                )

            # ---------------------------------------------
            # Actual event markers
            # ---------------------------------------------

            marker_map = {
                "reversal_event_vol":
                    (
                        "z_vol",
                        "Volatility Event Marker",
                        "red"
                    ),
                "reversal_event_volume":
                    (
                        "z_volume",
                        "Volume Event Marker",
                        "orange"
                    )
            }

            for event_col, (
                target_col,
                marker_name,
                marker_color
            ) in marker_map.items():

                if event_col not in df.columns:
                    continue

                if target_col not in df.columns:
                    continue

                mask = (
                    df[event_col]
                    .fillna(False)
                    .astype(bool)
                )

                if mask.sum() == 0:
                    continue

                fig.add_trace(
                    go.Scatter(
                        x=df.index[mask],
                        y=df.loc[
                            mask,
                            target_col
                        ],
                        mode="markers",
                        marker=dict(
                            size=10,
                            color=marker_color,
                            symbol="diamond"
                        ),
                        name=marker_name
                    )
                )

            # ---------------------------------------------
            # Config-driven series (optional)
            # ---------------------------------------------

            plotted = {
                "z_vol",
                "z_volume",
                "volatility"
            }

            for s in getattr(chart, "series", []):

                source = s.get("source")

                if (
                    not source
                    or source not in df.columns
                    or source in plotted
                ):
                    continue

                name = s.get(
                    "name",
                    source
                )

                style = s.get(
                    "style",
                    "line"
                )

                dash = (
                    "dash"
                    if style == "dash"
                    else None
                )

                IntradayReversalChart._plot_dataframe_series(
                    fig,
                    xbase,
                    df[source],
                    name,
                    dash=dash
                )

        # =====================================================
        # PASS 3
        # SIGNALS
        # =====================================================

        signals = (
            getattr(item, "signals", {})
            or {}
        )

        if isinstance(signals, dict):

            for sym, sig in signals.items():

                sig = (
                    IntradayReversalChart
                    ._safe_series(sig)
                )

                if sig is None:
                    continue

                x = (
                    IntradayReversalChart
                    ._resolve_xaxis(
                        sig.index,
                        len(sig)
                    )
                )

                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=sig.values,
                        mode="markers",
                        line=dict(
                            color="green",
                            dash="dot"
                        ),
                        opacity=0.6,
                        name=f"{sym} Signal"
                    )
                )

        # =====================================================
        # LAYOUT
        # =====================================================

        fig.update_layout(

            template="plotly_dark",

            title=getattr(
                chart,
                "title",
                "Intraday Reversal"
            ),

            xaxis=dict(
                title="Time",
                automargin=True
            ),

            yaxis=dict(
                title="Value",
                automargin=True
            ),

            legend_title="Series / Signals",

            hovermode="x unified",

            showlegend=True,

            margin=dict(
                r=220
            )
        )

        return fig
# =======================================================
# STREV chart
# =======================================================
import numpy as np
import pandas as pd
import plotly.graph_objects as go

class STREVReversalChart(StrategyChart):

    STATIC_MAP = {
        "metrics": [
            "lookback",
            "zscore_window",
            "holding",
            "assets",
            "total_return"
        ],
        "chart": [
            "portfolio_curve",
            "signal_curve",
            "benchmark",
            "entry_exit_events"
        ]
    }

    # ==================================================
    # SERIES
    # ==================================================
    @staticmethod
    def _plot_series(fig, series, name):

        if series is None:
            return False

        try:
            series = pd.Series(series)
        except Exception:
            return False

        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                name=name
            )
        )

        return True

    # ==================================================
    # ENTRY / EXIT MARKERS
    # ==================================================
    @staticmethod
    def _plot_markers(fig, events, signal_curve):

        if events is None or signal_curve is None:
            return

        try:
            events = pd.Series(events)
            signal_curve = pd.Series(signal_curve)
        except Exception:
            return

        mask = events != 0

        if not mask.any():
            return

        fig.add_trace(
            go.Scatter(
                x=events.index[mask],
                y=signal_curve.reindex(events.index)[mask],
                mode="markers",
                name="Entry / Exit",
                marker=dict(size=8, symbol="circle")
            )
        )

    # ==================================================
    # MAIN RENDER (FIXED LEGEND + LAYOUT)
    # ==================================================
    @staticmethod
    def render(item):

        fig = go.Figure()

        chart = getattr(item, "chart", None)
        data = getattr(chart, "chartdata", None)

        if data is None:
            data = {}

        metrics = getattr(item, "metrics", None)
        if metrics is None:
            metrics = {}

        portfolio_curve = data.get("portfolio_curve")
        signal_curve = data.get("signal_curve")
        benchmark_curve = data.get("benchmark")
        entry_exit_events = data.get("entry_exit_events")

        # ==================================================
        # SERIES
        # ==================================================
        portfolio_label = (
            f" Portfolio "
            f"(S={portfolio_curve})"
        )

        benchmark_label = (
            f"Benchmark "
            f"(b={benchmark_curve})"
        )

        signalstrength_label = (
            f"Signal Strength "
            f"(s={signal_curve})"
        )

 
        STREVReversalChart._plot_series(fig, portfolio_curve, portfolio_label)
        STREVReversalChart._plot_series(fig, benchmark_curve, benchmark_label)
        STREVReversalChart._plot_series(fig, signal_curve, signalstrength_label)

        # ==================================================
        # MARKERS
        # ==================================================
        STREVReversalChart._plot_markers(fig, entry_exit_events, signal_curve)

        total_return = metrics.get("total_return", 0.0)

        # ==================================================
        #  FIXED LAYOUT (LEGEND RIGHT SIDE)
        # ==================================================
        fig.update_layout(
            template="plotly_dark",

            title=f"STREV Mean Reversion (Return={total_return:.2%})",

            xaxis_title="Date",
            yaxis_title="Value",
            legend_title="Series / Signals",
            hovermode="x unified",
            showlegend=True,
            margin=dict(r=220)
        )
        return fig


# ====================================================
# Time Series Momentum Chart
# ====================================================
import numpy as np
import pandas as pd
import plotly.graph_objects as go

class TimeSeriesMomentumChart(StrategyChart):

    STATIC_MAP = {
        "metrics": [
            "formation",
            "vol_target",
            "lookback_vol",
            "assets"
        ],
        "chart": [
            "signal",
            "benchmark"
        ]
    }

    # ==================================================
    # LINE PLOT HELPER (same pattern as UMD)
    # ==================================================
    @staticmethod
    def _plot_series(fig, series, name):

        if series is None:
            return False

        try:
            series = pd.Series(series)
        except Exception:
            return False

        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                name=name
            )
        )
        return True

    # ==================================================
    # MARKERS (REGIME SWITCHES / SIGNAL CHANGES)
    # ==================================================
    @staticmethod
    def _plot_markers(fig, series, label):

        if series is None:
            return False

        try:
            series = pd.Series(series)
        except Exception:
            return False

        # detect regime flips or spikes
        spikes = np.where(np.abs(series.values) > series.std() * 2)[0]

        if len(spikes) == 0:
            return False

        fig.add_trace(
            go.Scatter(
                x=series.index[spikes],
                y=series.iloc[spikes],
                mode="markers",
                name=label,
                marker=dict(size=6)
            )
        )

        return True

    # ==================================================
    # MAIN RENDER
    # ==================================================
    @staticmethod
    def render(item):

        fig = go.Figure()
        chart = getattr(item, "chart", None)
        data = getattr(chart, "chartdata", None) 
        if data is None:
           data = pd.DataFrame()
        metrics = getattr(item, "metrics", None) 
        if metrics is None:
           metrics = pd.DataFrame()


        # ==================================================
        # CORE SERIES (FROM STRATEGYRESULT)
        # ==================================================
        signal = data.get("signal")
        benchmark = data.get("benchmark")
        # ==================================================
        # LEGEND LABELS
        # ==================================================
        signal_label = (
            f"Trend Signal "
            f"(S={signal})"
        )

        benchmark_label = (
            f"Benchmark "
            f"SPY"
        )

        TimeSeriesMomentumChart._plot_series(fig, signal, signal_label)
        TimeSeriesMomentumChart._plot_series(fig, benchmark, benchmark_label)

        # ==================================================
        # OPTIONAL MARKERS
        # ==================================================
        regime = data.get("regime_switches")
        TimeSeriesMomentumChart._plot_markers(fig, regime, "Regime Switch")

        # ==================================================
        # LAYOUT
        # ==================================================
        fig.update_layout(
            template="plotly_dark",
            title=getattr(item, "name", "Time Series Momentum"),
            xaxis_title="Date",
            yaxis_title="Signal Strength",
            legend_title="Series / Signals",
            hovermode="x unified",
            showlegend=True,
            margin=dict(r=220)
        )

        return fig


#================================================
# UMDMomentumChart
#================================================
import numpy as np
import pandas as pd
import plotly.graph_objects as go


class UMDMomentumChart(StrategyChart):

    STATIC_MAP = {
        "metrics": [
            "formation",
            "skip_month",
            "holding",
            "top_quantile",
            "bottom_quantile",
            "assets",
            "total_return"
        ],
        "chart": [
            "portfolio",
            "benchmark"
        ]
    }

    # ==================================================
    # HELPERS
    # ==================================================
    @staticmethod
    def _plot_series(fig, series, name):

        if series is None:
            return False

        try:
            series = pd.Series(series)
        except Exception:
            return False

        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                name=name
            )
        )
        return True
    
    # ==================================================
    # MARKERS (optional rebalance / signals)
    # ==================================================
    @staticmethod
    def _plot_markers(fig, series, label):

      if series is None:
         return False

         # ==================================================
         # CASE 1: dict of series (your UMD case)
         # ==================================================
         if isinstance(series, dict):

            for k, v in series.items():

                try:
                   s = pd.Series(v)
                except Exception:
                   continue

                # safety: avoid object comparisons
                mask = (s.values != 0)

                if not mask.any():
                   continue

                fig.add_trace(
                  go.Scatter(
                    x=s.index[mask] if hasattr(s.index, "__len__") else list(range(len(s)))[mask],
                    y=s.values[mask],
                    mode="markers",
                    name=f"{label}-{k}",
                    marker=dict(size=5)
                )
            )
            return True

         # ==================================================
         # CASE 2: single series
         # ==================================================
         try:
            s = pd.Series(series)
         except Exception:
            return False

         values = s.values
         mask = values != 0

         if not mask.any():
            return False

         fig.add_trace(
            go.Scatter(
               x=s.index[mask],
               y=values[mask],
               mode="markers",
               name=label,
               marker=dict(size=6)
            )
         )

         return True

    # ==================================================
    # MAIN RENDER
    # ==================================================
    @staticmethod
    def render(item):

        fig = go.Figure()

        chart = getattr(item, "chart", None)

        data = getattr(chart, "chartdata", None)

        if data is None:
           data = pd.DataFrame()
        metrics = getattr(item, "metrics", None) 
        if metrics is None:
           metrics = pd.DataFrame()


        # ==================================================
        # CORE SERIES (FROM STRATEGYRESULT)
        # ==================================================
        portfolio = data.get("portfolio")
        benchmark = data.get("benchmark")
        portfolio_label = (
            f"UMD Portfolio "
            f"(S={portfolio})"
        )

        benchmark_label = (
            f"Benchmark "
            f"SPY"
        )
        UMDMomentumChart._plot_series(fig, portfolio, portfolio_label)
        UMDMomentumChart._plot_series(fig, benchmark, benchmark_label)

        # ==================================================
        # OPTIONAL SIGNAL MARKERS
        # ==================================================
        signals = getattr(item, "signals", None)
        UMDMomentumChart._plot_markers(fig, signals, "Rebalance Signals")

        # ==================================================
        # LAYOUT
        # ==================================================
        fig.update_layout(
            template="plotly_dark",
            title=getattr(item, "name", "UMD Momentum Strategy"),
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            legend_title="Series / Signals",
            hovermode="x unified",
            showlegend=True,
            margin=dict(r=220)
        )

        return fig

#================================================
# BETANEUTRALSTRATEGY
#================================================
import numpy as np
import pandas as pd
import plotly.graph_objects as go


class BetaNeutralStrategyChart(StrategyChart):

    STATIC_MAP = {
        "metrics": ["beta_stats", "assets", "beta_window"],
        "chart": ["bn_returns"]
    }

    # ==================================================
    # CLEAN (NO RECREATION OF TIME)
    # ==================================================
    @staticmethod
    def _clean(series):

        if series is None:
            return None

        try:
            s = pd.Series(series)
        except:
            return None

        s = s.dropna()

        #  ONLY CLEAN INDEX, DO NOT REBUILD TIME
        s.index = pd.to_datetime(s.index, errors="coerce")
        s = s[s.index.notna()]
        s = s.sort_index()

        if len(s) < 5:
            return None

        return s

    # ==================================================
    @staticmethod
    def _plot(fig, series, name):

        s = BetaNeutralStrategyChart._clean(series)
        if s is None:
            return False

        fig.add_trace(
            go.Scatter(
                x=s.index,
                y=s.values,
                mode="lines",
                name=name
            )
        )
        return True

    # ==================================================
    @staticmethod
    def _plot_markers(fig, series, label):

        s = BetaNeutralStrategyChart._clean(series)
        if s is None:
            return

        z = (s - s.mean()) / (s.std() + 1e-8)
        spikes = np.where(np.abs(z) > 2.0)[0]

        if len(spikes) == 0:
            return

        fig.add_trace(
            go.Scatter(
                x=s.index[spikes],
                y=s.iloc[spikes],
                mode="markers",
                name=label,
                marker=dict(size=6)
            )
        )

    # ==================================================
    @staticmethod
    def render(item):

        fig = go.Figure()

        data = getattr(item, "data", {}) or {}
        metrics = getattr(item, "metrics", {}) or {}

        pnl = data.get("pnl")

        benchmark_key = next((k for k in data.keys() if k != "pnl"), None)
        benchmark = data.get(benchmark_key) if benchmark_key else None

        # ==================================================
        # PLOT (NO 1970 POSSIBLE)
        # ==================================================
        BetaNeutralStrategyChart._plot(fig, pnl, "Beta-Neutral PnL")

        if benchmark is not None:
            BetaNeutralStrategyChart._plot(fig, benchmark, benchmark_key)

        # ==================================================
        # SIGNALS
        # ==================================================
        signals = getattr(item, "signals", {}) or {}

        for sym, sig in signals.items():
            BetaNeutralStrategyChart._plot_markers(fig, sig, f"{sym} spikes")

        # ==================================================
        # FINAL LAYOUT (NO TIME GUESSING)
        # ==================================================
        fig.update_layout(
            template="plotly_dark",
            title="Beta Neutral Strategy (4Y Performance)",
            xaxis_title="Time",
            yaxis_title="Normalized Value",
            hovermode="x unified"
        )

        return fig
# ==================================================
# DASHBOARD
# ==================================================
class QXDashboardXXX:
    _instance = None

    STRATEGY_REGISTRY = {
        "AlphaStrategy": AlphaStrategyChart,
        "BetaNeutralStrategy": BetaNeutralStrategyChart,
        "UMDMomentum":UMDMomentumChart,
        "TimeSeriesMomentum":TimeSeriesMomentumChart,
        "STREV":STREVReversalChart,
        "IntradayReversal":IntradayReversalChart,
        "PairTrading": PairTradingChart,
        "BreadthStrategy": BreadthStrategyChart,
        "CorrelationStrategy" : CorrelationStrategyChart,
        "VolatilityStrategy": VolatilityStrategyChart,
        "DispersionStrategy": DispersionStrategyChart,
        "ForecastStrategy" : ForecastStrategyChart,
        "IndustryMomentumStrategy" : IndustryMomentumChart,
        "PairTradingFallback" : PairTradingFallbackChart,
        "CorrelationFallback" : CorrelationFallbackChart,
    }

    def __init__(self):
        self.strats = []
        self.moms = []
        self.revs = []
        self.inst = []

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = QXDashboard()
        return cls._instance

    def addData(self, tab, obj):
        target = {
            "strategies": self.strats,
            "momentum": self.moms,
            "reversal": self.revs,
            "institution": self.inst,
        }.get(tab, self.strats)

        target.append(obj)
        print("[DASHBOARD] ", len(self.strats),
           " moms ", len(self.moms),
           " revs ", len(self.revs),
           " inst ", len(self.inst))
    def build_tabs(self):
        return {
            "strategies": self.strats,
            "momentum": self.moms,
            "reversal": self.revs,
            "institution": self.inst,
        }

    # ==================================================
    # MAIN RENDER (FIXED)
    # ==================================================
    def render(self):

        tabs = self.build_tabs()
        figures = {}

        for tab_name, items in tabs.items():

            tab_figs = []

            for item in items:

                # -----------------------------
                # FIX 3 GOES HERE
                # normalize metrics BEFORE chart
                # -----------------------------
                if hasattr(item, "metrics") and hasattr(item.metrics, "__dict__"):
                   item.metrics = vars(item.metrics)

                name = getattr(item, "name", None)
                cls = self.STRATEGY_REGISTRY.get(name)
                print("[DASHBOARD] processing ", name, cls)
                if cls is None:
                    continue
                
                # IMPORTANT FIX: no constructor binding
                fig = cls.render(item)

                tab_figs.append(fig)

            figures[tab_name] = tab_figs
        #for k, v in figures.items():
        #    print(k, len(v))
        return figures


#===Dashboard =============
import webbrowser
import tempfile
import plotly.io as pio


class QXDashboard1:
    _instance = None

    STRATEGY_REGISTRY = {
        "AlphaStrategy": AlphaStrategyChart,
        "BetaNeutralStrategy": BetaNeutralStrategyChart,
        "UMDMomentum": UMDMomentumChart,
        "TimeSeriesMomentum": TimeSeriesMomentumChart,
        "STREV": STREVReversalChart,
        "IntradayReversal": IntradayReversalChart,
        "PairTrading": PairTradingChart,
        "BreadthStrategy": BreadthStrategyChart,
        "CorrelationStrategy": CorrelationStrategyChart,
        "VolatilityStrategy": VolatilityStrategyChart,
        "DispersionStrategy": DispersionStrategyChart,
        "ForecastStrategy": ForecastStrategyChart,
        "IndustryMomentumStrategy": IndustryMomentumChart,
        "PairTradingFallback": PairTradingFallbackChart,
        "CorrelationFallback": CorrelationFallbackChart,
    }

    def __init__(self):
        self.strats = []
        self.moms = []
        self.revs = []
        self.inst = []

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = QXDashboard()
        return cls._instance

    # -----------------------------
    # DATA INGESTION
    # -----------------------------
    def addData(self, tab, obj):

        target = {
            "strategies": self.strats,
            "momentum": self.moms,
            "reversal": self.revs,
            "institution": self.inst,
        }.get(tab, self.strats)

        target.append(obj)

    def build_tabs(self):
        return {
            "strategies": self.strats,
            "momentum": self.moms,
            "reversal": self.revs,
            "institution": self.inst,
        }

    # ==================================================
    # MAIN RENDER (REGISTRY-DRIVEN PER ITEM)
    # ==================================================
    def render(self):

        tabs = self.build_tabs()
        rendered_tabs = {}

        for tab_name, items in tabs.items():

            tab_html_parts = []

            for item in items:

                name = getattr(item, "name", None)
                cls = self.STRATEGY_REGISTRY.get(name)

                if cls is None:
                    continue

                try:
                    # IMPORTANT: registry-driven render per item
                    fig = cls.render(item)

                    chart_html = pio.to_html(
                        fig,
                        full_html=False,
                        include_plotlyjs=False
                    )

                    tab_html_parts.append(chart_html)

                except Exception as e:
                    tab_html_parts.append(f"<pre>Render error: {name} - {e}</pre>")

            rendered_tabs[tab_name] = tab_html_parts

        return rendered_tabs

    # ==================================================
    # SINGLE WINDOW DASHBOARD
    # ==================================================
    def display(self):

        tabs = self.render()

        tab_buttons = []
        tab_contents = []

        first = True

        for tab_name, charts in tabs.items():

            active = "active" if first else ""
            display = "block" if first else "none"
            first = False

            # -----------------------
            # TAB BUTTON
            # -----------------------
            tab_buttons.append(f"""
                <button class="tablink {active}" onclick="openTab('{tab_name}')">
                    {tab_name.upper()}
                </button>
            """)

            # -----------------------
            # TAB CONTENT
            # -----------------------
            tab_contents.append(f"""
                <div id="{tab_name}" class="tabcontent" style="display:{display}">
                    {"".join(charts)}
                </div>
            """)

        html = f"""
        <html>
        <head>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>

            <style>
                body {{
                    margin: 0;
                    background: #111;
                    color: white;
                    font-family: Arial;
                }}

                .tab {{
                    display: flex;
                    background: #222;
                }}

                .tab button {{
                    flex: 1;
                    padding: 12px;
                    background: #333;
                    border: none;
                    color: white;
                    cursor: pointer;
                }}

                .tab button.active {{
                    background: #555;
                }}

                .tabcontent {{
                    padding: 10px;
                }}
            </style>
        </head>

        <body>

        <div class="tab">
            {"".join(tab_buttons)}
        </div>

        {"".join(tab_contents)}

        <script>
            function openTab(tabName) {{

                let contents = document.getElementsByClassName("tabcontent");
                for (let i = 0; i < contents.length; i++) {{
                    contents[i].style.display = "none";
                }}

                let buttons = document.getElementsByClassName("tablink");
                for (let i = 0; i < buttons.length; i++) {{
                    buttons[i].classList.remove("active");
                }}

                document.getElementById(tabName).style.display = "block";
                event.currentTarget.classList.add("active");
            }}
        </script>

        </body>
        </html>
        """

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
        tmp.write(html.encode("utf-8"))
        tmp.close()

        webbrowser.open("file://" + tmp.name)

        return tmp.name


#====================================================
# Dashboard on browser
#====================================================
import copy
import json
import uuid
import webbrowser
import tempfile


class QXDashboard:
    _instance = None

    STRATEGY_REGISTRY = {
        "IntradayStrategy":IntradayStrategyChart,
        "AlphaStrategy": AlphaStrategyChart,
        "BetaNeutralStrategy": BetaNeutralStrategyChart,
        "PairTrading": PairTradingChart,
        "BreadthStrategy": BreadthStrategyChart,
        "CorrelationStrategy": CorrelationStrategyChart,
        "VolatilityStrategy": VolatilityStrategyChart,
        "DispersionStrategy": DispersionStrategyChart,
        "ForecastStrategy": ForecastStrategyChart,
        "UMDMomentum": UMDMomentumChart,
        "TimeSeriesMomentum": TimeSeriesMomentumChart,        
        "IndustryMomentumStrategy": IndustryMomentumChart,
        "STREV": STREVReversalChart,
        "IntradayReversal": IntradayReversalChart,
        "PairTradingFallback": PairTradingFallbackChart,
        "CorrelationFallback": CorrelationFallbackChart,
    }

    def __init__(self):
        self.strats = []
        self.moms = []
        self.revs = []
        self.inst = []

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = QXDashboard()
        return cls._instance

    # ==================================================
    # SAFE INGESTION
    # ==================================================
    def addData(self, tab, obj):
        target = {
            "strategies": self.strats,
            "momentum": self.moms,
            "reversal": self.revs,
            "institution": self.inst,
        }.get(tab, self.strats)

        target.append(copy.deepcopy(obj))

    # ==================================================
    # TAB STRUCTURE
    # ==================================================
    def build_tabs(self):
        return {
            "strategies": self.strats,
            "momentum": self.moms,
            "reversal": self.revs,
            "institution": self.inst,
        }

    # ==================================================
    # STANDARD RENDER (UNCHANGED MODEL)
    # ==================================================
    def render(self):
        tabs = self.build_tabs()
        rendered = {}

        for tab_name, items in tabs.items():
            print("[DASHBOARD] render tab " , tab_name)
            tab_figs = []

            for item in items:

                name = getattr(item, "name", None)
                cls = self.STRATEGY_REGISTRY.get(name)

                if cls is None:
                    continue

                fig = cls.render(copy.deepcopy(item))
                tab_figs.append(fig)
                print("[DASHBOARD] render " , name, " completed ")
            rendered[tab_name] = tab_figs
            
        return rendered

    # ==================================================
    # NEW: FIGURE → JSON SERIALIZER
    # ==================================================
    def renderJSON(self):

        tabs = self.render()
        json_tabs = {}

        for tab_name, fig_list in tabs.items():

            json_list = []

            for fig in fig_list:

                fig_copy = copy.deepcopy(fig)

                json_list.append({
                    "id": f"{tab_name}_{uuid.uuid4().hex}",
                    "fig": fig_copy.to_json()
                })

            json_tabs[tab_name] = json_list

        return json_tabs

    # ==================================================
    # 🚀 NEW SAFE DASHBOARD (LAZY RENDERING)
    # ==================================================
    def displayJSON(self):

        tabs = self.renderJSON()

        tab_buttons = []
        tab_contents = []

        first = True

        for tab_name, charts in tabs.items():

            active = "active" if first else ""
            display = "block" if first else "none"
            first = False

            # -------------------------
            # TAB BUTTON
            # -------------------------
            tab_buttons.append(f"""
                <button class="tablink {active}" onclick="openTab('{tab_name}')">
                    {tab_name.upper()}
                </button>
            """)

            # -------------------------
            # EMPTY CONTAINERS (IMPORTANT)
            # -------------------------
            containers = []

            for c in charts:
                containers.append(f"""
                    <div id="{c['id']}" style="height:400px;"></div>
                """)

            tab_contents.append(f"""
                <div id="{tab_name}" class="tabcontent" style="display:{display}">
                    {"".join(containers)}
                </div>

                <script type="application/json" id="data_{tab_name}">
                    {json.dumps(charts)}
                </script>
            """)

        # ==================================================
        # HTML + JS ENGINE
        # ==================================================
        html = f"""
        <html>
        <head>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>

            <style>
                body {{
                    margin: 0;
                    background: #111;
                    color: white;
                    font-family: Arial;
                }}

                .tab {{
                    display: flex;
                    background: #222;
                }}

                .tab button {{
                    flex: 1;
                    padding: 12px;
                    background: #333;
                    border: none;
                    color: white;
                    cursor: pointer;
                }}

                .tab button.active {{
                    background: #555;
                }}

                .tabcontent {{
                    padding: 10px;
                }}
            </style>
        </head>

        <body>

        <div class="tab">
            {"".join(tab_buttons)}
        </div>

        {"".join(tab_contents)}

        <script>

        function renderTab(tabName) {{

            const raw = document.getElementById("data_" + tabName).textContent;
            const charts = JSON.parse(raw);

            charts.forEach(c => {{

                const fig = JSON.parse(c.fig);

                Plotly.newPlot(
                    c.id,
                    fig.data,
                    fig.layout,
                    {{responsive: true}}
                );

            }});
        }}

        function openTab(tabName) {{

            let contents = document.getElementsByClassName("tabcontent");

            for (let i = 0; i < contents.length; i++) {{
                contents[i].style.display = "none";
            }}

            let buttons = document.getElementsByClassName("tablink");

            for (let i = 0; i < buttons.length; i++) {{
                buttons[i].classList.remove("active");
            }}

            document.getElementById(tabName).style.display = "block";
            event.currentTarget.classList.add("active");

            // 🚀 CRITICAL FIX: render only visible tab
            setTimeout(() => {{
                renderTab(tabName);
            }}, 50);
        }}

        // auto-render first tab
        setTimeout(() => {{
            const firstTab = document.getElementsByClassName("tabcontent")[0];
            if (firstTab) {{
                const id = firstTab.id;
                renderTab(id);
            }}
        }}, 200);

        </script>

        </body>
        </html>
        """

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
        tmp.write(html.encode("utf-8"))
        tmp.close()

        webbrowser.open("file://" + tmp.name)

        return tmp.name

    # ==================================================
    #  FRAME-BASED DASHBOARD (SAFE FIG.SHOW EQUIVALENT)
    # ==================================================
    def displayFrame(self):

        tabs = self.render()

        tab_buttons = []
        tab_contents = []

        first = True
        print("[DASHBOARD] Total tabs : ", len(tabs))
        for tab_name, fig_list in tabs.items():

            active = "active" if first else ""
            display = "block" if first else "none"
            first = False

            # -----------------------------
            # TAB BUTTON
            # -----------------------------
            tab_buttons.append(f"""
                <button class="tablink {active}" onclick="openTab('{tab_name}')">
                    {tab_name.upper()}
                </button>
            """)

            iframe_blocks = []

            # ==================================================
            # EACH FIGURE → FIG.SHOW EQUIVALENT HTML
            # ==================================================
            for i, fig in enumerate(fig_list):

                safe_fig = copy.deepcopy(fig)
                print("[DASHBOARD] Loading " ,  safe_fig.layout.title.text ) 
                tmp_file = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".html"
                )

                # KEY: this is EXACT fig.show equivalent output
                safe_fig.write_html(
                    tmp_file.name,
                    include_plotlyjs="cdn",
                    full_html=True,
                    config={"responsive": True}
                )

                iframe_blocks.append(f"""
                    <iframe
                        src="file://{tmp_file.name}"
                        style="
                            width:100%;
                            height:480px;
                            border:none;
                            margin-bottom:12px;
                            border-radius:8px;
                            background:#111;
                        "
                    ></iframe>
                """)

            tab_contents.append(f"""
                <div id="{tab_name}" class="tabcontent" style="display:{display}">
                    {"".join(iframe_blocks)}
                </div>
            """)

        # ==================================================
        # DASHBOARD SHELL
        # ==================================================
        html = f"""
        <html>
        <head>
            <style>
                body {{
                    margin:0;
                    background:#111;
                    color:white;
                    font-family:Arial;
                }}

                .tab {{
                    display:flex;
                    background:#222;
                }}

                .tab button {{
                    flex:1;
                    padding:12px;
                    border:none;
                    background:#333;
                    color:white;
                    cursor:pointer;
                }}

                .tab button.active {{
                    background:#555;
                }}

                .tabcontent {{
                    padding:10px;
                }}
            </style>
            <meta charset="utf-8">
            <title>QuantX Dashboard</title>
        </head>

        <body>

        <div class="tab">
            {"".join(tab_buttons)}
        </div>

        {"".join(tab_contents)}

        <script>
            function openTab(tabName) {{

                let tabs = document.getElementsByClassName("tabcontent");

                for (let i = 0; i < tabs.length; i++) {{
                    tabs[i].style.display = "none";
                }}

                let buttons = document.getElementsByClassName("tablink");

                for (let i = 0; i < buttons.length; i++) {{
                    buttons[i].classList.remove("active");
                }}

                document.getElementById(tabName).style.display = "block";
                event.currentTarget.classList.add("active");
            }}
        </script>

        </body>
        </html>
        """

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
        tmp.write(html.encode("utf-8"))
        tmp.close()

        webbrowser.open("file://" + tmp.name)

        return tmp.name