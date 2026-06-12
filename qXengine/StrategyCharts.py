
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

      print("[DEBUG] alpha:", type(alpha), alpha is not None)
      print("[DEBUG] beta:", type(beta), beta is not None)

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

#
# IntradayStrategyChart
#
import pandas as pd
import plotly.graph_objects as go


class IntradayStrategyChart(StrategyChart):

    # ==================================================
    # FIXED STATIC MAP (FULLY ALIGNED WITH STRATEGY)
    # ==================================================
    STATIC_MAP = {
        "metrics": [
            "lookback",
            "vol_window",
            "volume_window",
            "signal_threshold",
            "universe_size",
            "average_score"
        ],
        "chart": [
            "signal",
            "volume_stress",
            "dislocation_events",
            "score"
        ]
    }

    # ==================================================
    # LINE PLOT HELPER
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
                line=dict(color=color) if color else None
            )
        )

        return True

    # ==================================================
    # STEP SIGNAL HELPER (DISCRETE SERIES)
    # ==================================================
    @staticmethod
    def _plot_signal(fig, series, name, color=None):

        if series is None:
            return False

        try:
            series = pd.Series(series).fillna(0)
        except Exception:
            return False

        if series.empty:
            return False

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

    # ==================================================
    # MARKER PLOT (DISLOCATION EVENTS)
    # ==================================================
    @staticmethod
    def _plot_markers(fig, series, name):

        if series is None:
            return False

        try:
            series = pd.Series(series).fillna(0)
        except Exception:
            return False

        if series.empty:
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
                marker=dict(
                    size=6,
                    color="red",
                    symbol="circle"
                ),
                opacity=0.8
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

        signal_map = signals.get("signal", {})
        volume_map = signals.get("volume_stress", {})
        event_map = signals.get("dislocation_events", {})
        score_map = signals.get("score", {})

        # ==================================================
        # SIGNAL + VOLUME STRESS
        # ==================================================
        for sym in signal_map.keys():

            IntradayStrategyChart._plot_series(
                fig,
                signal_map.get(sym),
                f"{sym} Signal",
                "steelblue"
            )

            IntradayStrategyChart._plot_series(
                fig,
                volume_map.get(sym),
                f"{sym} Volume Stress",
                "orange"
            )

            # optional score overlay (faint)
            IntradayStrategyChart._plot_series(
                fig,
                score_map.get(sym),
                f"{sym} Score",
                "gray"
            )

        # ==================================================
        # DISLOCATION EVENTS (MARKERS)
        # ==================================================
        for sym, events in event_map.items():

            IntradayStrategyChart._plot_markers(
                fig,
                events,
                f"{sym} Dislocations"
            )

        # ==================================================
        # ZERO LINE (REFERENCE)
        # ==================================================
        fig.add_hline(
            y=0,
            line_dash="solid",
            line_color="gray",
            opacity=0.4
        )

        # ==================================================
        # LAYOUT
        # ==================================================
        fig.update_layout(
            template="plotly_dark",
            title=getattr(item, "name", "Intraday Strategy"),
            xaxis_title="Time",
            yaxis_title="Normalized Signal",
            legend_title="Series",
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
        "chart": []  # chart already built by strategy via build_chart()
    }

    # ==================================================
    # HELPERS
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
        # 1. CORRELATION LINE
        # ==================================================
        CorrelationFallbackChart._plot_series(
            fig,
            correlation,
            "Rolling Correlation",
            "orange"
        )

        # ==================================================
        # 2. SIGNAL (REGIME ALLOCATION)
        # ==================================================
        CorrelationFallbackChart._plot_series(
            fig,
            signal,
            "Smoothed Regime Signal",
            "cyan"
        )

        # ==================================================
        # 3. DISPERSION (INSTABILITY)
        # ==================================================
        CorrelationFallbackChart._plot_series(
            fig,
            dispersion,
            "Market Dispersion",
            "red"
        )

        # ==================================================
        # 4. EQUITY CURVE (OPTIONAL CONTEXT)
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
        # 6. REGIME (STEP STYLE)
        # ==================================================
        CorrelationFallbackChart._plot_series(
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

    # ==================================================
    # FIXED STATIC MAP (matches strategy output exactly)
    # ==================================================
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
        "chart": []  # IMPORTANT: chart already built by strategy
    }

    # ==================================================
    # HELPERS
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
                line=dict(color=color) if color else None
            )
        )
        return True

    @staticmethod
    def _plot_step(fig, series, name):

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
        # 1. SPREAD + ZSCORE
        # ==================================================
        for pair, pair_data in spreads.items():

            if not isinstance(pair_data, dict):
                continue

            spread = pair_data.get("spread")
            zscore = pair_data.get("zscore")

            PairTradingFallbackChart._plot_series(
                fig,
                spread,
                f"{pair} Spread",
                "steelblue"
            )

            PairTradingFallbackChart._plot_series(
                fig,
                zscore,
                f"{pair} Z-Score",
                "orange"
            )

        # ==================================================
        # 2. SIGNALS
        # ==================================================
        for pair, signal in trade_signals.items():

            PairTradingFallbackChart._plot_step(
                fig,
                signal,
                f"{pair} Signal"
            )

        # ==================================================
        # 3. ENTRY / EXIT LEVELS
        # ==================================================
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
        # 4. LAYOUT (NO CHART OVERRIDE)
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
    # LINE PLOT HELPER
    # ==================================================
    @staticmethod
    def _plot_series(fig, series, name, color=None, dashed=False, opacity=1.0):

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
        # FORECAST VS ACTUAL (PER SYMBOL)
        # ==================================================
        for sym, forecast in forecast_data.items():

            actual = actual_data.get(sym)

            # ----------------------------
            # FORECAST LINE
            # ----------------------------
            ForecastStrategyChart._plot_series(
                fig,
                forecast,
                f"{sym} Forecast Return",
                color="orange"
            )

            # ----------------------------
            # ACTUAL LINE
            # ----------------------------
            ForecastStrategyChart._plot_series(
                fig,
                actual,
                f"{sym} Actual Return",
                color="steelblue"
            )

        # ==================================================
        # ZERO LINE (REFERENCE)
        # ==================================================
        fig.add_hline(
            y=0,
            line_dash="dot",
            line_color="gray"
        )

        # ==================================================
        # OPTIONAL PERFORMANCE ANNOTATION
        # ==================================================
        assets = metrics.get("assets")
        horizon = metrics.get("forecast_horizon")

        if assets is not None:

            fig.add_annotation(
                text=f"Assets: {assets} | Horizon: {horizon}",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.01,
                y=0.99,
                align="left",
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
            "stock_momentum_index",
            "industry_momentum_index",
            "rebalance_events"
        ],
        "chart": [
            "stock_momentum_index",
            "industry_momentum_index",
            "benchmark",
            "rebalance_events"
        ]
    }

    # ==================================================
    # SAFE SERIES PLOT
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
                line=dict(color=color) if color else None
            )
        )
        return True

    # ==================================================
    # STEP SIGNAL (rebalance events)
    # ==================================================
    @staticmethod
    def _plot_step(fig, series, name, color=None):

        if series is None:
            return False

        try:
            series = pd.Series(series).fillna(0)
        except Exception:
            return False

        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                mode="lines",
                line_shape="hv",
                name=name,
                opacity=0.5,
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

        # ==================================================
        # CORE SERIES
        # ==================================================
        stock_index = signals.get("stock_momentum_index")
        industry_index = signals.get("industry_momentum_index")
        benchmark = signals.get("benchmark")
        rebalance = signals.get("rebalance_events")

        # ==================================================
        # PLOT: STOCK MOMENTUM INDEX
        # ==================================================
        IndustryMomentumChart._plot_series(
            fig,
            stock_index,
            "Stock Momentum Index",
            "cyan"
        )

        # ==================================================
        # PLOT: INDUSTRY MOMENTUM INDEX
        # ==================================================
        IndustryMomentumChart._plot_series(
            fig,
            industry_index,
            "Industry Momentum Index",
            "orange"
        )

        # ==================================================
        # PLOT: BENCHMARK (SPY)
        # ==================================================
        IndustryMomentumChart._plot_series(
            fig,
            benchmark,
            "Benchmark (SPY)",
            "white"
        )

        # ==================================================
        # PLOT: REBALANCE EVENTS
        # ==================================================
        IndustryMomentumChart._plot_step(
            fig,
            rebalance,
            "Rebalance Events",
            "green"
        )

        # ==================================================
        # METRICS ANNOTATION
        # ==================================================
        best = metrics.get("best_industry")
        worst = metrics.get("worst_industry")

        if best or worst:
            fig.add_annotation(
                text=f"Best: {best} | Worst: {worst}",
                x=0.01,
                y=1.08,
                xref="paper",
                yref="paper",
                showarrow=False
            )

        # ==================================================
        # LAYOUT
        # ==================================================
        fig.update_layout(
            template="plotly_dark",
            title=getattr(item, "name", "Industry Momentum Strategy"),
            xaxis_title="Date",
            yaxis_title="Normalized Performance",
            legend_title="Series",
            hovermode="x unified"
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
            "assets"
        ],
        "chart": [
            "dispersion",
            "ma_63",
            "regime_switches"
        ]
    }

    # ==================================================
    # LINE PLOT HELPER
    # ==================================================
    @staticmethod
    def _plot_series(fig, series, name, color=None, dashed=False, opacity=1.0):

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
    # STEP SIGNAL HELPER (for regime switches)
    # ==================================================
    @staticmethod
    def _plot_step(fig, series, name, color=None):

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
                line_shape="hv",
                opacity=0.6,
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

        dispersion = signals.get("dispersion")
        ma_63 = signals.get("ma_63")
        regime_switches = signals.get("regime_switches")

        # ==================================================
        # DISPERSION (MAIN SIGNAL)
        # ==================================================
        DispersionStrategyChart._plot_series(
            fig,
            dispersion,
            "Cross-sectional Dispersion",
            color="steelblue"
        )

        # ==================================================
        # MOVING AVERAGE
        # ==================================================
        DispersionStrategyChart._plot_series(
            fig,
            ma_63,
            "MA (Lookback)",
            color="orange",
            dashed=True
        )

        # ==================================================
        # REGIME SWITCHES
        # ==================================================
        DispersionStrategyChart._plot_step(
            fig,
            regime_switches,
            "Regime Switches",
            color="green"
        )

        # ==================================================
        # OPTIONAL THRESHOLD LINE (MEAN)
        # ==================================================
        mean_disp = metrics.get("mean_dispersion")

        if mean_disp is not None:

            fig.add_hline(
                y=mean_disp,
                line_dash="dot",
                line_color="red",
                annotation_text=f"Mean Dispersion {round(mean_disp, 4)}"
            )

        # ==================================================
        # LAYOUT
        # ==================================================
        fig.update_layout(
            template="plotly_dark",
            title=getattr(item, "name", "Dispersion Strategy"),
            xaxis_title="Date",
            yaxis_title="Cross-sectional Dispersion",
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
            "vol_window",
            "target_vol"
        ],
        "chart": [
            "volatility",
            "target_vol",
            "vol_signal"
        ]
    }

    # ==================================================
    # LINE PLOT HELPER
    # ==================================================
    @staticmethod
    def _plot_series(fig, series, name, color=None, dashed=False):

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
                line=dict(
                    color=color,
                    dash="dash" if dashed else None
                ) if color else dict(dash="dash" if dashed else None)
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

        vol_series = signals.get("vol_series", {})
        vol_signals = signals.get("vol_signal", {})

        # ==================================================
        # VOLATILITY CURVES PER SYMBOL
        # ==================================================
        for sym, data in vol_series.items():

            if not isinstance(data, dict):
                continue

            volatility = data.get("volatility")
            target_vol = data.get("target_vol")

            VolatilityStrategyChart._plot_series(
                fig,
                volatility,
                f"{sym} Realized Volatility",
                color="steelblue"
            )

            VolatilityStrategyChart._plot_series(
                fig,
                target_vol,
                f"{sym} Target Volatility",
                color="orange",
                dashed=True
            )

        # ==================================================
        # VOLATILITY SIGNAL (POSITION SCALING)
        # ==================================================
        for sym, signal in vol_signals.items():

            VolatilityStrategyChart._plot_series(
                fig,
                signal,
                f"{sym} Vol Signal",
                color="green"
            )

        # ==================================================
        # TARGET LINE FROM METRICS (GLOBAL)
        # ==================================================
        target_vol = metrics.get("target_vol")

        if target_vol is not None:

            fig.add_hline(
                y=target_vol,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Target Vol {target_vol}"
            )

        # ==================================================
        # LAYOUT
        # ==================================================
        fig.update_layout(
            template="plotly_dark",
            title=getattr(item, "name", "Volatility Strategy"),
            xaxis_title="Date",
            yaxis_title="Volatility / Signal",
            legend_title="Series",
            hovermode="x unified"
        )

        return fig


# ====================================================
# Correlation Chart
# ====================================================
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
        # CORRELATION
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
        # REGIME
        # ==================================================
        CorrelationStrategyChart._plot_regime(
            fig,
            regime_series,
            "Regime"
        )

        # ==================================================
        # THRESHOLD
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
        # MEAN CORRELATION
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
        # REGIME ANNOTATION
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
        # LAYOUT
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
#
# Breadth Chart
#
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
    # SERIES PLOT
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
                line=dict(color=color) if color else None
            )
        )

        return True

    # ==================================================
    # RESOLVE SERIES (signals → fallback chartdata)
    # ==================================================
    @staticmethod
    def _resolve_series(item, key):

      signals = getattr(item, "signals", None) or {}

      chart = getattr(item, "chart", None)

    # ==================================================
    # SAFE NORMALIZATION OF chart
    # ==================================================
      if isinstance(chart, dict):
        chartdata = chart.get("chartdata", None)
      else:
        chartdata = getattr(chart, "chartdata", None)

    # fallback safety
      if chartdata is None:
        chartdata = {}

    # ==================================================
    # PRIORITY 1: signals
    # ==================================================
      if isinstance(signals, dict) and key in signals:
        return signals.get(key)

    # ==================================================
    # PRIORITY 2: chartdata (matrix)
    # ==================================================
      if isinstance(chartdata, dict):
        return chartdata.get(key)

      if isinstance(chartdata, pd.DataFrame):
        if key in chartdata.columns:
            return chartdata[key]

      return None

    @staticmethod
    def _plot_markers(fig, series, label):

        if series is None:
           return False

        try:
           series = pd.Series(series).dropna()
        except Exception:
           return False

    # ==================================================
    # FORCE NUMERIC CLEANUP (CRITICAL FIX)
    # ==================================================
        series = pd.to_numeric(series, errors="coerce").dropna()

        if series.empty:
            return False

        values = series.values.astype(float)

        std = values.std()
        if std == 0 or np.isnan(std):
            return False

        spikes = np.where(np.abs(values / (std + 1e-8)) > 2.0)[0]

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
    # RENDER
    # ==================================================
    @staticmethod
    def render(item):

        fig = go.Figure()

        # ==================================================
        # USE STATIC_MAP (THIS WAS MISSING BEFORE)
        # ==================================================
        series_keys = BreadthStrategyChart.STATIC_MAP.get("chart", [])

        color_map = {
            "breadth": "blue",
            "ma_short": "green",
            "ma_long": "red",
            "breadth_spread": "purple",
            "signals": "pink"
        }

        signals = getattr(item, "signals", None)
        BreadthStrategyChart._plot_markers(fig, signals, "Breadth Signals")     
        # ==================================================
        # PLOT ALL SERIES FROM CONTRACT
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
        # MARKERS (optional enrichment)
        # ==================================================
        breadth = BreadthStrategyChart._resolve_series(item, "breadth")

        if breadth is not None:
            series = pd.Series(breadth).dropna()

            spikes_high = np.where(series.values > 0.8)[0]
            spikes_low = np.where(series.values < 0.2)[0]

            spikes = np.unique(np.concatenate([spikes_high, spikes_low]))

            if len(spikes) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=series.index[spikes],
                        y=series.iloc[spikes],
                        mode="markers",
                        name="Extreme Breadth Zones",
                        marker=dict(size=6, color="orange")
                    )
                )

        # ==================================================
        # LAYOUT
        # ==================================================
        fig.update_layout(
            template="plotly_dark",
            title=getattr(item, "name", "Market Breadth & Participation Strength"),
            xaxis_title="Date",
            yaxis_title="Breadth (0–1)",
            legend_title="Series",
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
    # LINE PLOT HELPER
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
                line=dict(color=color) if color else None
            )
        )

        return True

    # ==================================================
    # STEP SIGNAL HELPER
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

        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
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
        # SPREADS / ZSCORES
        # ==================================================
        for pair, pair_data in spreads.items():

            if not isinstance(pair_data, dict):
                continue

            spread = pair_data.get("spread")
            zscore = pair_data.get("zscore")

            PairTradingChart._plot_series(
                fig,
                spread,
                f"{pair} Spread",
                "steelblue"
            )

            PairTradingChart._plot_series(
                fig,
                zscore,
                f"{pair} Z-Score",
                "orange"
            )

        # ==================================================
        # TRADE SIGNALS
        # ==================================================
        for pair, signal in trade_signals.items():

            PairTradingChart._plot_signal(
                fig,
                signal,
                f"{pair} Signal"
            )

        # ==================================================
        # ENTRY / EXIT THRESHOLDS
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
            xaxis_title="Date",
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

    # =========================================================
    # RENDER
    # =========================================================
    @staticmethod
    def render(item):

        fig = go.Figure()

        chart = item.chart
        df = chart.chartdata
        series_list = chart.series or []
        signals = item.signals or {}

        if df is None or not isinstance(df, pd.DataFrame):
            return fig

        # =========================================================
        # SERIES RENDERING (FROM CONFIG ONLY)
        # =========================================================
        for s in series_list:

            source = s.get("source")
            if source not in df.columns:
                continue

            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[source],
                    mode="lines",
                    name=s.get("name", source)
                )
            )

        # =========================================================
        # SIGNAL RENDERING
        # =========================================================
        for sym, sig in signals.items():

            fig.add_trace(
                go.Scatter(
                    x=sig.index,
                    y=sig.values,
                    mode="lines",
                    name=f"{sym} signal"
                )
            )

        # =========================================================
        # EVENT MARKERS
        # =========================================================
        if "reversal_event_vol" in df.columns:

            fig.add_trace(
                go.Scatter(
                    x=df.index[df["reversal_event_vol"]],
                    y=df["z_vol"][df["reversal_event_vol"]],
                    mode="markers",
                    name="vol events"
                )
            )

        if "reversal_event_volume" in df.columns:

            fig.add_trace(
                go.Scatter(
                    x=df.index[df["reversal_event_volume"]],
                    y=df["z_volume"][df["reversal_event_volume"]],
                    mode="markers",
                    name="volume events"
                )
            )

        # =========================================================
        # THEME
        # =========================================================
        fig.update_layout(
            template="plotly_dark",
            hovermode="x unified",
            legend=dict(orientation="h"),
            title=chart.title
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

        if events is None:
            return

        if signal_curve is None:
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

                marker=dict(
                    size=8,
                    symbol="circle"
                )
            )
        )

    # ==================================================
    # MAIN RENDER
    # ==================================================

    @staticmethod
    def render(item):

        fig = go.Figure()

        data = getattr(item, "data", {}) or {}
        metrics = getattr(item, "metrics", {}) or {}

        portfolio_curve = data.get("portfolio_curve")
        signal_curve = data.get("signal_curve")
        benchmark_curve = data.get("benchmark")
        entry_exit_events = data.get("entry_exit_events")

        #
        # Main Portfolio Curve
        #
        STREVReversalChart._plot_series(
            fig,
            portfolio_curve,
            "STREV Portfolio"
        )

        #
        # Benchmark
        #
        STREVReversalChart._plot_series(
            fig,
            benchmark_curve,
            "SPY Benchmark"
        )

        #
        # Signal Strength
        #
        STREVReversalChart._plot_series(
            fig,
            signal_curve,
            "Signal Strength"
        )

        #
        # Entry/Exit markers
        #
        STREVReversalChart._plot_markers(
            fig,
            entry_exit_events,
            signal_curve
        )

        total_return = metrics.get("total_return", 0.0)

        fig.update_layout(
            template="plotly_dark",

            title=(
                f"STREV Mean Reversion "
                f"(Return={total_return:.2%})"
            ),

            xaxis_title="Date",

            yaxis_title="Value",

            hovermode="x unified",

            legend=dict(
                orientation="h"
            )
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

        data = getattr(item, "data", {}) or {}
        metrics = getattr(item, "metrics", {}) or {}

        # ==================================================
        # CORE SERIES (FROM STRATEGYRESULT)
        # ==================================================
        signal = data.get("signal")
        benchmark = data.get("benchmark")

        TimeSeriesMomentumChart._plot_series(fig, signal, "Trend Signal")
        TimeSeriesMomentumChart._plot_series(fig, benchmark, "SPY Benchmark")

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
            yaxis_title="Signal Strength"
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

        data = getattr(item, "data", {}) or {}
        metrics = getattr(item, "metrics", {}) or {}

        # ==================================================
        # CORE SERIES (FROM STRATEGYRESULT)
        # ==================================================
        portfolio = data.get("portfolio")
        benchmark = data.get("benchmark")

        UMDMomentumChart._plot_series(fig, portfolio, "UMD Portfolio")
        UMDMomentumChart._plot_series(fig, benchmark, "SPY Benchmark")

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
            yaxis_title="Cumulative Return"
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
        "metrics": ["beta_stats", "assets","beta_window"],
        "chart": ["bn_returns"]
    }

    # ==================================================
    # HELPERS
    # ==================================================
    @staticmethod
    def _plot_series(fig, series, name):

        if series is None:
            return False

        try:
            if isinstance(series, dict):
                series = pd.Series(series)

            series = pd.Series(series)
        except Exception:
            return False

        fig.add_trace(
            go.Scatter(
                y=series.values,
                mode="lines",
                name=name
            )
        )
        return True

    @staticmethod
    def _plot_markers(fig, series, label):

        if series is None:
            return

        if isinstance(series, dict):
            series = pd.Series(series)

        series = pd.Series(series)

        spikes = np.where(np.abs(series / (series.std() + 1e-8)) > 2.0)[0]

        fig.add_trace(
            go.Scatter(
                x=spikes,
                y=series.iloc[spikes],
                mode="markers",
                name=label,
                marker=dict(size=6)
            )
        )

    # ==================================================
    # MAIN RENDER
    # ==================================================
    @staticmethod
    def render(item):

        fig = go.Figure()

        # -----------------------
        # DATA CONTRACT
        # -----------------------
        data = getattr(item, "data", {}) or {}
        metrics = getattr(item, "metrics", {}) or {}

        pnl = data.get("pnl")
        benchmark_key = None

        # extract benchmark dynamically (SPY or config-defined)
        for k in data.keys():
            if k != "pnl":
                benchmark_key = k

        benchmark = data.get(benchmark_key) if benchmark_key else None

        # -----------------------
        # SERIES (CONFIG DRIVEN)
        # -----------------------
        BetaNeutralStrategyChart._plot_series(fig, pnl, "Beta-Neutral PnL")

        if benchmark is not None:
            BetaNeutralStrategyChart._plot_series(fig, benchmark, benchmark_key)

        # -----------------------
        # METRICS-DRIVEN MARKERS
        # -----------------------
        beta_stats = metrics.get("beta_stats", {})

        if isinstance(beta_stats, dict):

            instability_events = []

            for asset, stats in beta_stats.items():

                beta_vol = stats.get("beta_vol", 0)

                if beta_vol > 0.3:
                    instability_events.append((asset, beta_vol))

            if instability_events:
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(instability_events))),
                        y=[v for _, v in instability_events],
                        mode="markers",
                        name="Beta Instability",
                        marker=dict(size=10, color="red")
                    )
                )

        # -----------------------
        # OPTIONAL: SIGNAL SPIKES (FROM STRATEGY OUTPUT)
        # -----------------------
        signals = getattr(item, "signals", {}) or {}

        if isinstance(signals, dict):
            for sym, sig in signals.items():
                BetaNeutralStrategyChart._plot_markers(fig, sig, f"{sym} spikes")

        # -----------------------
        # LAYOUT (CONFIG ALIGNED)
        # -----------------------
        fig.update_layout(
            template="plotly_dark",
            title="Beta Neutral Strategy (4Y Performance)",
            xaxis_title="Time",
            yaxis_title="Normalized Value",
            showlegend=True
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
        print(" strats ", len(self.strats),
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
                print(" processing ", name, cls)
                if cls is None:
                    continue
                
                # IMPORTANT FIX: no constructor binding
                fig = cls.render(item)

                tab_figs.append(fig)

            figures[tab_name] = tab_figs
        for k, v in figures.items():
            print(k, len(v))
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
                    # 🔥 IMPORTANT: registry-driven render per item
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
    # 🚀 FRAME-BASED DASHBOARD (SAFE FIG.SHOW EQUIVALENT)
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

                # 🔥 KEY: this is EXACT fig.show equivalent output
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