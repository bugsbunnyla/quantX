import pandas as pd
import numpy as np

from .BaseStrategy import BaseStrategy
from ..StrategyResult import StrategyResult

#
# 1. CORE DATA FORMULA (Returns) Formula Rt​=​(Pt/Pt−1 )​−1 Purpose Convert prices → comparable normalized series output is pd.Dataframe of returns matrix
# 2. PAIRWISE CORRELATION FORMULA  Formula (Pearson correlation) ρX,Y​=​Cov(X,Y)/σX * ​σY  Rolling version (core of strategy) ρt=Corr(Rt−w:ti ,Rt−w:tj) 🧾 Output type pd.Series   # time series correlation
# 3. AVERAGE CORRELATION (MARKET BREADTH STYLE) Used when multiple assets exist.  Formula ρˉt	​  =1/N	​i=j∑	​ρi,j,t	​or simplified: ρ​t​=mean(corr_matrixt​)  pd.Series
# 4. ROLLING CORRELATION (CORE STRATEGY SIGNAL) Formula ρt	​=Corr(Xt−w:t	​,Yt−w:t	​) Interpretation  ising → assets moving together (risk-on/off regimes) falling → diversification regime.pd.Series
# 5.CORRELATION Z-SCORE (REGIME NORMALIZATION) Formula Zt	​=( ​ρt​−μw ) / σw ​where: μ = rolling mean σ = rolling std  Purpose Detect regime extremes pd.Series
# 6. REGIME SIGNAL FORMULA (VERY IMPORTANT) Rule-based model Regime={risk_on risk_off ρt >θ  ρt​≤θ  Output type dict or pd.Series  # 0/1 regime timeline Detect regime extremes
# 7. CORRELATION SPREAD FORMULA  Measures divergence between assets. Formula Spreadt	​=max(ρt​)−min(ρt)  Meaning high spread → unstable market structure low spread → synchronized market pd.Series
# 8. CORRELATION VOLATILITY  Formula σρ 	​=StdDev(ρ t−w:t 	​)  Purpose Measures instability of correlation regime pd.Series
# 9. HRESHOLD FORMULA (STATIC OR DYNAMIC)  Static θ=0.5 Dynamic (better) θt  =μρ +k⋅σρ float or pd.Series
# 10. CORRELATION MOMENTUM Formula Mt	​=ρt​−ρt−1	​ Meaning positive → strengthening correlation regime negative → decoupling pd.Series
import numpy as np
import pandas as pd

from .BaseStrategy import BaseStrategy
from ..StrategyResult import StrategyResult


class CorrelationStrategy(BaseStrategy):

    # ==================================================
    # MAIN STRATEGY
    # ==================================================
    def run(self):

        cfg = self.cfg

        lookback = cfg.get("lookback", 20)
        corr_window = cfg.get("corr_window", 20)
        signal_threshold = cfg.get("signal_threshold", 0.50)

        # ==================================================
        # PRICE MATRIX
        # ==================================================
        prices = pd.DataFrame({
            sym: df["close"]
            for sym, df in self.data.items()
            if isinstance(df, pd.DataFrame)
            and "close" in df.columns
        }).dropna()

        # ==================================================
        # VALIDATION
        # ==================================================
        if prices.empty or prices.shape[1] < 2:

            return StrategyResult(
                name="CorrelationStrategy",
                data=self.data,
                metrics={
                    "error": "insufficient assets"
                },
                signals={}
            )

        # ==================================================
        # RETURNS
        # ==================================================
        returns = prices.pct_change().dropna()

        if len(returns) < corr_window:

            return StrategyResult(
                name="CorrelationStrategy",
                data=self.data,
                metrics={
                    "error": "insufficient return history"
                },
                signals={}
            )

        # ==================================================
        # AVERAGE CORRELATION SERIES
        # ==================================================
        avg_corr_values = []
        dates = []

        for i in range(corr_window, len(returns)):

            window = returns.iloc[i - corr_window:i]

            corr_matrix = window.corr()

            values = corr_matrix.values

            # remove diagonal (self-correlation)
            mask = ~np.eye(values.shape[0], dtype=bool)

            avg_corr = np.nanmean(values[mask])

            avg_corr_values.append(avg_corr)
            dates.append(returns.index[i])

        avg_correlation = pd.Series(
            avg_corr_values,
            index=dates,
            name="avg_correlation"
        )

        # ==================================================
        # VALIDATION
        # ==================================================
        if avg_correlation.empty:

            return StrategyResult(
                name="CorrelationStrategy",
                data=self.data,
                metrics={
                    "error": "unable to calculate correlations"
                },
                signals={}
            )

        # ==================================================
        # ROLLING MEAN
        # ==================================================
        rolling_mean_corr = (
            avg_correlation
            .rolling(lookback, min_periods=1)
            .mean()
        )

        # ==================================================
        # CORRELATION VOLATILITY
        # ==================================================
        correlation_volatility = (
            avg_correlation
            .rolling(lookback, min_periods=1)
            .std()
            .fillna(0)
        )

        # ==================================================
        # SPREAD
        # ==================================================
        spread = avg_correlation - rolling_mean_corr

        # ==================================================
        # ZSCORE
        # ==================================================
        zscore = (
            avg_correlation - rolling_mean_corr
        ) / (correlation_volatility + 1e-8)

        zscore = (
            zscore
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )

        # ==================================================
        # MOMENTUM
        # ==================================================
        momentum = avg_correlation.diff().fillna(0)

        # ==================================================
        # REGIME DETECTION
        # ==================================================
        regime_series = (
            avg_correlation > signal_threshold
        ).astype(int)

        latest_corr = float(avg_correlation.iloc[-1])

        regime = (
            "risk_on"
            if latest_corr > signal_threshold
            else "risk_off"
        )

        # ==================================================
        # BUILD CHART (PAIRTRADING STYLE)
        # ==================================================
        chart = self.build_chart(
            series=self.cfg.get("chart", {}).get("series"),
            title=self.cfg.get("title"),
            charttype=self.cfg.get("chart", {}).get("type"),
            chartmode=self.cfg.get("chart", {}).get("mode"),
        )

        # ==================================================
        # METRICS
        # ==================================================
        metrics = {
            "lookback": lookback,
            "corr_window": corr_window,
            "signal_threshold": signal_threshold,

            "assets": prices.shape[1],

            "mean_corr": float(avg_correlation.mean()),
            "std_corr": float(avg_correlation.std()),
            "latest_corr": latest_corr,

            "latest_zscore": (
                float(zscore.iloc[-1])
                if not zscore.empty
                else 0.0
            ),

            "regime": regime
        }

        # ==================================================
        # SIGNALS
        # ==================================================
        signals = {
            "avg_correlation": avg_correlation,
            "rolling_mean_corr": rolling_mean_corr,
            "correlation_volatility": correlation_volatility,

            "spread": spread,
            "zscore": zscore,
            "momentum": momentum,

            "regime_series": regime_series,

            "latest_corr": latest_corr,
            "regime": regime
        }

        # ==================================================
        # FINAL OUTPUT
        # ==================================================
        return StrategyResult(
            name="CorrelationStrategy",
            data=self.data,
            metrics=metrics,
            signals=signals,
            chart=chart
        )