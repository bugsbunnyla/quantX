import pandas as pd
import numpy as np
from .BaseStrategy import BaseStrategy
from ..StrategyResult import StrategyResult
#
# Half-life comes from modeling the spread as an Ornstein–Uhlenbeck (OU) mean-reverting process: OU process: dSt​=θ(μ−St	​)dt+σdWt Where:St = spreadμ = long-run mean 
# θ = speed of mean reversion σ = noise Discrete approximation used in your code: You approximate:ΔSt	​=βSt−1	​+ϵt	​Where:β<0 indicates mean reversion 
# Half-life formula:t1/2	​=−ln(2)/β	b​ut since β<0, this becomes positive.
# Interpretation: Small half-life → fast mean reversion → good trading signal, Large half-life → slow drift → weak signal, Infinite → non mean-reverting → ignore pair
# Fallback -> Tier 1 (Best quality) ->  		Tier 2 (Fallback)   -> 			Tier 3 (Emergency fallback)
# 		Cointegration + OU modeling		correlation-based pairing		pure correlation / sector pairs
# 		strict statistical validation		simpler hedge ratio (OLS)		no statistical validation
# 		low false positives			optional filters (like half-life)	just to keep system alive
# 
class PairTradingFallback(BaseStrategy):

    # ==================================================
    # HALF-LIFE
    # ==================================================
    def compute_half_life(self, spread: pd.Series) -> float:

        try:

            spread = spread.dropna()

            if len(spread) < 20:
                return np.inf

            lagged = spread.shift(1).dropna()
            delta = spread.diff().dropna()

            if len(lagged) != len(delta):
                return np.inf

            beta = np.polyfit(
                lagged.values,
                delta.values,
                1
            )[0]

            if beta >= 0 or np.isnan(beta):
                return np.inf

            return -np.log(2) / beta

        except:
            return np.inf

    # ==================================================
    # MAIN STRATEGY
    # ==================================================
    def run(self):

        cfg = self.cfg

        lookback = cfg.get("lookback", 126)
        corr_window = cfg.get("corr_window", 63)
        min_corr = cfg.get("min_corr", 0.6)

        entry_z = cfg.get("entry_zscore", 1.5)
        exit_z = cfg.get("exit_zscore", 0.3)

        max_pairs = cfg.get("max_pairs", 5)
        min_half_life = cfg.get("min_half_life", 5)

        # ==================================================
        # CLEAN INPUT DATA (FIX FOR 1970 ISSUES)
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

                if not isinstance(tmp.index, pd.DatetimeIndex):

                    tmp.index = pd.to_datetime(
                        tmp.index,
                        errors="coerce"
                    )

                    tmp = tmp[tmp.index.notna()]

                tmp = tmp.sort_index()

            tmp = tmp[~tmp.index.duplicated(keep="last")]

            cleaned[sym] = tmp

        # ==================================================
        # PRICE MATRIX
        # ==================================================
        prices = pd.DataFrame({
            sym: df["close"]
            for sym, df in cleaned.items()
        })

        prices = prices.sort_index()
        prices = prices.ffill()
        prices = prices.dropna(how="all")

        if prices.empty or prices.shape[1] < 2:

            return StrategyResult(
                name="PairTradingFallback",
                data=self.data,
                metrics={"error": "insufficient data"},
                signals={}
            )

        # ==================================================
        # RETURNS
        # ==================================================
        rets = prices.pct_change().dropna()

        if len(rets) < corr_window:

            return StrategyResult(
                name="PairTradingFallback",
                data=self.data,
                metrics={"error": "insufficient history"},
                signals={}
            )

        corr = rets.tail(corr_window).corr().abs()

        pairs = (
            corr.unstack()
            .sort_values(ascending=False)
            .drop_duplicates()
        )

        pairs = pairs[pairs < 0.999]

        top_pairs = list(pairs.head(max_pairs).index)

        spreads_out = {}
        signals_out = {}

        # ==================================================
        # PAIR LOOP
        # ==================================================
        for sym_x, sym_y in top_pairs:

            if sym_x not in prices or sym_y not in prices:
                continue

            x = prices[sym_x]
            y = prices[sym_y]

            spread_series = []
            z_series = []
            signal_series = []
            time_index = []

            for i in range(lookback, len(prices)):

                x_win = x.iloc[i - lookback:i]
                y_win = y.iloc[i - lookback:i]

                if x_win.isna().any() or y_win.isna().any():
                    continue

                beta = np.polyfit(
                    x_win.values,
                    y_win.values,
                    1
                )[0]

                spread_hist = y_win - beta * x_win

                spread = spread_hist.iloc[-1]
                mean = spread_hist.mean()
                std = spread_hist.std()

                if std == 0 or np.isnan(std):
                    continue

                z = (spread - mean) / (std + 1e-8)

                if z > entry_z:
                    signal = -1
                elif z < -entry_z:
                    signal = 1
                elif abs(z) < exit_z:
                    signal = 0
                else:
                    signal = 0

                time_index.append(prices.index[i])
                spread_series.append(spread)
                z_series.append(z)
                signal_series.append(signal)

            if len(spread_series) < 20:
                continue

            # ==================================================
            # FORCE CLEAN DATETIME SERIES (CRITICAL FIX)
            # ==================================================
            spread_series = pd.Series(
                spread_series,
                index=pd.to_datetime(time_index)
            )

            z_series = pd.Series(
                z_series,
                index=pd.to_datetime(time_index)
            )

            signal_series = pd.Series(
                signal_series,
                index=pd.to_datetime(time_index)
            )

            hl = self.compute_half_life(spread_series)

            if hl < min_half_life:
                continue

            key = f"{sym_y}_{sym_x}"

            spreads_out[key] = {
                "spread": spread_series,
                "zscore": z_series
            }

            signals_out[key] = signal_series

        # ==================================================
        # CHART
        # ==================================================
        chart_cfg = self.cfg.get("chart", {})

        chart = self.build_chart(
            series=chart_cfg.get("series"),
            title=self.cfg.get("title", "Pair Trading Fallback"),
            charttype=chart_cfg.get("type", "line"),
            chartmode=chart_cfg.get("mode", "overlay")
        )

        # ==================================================
        # METRICS
        # ==================================================
        metrics = {
            "lookback": lookback,
            "corr_window": corr_window,
            "min_corr": min_corr,
            "entry_zscore": entry_z,
            "exit_zscore": exit_z,
            "max_pairs": max_pairs,
            "min_half_life": min_half_life,
            "pairs_used": len(spreads_out),
            "avg_half_life": float(np.mean([
                self.compute_half_life(v["spread"])
                for v in spreads_out.values()
            ])) if spreads_out else None
        }

        return StrategyResult(
            name="PairTradingFallback",
            data=self.data,
            metrics=metrics,
            signals={
                "spread": spreads_out,
                "signal": signals_out
            },
            chart=chart
        )