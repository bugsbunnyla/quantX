import numpy as np
import pandas as pd

from ..StrategyResult import StrategyResult
from ..strategies.BaseStrategy import BaseStrategy
# A classic simplified statistical arbitrage model
#1. Returns (pair selection input)rt	​=​(Pt −Pt−1)/Pt−1	​	​	​
#2. Pair selection (correlation filter) ρxy	​=corr(rx,ry)
#3. Hedge ratio (OLS regression)You are using:y=α+βx implemented simplified (no intercept used in spread): β=argmin∣∣y−βx∣∣**2
#4. Spread construction spreadt	​=yt	​−βxt	​
#5. Rolling mean & standard deviation μt	​=1/N *	​i=t−N∑t	​spreadi	  and ​σt​=std(spreadt−N:t)
#6. Z-score (mean reversion signal) zt	​=	​spreadt	​−μt/σt
#7. Trading signal logic signalt	​=⎩⎨⎧	​−1 +1 0	 ​z>entry z<−entry  ∣z∣<exit	​
#8. Half life mean reversion filtered Ornstein–Uhlenbeck process: ΔSt	​=λSt−1	​+ϵt	​Then:half_life=−ln(2)/λ
class PairTrading(BaseStrategy):

    # ----------------------------
    # HALF-LIFE ESTIMATION (SAFE)
    # ----------------------------
    def compute_half_life(self, spread: pd.Series) -> float:
        try:
            spread = spread.dropna()

            if len(spread) < 20:
                return np.inf

            lagged = spread.shift(1).dropna()
            delta = spread.diff().dropna()

            if len(lagged) != len(delta):
                return np.inf

            # OU approximation
            beta = np.polyfit(lagged.values, delta.values, 1)[0]

            if beta >= 0 or np.isnan(beta):
                return np.inf

            return -np.log(2) / beta

        except:
            return np.inf

    # ----------------------------
    # MAIN STRATEGY
    # ----------------------------
    def run(self):

        cfg = self.cfg

        lookback = cfg.get("lookback", 252)
        entry_z = cfg.get("entry_zscore", 2.0)
        exit_z = cfg.get("exit_zscore", 0.5)
        max_pairs = cfg.get("max_pairs", 10)
        min_half_life = cfg.get("min_half_life", 5)

        # ----------------------------
        # PRICE MATRIX
        # ----------------------------
        prices = pd.DataFrame({
            sym: df["close"]
            for sym, df in self.data.items()
            if "close" in df.columns
        }).dropna()

        if prices.empty or prices.shape[1] < 2:
            return StrategyResult(
                name="PairTrading",
                data=self.data,
                metrics={"error": "insufficient data"},
                signals={}
            )

        # ----------------------------
        # RETURNS + CORRELATION
        # ----------------------------
        rets = prices.pct_change().dropna()
        corr = rets.corr().abs()

        pairs = (
            corr.unstack()
            .sort_values(ascending=False)
            .drop_duplicates()
        )

        pairs = pairs[pairs < 0.999]
        top_pairs = list(pairs.head(max_pairs).index)

        spreads_out = {}
        signals_out = {}

        # ----------------------------
        # PAIR LOOP
        # ----------------------------
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

                # ----------------------------
                # OLS hedge ratio
                # ----------------------------
                beta = np.polyfit(x_win.values, y_win.values, 1)[0]

                spread_hist = y_win - beta * x_win

                spread = spread_hist.iloc[-1]
                mean = spread_hist.mean()
                std = spread_hist.std()

                if std == 0 or np.isnan(std):
                    continue

                z = (spread - mean) / (std + 1e-8)

                # ----------------------------
                # SIGNAL LOGIC
                # ----------------------------
                if z > entry_z:
                    signal = -1
                elif z < -entry_z:
                    signal = 1
                elif abs(z) < exit_z:
                    signal = 0
                else:
                    signal = 0

                # ----------------------------
                # STORE WITH TIME INDEX
                # ----------------------------
                t = prices.index[i]

                time_index.append(t)
                spread_series.append(spread)
                z_series.append(z)
                signal_series.append(signal)

            if len(spread_series) < 20:
                continue

            spread_series = pd.Series(spread_series, index=time_index)
            z_series = pd.Series(z_series, index=time_index)
            signal_series = pd.Series(signal_series, index=time_index)

            # ----------------------------
            # HALF-LIFE FILTER
            # ----------------------------
            hl = self.compute_half_life(spread_series)

            if hl < min_half_life:
                continue

            key = f"{sym_y}_{sym_x}"

            spreads_out[key] = {
                "spread": spread_series,
                "zscore": z_series
            }

            signals_out[key] = signal_series

        # ----------------------------
        # BUILD CHART (BASE CLASS)
        # ----------------------------
        chart = self.build_chart(
            series= self.cfg.get("chart").get("series"),
     title=self.cfg.get("title"),
     charttype=self.cfg.get("chart").get("type"),
     chartmode=self.cfg.get("chart").get("mode"),
        )

        # ----------------------------
        # METRICS
        # ----------------------------
        metrics = {
            "lookback": lookback,
            "entry_zscore": entry_z,
            "exit_zscore": exit_z,
            "max_pairs": max_pairs,
            "min_half_life": min_half_life,
            "pairs_used": len(spreads_out),
            "avg_half_life": float(np.mean([
                self.compute_half_life(v["spread"]) for v in spreads_out.values()
            ])) if spreads_out else None
        }

        # ----------------------------
        # FINAL OUTPUT
        # ----------------------------
        return StrategyResult(
            name="PairTrading",
            data=self.data,
            metrics=metrics,
            signals={
                "spread": spreads_out,
                "signal": signals_out
            },
            chart=chart
        )