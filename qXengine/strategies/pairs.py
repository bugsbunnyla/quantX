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

    # ==================================================
    # HALF-LIFE (UNCHANGED)
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

            beta = np.polyfit(lagged.values, delta.values, 1)[0]

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
        data = self.data.copy()
        #print("[DEBUG] data",  data.keys())

        lookback = cfg.get("lookback", 252)
        entry_z = cfg.get("entry_zscore", 2.0)
        exit_z = cfg.get("exit_zscore", 0.5)
        max_pairs = cfg.get("max_pairs", 10)
        min_half_life = cfg.get("min_half_life", 5)

        # ==================================================
        # FIX: DATE → INDEX
        # ==================================================
        #cleaned = {}

        #for sym, df in data.items():
        #  tmp = df.copy()
        #  if isinstance(tmp.index, pd.DatetimeIndex):
            # Already correctly indexed
        #    pass
        #  elif "date" in tmp.columns:
        #    tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
        #    tmp = tmp.dropna(subset=["date"])
        #    tmp = tmp.set_index("date")
        #  else:
            # Existing index contains dates
        #    tmp.index = pd.to_datetime(tmp.index, errors="coerce")
        #    tmp = tmp[~tmp.index.isna()]
        #  tmp = tmp.sort_index()
        #  cleaned[sym] = tmp
        #print("[DEBUG] cleaned and data", cleaned.keys(), data.keys())

        # ==================================================
        # PRICE MATRIX
        # ==================================================
        fo=self.formulaOutput
        fo.assemble()
        prices = fo.get("mkt_price")
        #print("[DEBUG] prices ",len(prices),prices)    
        #prices = prices.dropna()
        #prices = prices.sort_index()        
        #print("[DEBUG] prices sort",len(prices),prices)   
        #if prices.empty or prices.shape[1] < 2:
        if prices.empty:
            return StrategyResult(
                name="PairTrading",
                data=self.data,
                metrics={"error": "insufficient data"},
                signals={},
                chart = None
            )

        # ==================================================
        # RETURNS + CORRELATION (UNCHANGED)
        # ==================================================
        rets = fo.get("ret").dropna()
        corr = rets.corr().abs()

        pairs = (
            corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            .stack()
            .sort_values(ascending=False)
        )
        #print("[DEBUG] pairs", pairs)
        pairs = pairs[pairs < 0.999]
        pairs = pairs.dropna()
        top_pairs = list(pairs.head(max_pairs).index)

        spreads_out = {}
        signals_out = {}

        # ==================================================
        # PAIR LOOP (UNCHANGED FORMULAS)
        # ==================================================
        for sym_x, sym_y in top_pairs:

            if sym_x not in prices.columns or sym_y not in prices.columns:
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

                # OLS hedge ratio (UNCHANGED)
                beta = np.polyfit(x_win.values, y_win.values, 1)[0]

                spread_hist = y_win - beta * x_win
                spread = spread_hist.iloc[-1]

                mean = spread_hist.mean()
                std = spread_hist.std()

                if std == 0 or np.isnan(std):
                    continue

                z = (spread - mean) / (std + 1e-8)

                # SIGNAL LOGIC (UNCHANGED)
                if z > entry_z:
                    signal = -1
                elif z < -entry_z:
                    signal = 1
                else:
                    signal = 0

                # NOW VALID BECAUSE INDEX IS DATETIME
                t = prices.index[i]

                time_index.append(t)
                spread_series.append(spread)
                z_series.append(z)
                signal_series.append(signal)

            if len(spread_series) < 20:
                continue

            spread_series = pd.Series(spread_series, index=pd.DatetimeIndex(time_index))
            z_series = pd.Series(z_series, index=pd.DatetimeIndex(time_index))
            signal_series = pd.Series(signal_series, index=pd.DatetimeIndex(time_index))

            hl = self.compute_half_life(spread_series)

            if hl < min_half_life:
                continue

            key = f"{sym_y}_{sym_x}"

            spreads_out[key] = {
                "spread": spread_series,
                "zscore": z_series
            }

            signals_out[key] = signal_series
            #print("spread", spreads_out,  signals_out)
        # ==================================================
        # BUILD CHART
        # ==================================================
        chartdata = {
           "signal" : signals_out,
           "spread" : spreads_out
        }
        chart = self.build_chart(
            series=self.cfg.get("chart").get("series"),
            title=self.cfg.get("title"),
            charttype=self.cfg.get("chart").get("type"),
            chartmode=self.cfg.get("chart").get("mode"),
            chartdata=chartdata
        )

        # ==================================================
        # METRICS
        # ==================================================
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
       

        # ==================================================
        # FINAL OUTPUT (UNCHANGED STRUCTURE)
        # ==================================================
        return StrategyResult(
            name="PairTrading",
            data=self.data,
            metrics=metrics,
            signals=chartdata,
            chart=chart
        )