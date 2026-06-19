import pandas as pd

from .BaseStrategy import BaseStrategy
from ..StrategyResult import StrategyResult

#
#1. Returns matrix For each asset i: ri,t	​={​Closei,t/Closei,t−1	​−1  ​if close available if provided
#	​                                          reti,t
#2. Breadth (Participation Rate) At time t:Breadtht	​=N#(ri,t	​>0)	​Where:N = number of assetsri,t	​>0 = asset closed positive Output range:0≤Breadth t ≤1 
#3. Step 3 — Smoothed Breadth (moving averages) Short MA:MAshort,t	​=SMA(Breadtht	​,20)Long MA:MAlong,t	​=SMA(Breadtht	​,50)
#4. Breadth Momentum / Spread Spreadt	​=MAshort,t	​−MAlong,t 	​Interpretation: 0 → improving participation< 0 → weakening participation 
#

class BreadthStrategy(BaseStrategy):

    # =====================================================
    #  FORCE CLEAN DATETIME INDEX 
    # =====================================================
    def _prepare_returns(self):

        frames = {}

        for sym, df in self.data.items():

            if not isinstance(df, pd.DataFrame):
                continue

            tmp = df.copy()

            # ---- CASE 1: explicit date column ----
            if "date" in tmp.columns:
                tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
                tmp = tmp.dropna(subset=["date"])
                tmp = tmp[tmp["date"] > pd.Timestamp("2000-01-01")]
                tmp = tmp.sort_values("date")
                tmp = tmp.set_index("date")

            # ---- CASE 2: index already exists ----
            else:
                tmp.index = pd.to_datetime(tmp.index, errors="coerce")
                tmp = tmp[tmp.index.notna()]
                tmp = tmp[tmp.index > pd.Timestamp("2000-01-01")]
                tmp = tmp.sort_index()

            # ---- ensure numeric returns ----
            if "ret" in tmp.columns:
                frames[sym] = tmp["ret"]
            elif "close" in tmp.columns:
                frames[sym] = tmp["close"].pct_change()

        return pd.DataFrame(frames).dropna()

    # =====================================================
    def run(self):

        lookback = self.get_cfg("lookback", 20)
        ma_short = self.get_cfg("ma_short", 20)
        ma_long = self.get_cfg("ma_long", 50)

        # =====================================================
        # CLEAN UNIVERSE 
        # =====================================================
        rets = self._prepare_returns()

        if rets.empty:
            return StrategyResult(
                name="BreadthStrategy",
                data=self.data,
                metrics={"error": "no valid return data"},
                signals={},
                chart={"chartdata": pd.DataFrame()}
            )

        # =====================================================
        # OPTIONAL LOOKBACK WINDOW (SAFE NOW)
        # =====================================================
        if len(rets) > lookback:
            rets = rets.tail(lookback)

        # =====================================================
        # CORE BREADTH (UNCHANGED FORMULA)
        # =====================================================
        breadth = (rets > 0).sum(axis=1) / float(rets.shape[1])

        ma_short_series = breadth.rolling(ma_short, min_periods=1).mean()
        ma_long_series = breadth.rolling(ma_long, min_periods=1).mean()

        breadth_spread = ma_short_series - ma_long_series

        # =====================================================
        # FINAL CHART MATRIX (INDEX IS NOW CLEAN DATETIME)
        # =====================================================
        chart_matrix = pd.DataFrame({
            "breadth": breadth,
            "ma_short": ma_short_series,
            "ma_long": ma_long_series,
            "breadth_spread": breadth_spread
        })

        chart_matrix.index = pd.to_datetime(chart_matrix.index, errors="coerce")
        chart_matrix = chart_matrix[chart_matrix.index.notna()]
        chart_matrix = chart_matrix[chart_matrix.index > pd.Timestamp("2000-01-01")]
        chart_matrix = chart_matrix.sort_index()

        # =====================================================
        # RESULT
        # =====================================================
        return StrategyResult(
            name="BreadthStrategy",
            data=self.data,
            metrics={
                "lookback": lookback,
                "ma_short": ma_short,
                "ma_long": ma_long,
                "avg_breadth": float(breadth.mean()),
                "assets": rets.shape[1]
            },
            signals={
                "breadth": breadth,
                "ma_short": ma_short_series,
                "ma_long": ma_long_series,
                "breadth_spread": breadth_spread
            },
            chart={
                "chartdata": chart_matrix,
                "type": "line",
                "title": "Market Breadth & Participation Strength"
            }
        )