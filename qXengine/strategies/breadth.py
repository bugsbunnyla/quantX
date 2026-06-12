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
import pandas as pd

from .BaseStrategy import BaseStrategy
from ..StrategyResult import StrategyResult


class BreadthStrategy(BaseStrategy):

    def run(self):

        lookback = self.get_cfg("lookback", 20)
        ma_short = self.get_cfg("ma_short", 20)
        ma_long = self.get_cfg("ma_long", 50)

        # =====================================================
        # RETURNS MATRIX
        # =====================================================
        rets = pd.DataFrame({
            sym: df["ret"]
            for sym, df in self.data.items()
            if isinstance(df, pd.DataFrame) and "ret" in df.columns
        })

        if rets.empty:
            rets = pd.DataFrame({
                sym: df["close"].pct_change()
                for sym, df in self.data.items()
                if isinstance(df, pd.DataFrame) and "close" in df.columns
            })

        if rets.empty:
            return StrategyResult(
                name="BreadthStrategy",
                data=self.data,
                metrics={"error": "no valid return data"},
                signals={},
                chart={
                    "chartdata": pd.DataFrame()
                }
            )

        # =====================================================
        # LOOKBACK
        # =====================================================
        if lookback and len(rets) > lookback:
            rets = rets.tail(lookback)

        # =====================================================
        # CORE BREADTH MATRIX
        # =====================================================
        breadth = (rets > 0).sum(axis=1) / float(rets.shape[1])

        ma_short_series = breadth.rolling(ma_short, min_periods=1).mean()
        ma_long_series = breadth.rolling(ma_long, min_periods=1).mean()

        breadth_spread = ma_short_series - ma_long_series

        # =====================================================
        # SIGNALS (analysis layer only)
        # =====================================================
        signals = {
            "breadth": breadth,
            "ma_short": ma_short_series,
            "ma_long": ma_long_series,
            "breadth_spread": breadth_spread
        }

        # =====================================================
        # METRICS (scalar layer only)
        # =====================================================
        metrics = {
            "lookback": lookback,
            "ma_short": ma_short,
            "ma_long": ma_long,
            "avg_breadth": float(breadth.mean()),
            "assets": rets.shape[1]
        }

        # =====================================================
        # CHART MATRIX (THIS IS THE KEY FIX)
        # =====================================================
        chart_matrix = pd.DataFrame({
            "breadth": breadth,
            "ma_short": ma_short_series,
            "ma_long": ma_long_series,
            "breadth_spread": breadth_spread
        })

        # =====================================================
        # RESULT
        # =====================================================
        return StrategyResult(
            name="BreadthStrategy",
            data=self.data,
            metrics=metrics,
            signals=signals,
            chart={
                "chartdata": chart_matrix,
                "type": "line",
                "title": "Market Breadth & Participation Strength"
            }
        )