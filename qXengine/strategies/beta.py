# ===============================================================
# BetaNeutralStrategy : beta strategy in Quant Xpert
# Date: 2026/06/24 
# Author : bugsbunnyla
# Comment : Processing of beta neutral strategy
# ===============================================================
import numpy as np
import pandas as pd

from scipy.stats import linregress
from ..StrategyResult import StrategyResult
from ..strategies.BaseStrategy import BaseStrategy
#
#  rolling beta estimator = beta_t = Cov(asset, market) / Var(market)
#  residual extractor = alpha = asset - beta * market
#  standardization = z-score residual → comparable across assets
# Addres pure alpha αt​=rportfolio,t​−rSPY,t​
# 1rolling beta βi,t	​=Var(rm )Cov(ri	​,rm)
# 2beta-neutral return (correct time series) ri,tBN	​=ri,t	​−βi,t * ​rm,t	​
# 3portfolio aggregation Instead of mean():correct weighting: wi=1/σi
# 4cumulative 4Y curve (for chart)Ct​=k≤t∑​Sk​
import numpy as np
import pandas as pd

from scipy.stats import linregress
from ..StrategyResult import StrategyResult
from ..strategies.BaseStrategy import BaseStrategy
class BetaNeutralStrategy(BaseStrategy):

    # ==================================================
    # INDEX HANDLING
    # ==================================================
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

    # ==================================================
    # CLEAN SERIES
    # ==================================================
    def _clean_series(self, s):

        if s is None:
            return pd.Series(dtype=float)

        s = pd.Series(s).copy()

        s = s.replace([np.inf, -np.inf], np.nan).dropna()

        s.index = pd.to_datetime(s.index, errors="coerce")

        s = s[s.index.notna()]

        s = s[s.index > pd.Timestamp("2000-01-01")]

        s = s.sort_index()

        s = s[~s.index.duplicated(keep="last")]

        return s

    # ==================================================
    # LAST WINDOW
    # ==================================================
    def _last_window(self, s):
        s = self._clean_series(s)
        lookback = self.get_cfg("lookback", 252)
        return s.tail(lookback * 4)

    # ==================================================
    #  FORCE 4Y WINDOW (INTRADAY STYLE)
    # ==================================================
    def _last_4y(self, s):
        if s is None:
            return None
        s = self._clean_series(s)
        if s is None:
            return None
        return s.tail(252 * 4)

    # ==================================================
    # METRIC SERIES
    # ==================================================
    def _build_metric_series(self, rpt, symbols, cat, metric):
        series = {}
        #for symbol in symbols:
        try:
                s = rpt.loc[(cat, metric), symbols]
                s = pd.Series(s).copy()
                s = s.replace([np.inf, -np.inf], np.nan).dropna()
                s.index = pd.to_datetime(s.index, errors="coerce")
                s = s[s.index.notna()]
                if len(s) > 0:
                    series[symbols] = s
        except Exception as e:
                print(f"[WARN] {metric} {symbols}: {e}")
        return series

    #
    # 
    #
    def _build_metric_series_fo(self, fo, symbols, category, metric):
       return fo.get(metric)

    # ==================================================
    # RUN
    # ==================================================
    def run(self):
        #print("[DEBUG] BetaNeutralStrategy start")
        # ------------------------------------------
        # PREPARE DATA + BENCHMARK FIRST
        # ------------------------------------------
        data = self.data.copy()
        #print("[DEBUG] available symbols:", list(data.keys()))
        configured_benchmark = self.get_cfg("benchmark", None)
        benchmark_candidates = [
            configured_benchmark,
            "BTCUSDT",
            "SPY",
        ]
        benchmark = next((b for b in benchmark_candidates if b and b in data), None)
        if benchmark is None:
            print("[DEBUG] no valid benchmark found")
            return StrategyResult(
                name="BetaNeutralStrategy",
                data={},
                metrics={"error": "No benchmark found"},
                signals={},
                chart=None,
            )
        self.formulaOutput.assemble()
        fo = self.formulaOutput
        #print("[DEBUG] selected benchmark:", benchmark)
        # ------------------------------------------
        # COMPUTE FORMULAS
        # ------------------------------------------
        formulas = [
            "beta",
            "residual",
            "tstat_beta",
            "ic",
            "hit_ratio",
            "sharpe",
            "tstat_alpha",
            "drawdown",
            "corr_rm",
            "cvar",
            "volatility",
            "max_drawdown",
            "turnover",
            "slippage",
            "transaction_cost",
            "kelly",
        ]
        for f in formulas:
            try:
                fo.compute(f)
            except Exception as e:
                print("[WARN] formula failed:", f, e)
        beta_window = self.get_cfg("beta_window", 63)
        # ------------------------------------------
        # RETURNS / PRICE
        # ------------------------------------------
        ret_df = fo.get("ret")
        close_df = fo.get("mkt_price")
        if ret_df is None:
            #print("[DEBUG] no return dataframe")
            return StrategyResult(
                name="BetaNeutralStrategy",
                data={},
                metrics={"error": "No return data"},
                signals={},
                chart=None,
            )
        # FIXED INDENTATION
        if benchmark not in ret_df.columns:
            #print("[DEBUG] benchmark missing in returns:", benchmark)
            return StrategyResult(
                name="BetaNeutralStrategy",
                data={},
                metrics={"error": "Benchmark return missing"},
                signals={},
                chart=None,
            )
        benchmark_ret = ret_df[benchmark].replace([np.inf, -np.inf], np.nan).fillna(0)
        if close_df is not None and benchmark in close_df.columns:
            price_series = close_df[benchmark].dropna()
        else:
            price_series = pd.Series(dtype=float)

        # ------------------------------------------
        # RESIDUAL / BETA
        # ------------------------------------------
        #print("[DEBUG] ret shape:", ret_df.shape)
        fit_df = fo.get("fitted")
        residual_df = fo.get("residual")
        # SAFE DEBUG EXPORTS
        #try:
        #    ret_df.to_csv("debug_ret.csv")
        #    benchmark_ret.to_csv("debug_benchmark.csv")
        #    if fit_df is not None:
        #        fit_df.to_csv("debug_fitted.csv")
        #    if residual_df is not None:
        #        residual_df.to_csv("debug_residual.csv")
        #except Exception as e:
        #    print("[WARN] debug csv failed:", e)
        if residual_df is not None:
            #print("[DEBUG] residual before cleanup:", residual_df.shape)
            # Detect symbols as rows
            # and dates as columns
            if any(isinstance(c, pd.Timestamp) for c in residual_df.columns):
                residual_df = residual_df.T
            residual_df.index = pd.to_datetime(residual_df.index, errors="coerce")
            residual_df = residual_df.loc[residual_df.index.notna()]
            residual_df = residual_df.loc[:, ~residual_df.columns.duplicated()]
            residual_df = residual_df.replace([np.inf, -np.inf], np.nan)
        else:
            print("[DEBUG] residual dataframe missing")
        #print("===== RESIDUAL DEBUG =====")
        #if residual_df is not None:
            #print("[DEBUG] residual shape:", residual_df.shape)
            #print("[DEBUG] residual columns:", list(residual_df.columns)[:10])
            #print("[DEBUG] residual index:", residual_df.index[:3])
        beta_df = fo.get("beta")
        residuals = {}
        beta_values = {}
        for symbol in data:
            if symbol == benchmark:
                continue
            if residual_df is None:
                continue
            if symbol not in residual_df.columns:
                print("[DEBUG] no residual column:", symbol)
                continue
            residual_series = (
                residual_df[symbol].replace([np.inf, -np.inf], np.nan).dropna()
            )
            if residual_series.empty:
                print("[DEBUG] empty residual:", symbol)
                continue
            cleaned = self._last_window(residual_series)
            if cleaned.empty:
                print("[DEBUG] empty cleaned residual:", symbol)
                continue
            residuals[symbol] = cleaned

            if beta_df is not None and symbol in beta_df.index:
                 beta_value = beta_df.get(symbol, np.nan)
                 if np.isfinite(beta_value):
                    beta_values[symbol] = float(beta_value)
                 else:
                    beta_values[symbol] = np.nan
            else:
                beta_values[symbol] = np.nan

        #print("[DEBUG] residual count:", len(residuals))

        if not residuals:
            return StrategyResult(
                name="BetaNeutralStrategy",
                data={},
                metrics={"error": "No residual series"},
                signals={},
                chart=None,
            )

        # ==================================================
        # CORE LOOP (UNCHANGED FORMULA, FIXED TIME)
        # ==================================================
        bn_returns = {}
        beta_stats = {}

        for sym in data.keys():
           if sym == benchmark:
              continue
           # Make sure returns exist for this asset
           if sym not in ret_df.columns:
              print(f"[DEBUG] Missing return series for {sym}")
              continue
           asset_ret = ( ret_df[sym].replace([np.inf, -np.inf], np.nan).fillna(0) )

           # Align asset and benchmark by date
           aligned = pd.concat(
             [
              asset_ret.rename("asset"),
              benchmark_ret.rename("benchmark"),
             ],
             axis=1,
             join="inner",
            ).dropna()

           if len(aligned) < beta_window:
             print(f"[DEBUG] Not enough observations for {sym}")
             continue
           x = aligned["benchmark"]
           y = aligned["asset"]

           bn_series = []
           beta_series = []
           idx = []
           for iter in range(beta_window, len(aligned)):
             x_win = x.iloc[iter - beta_window:iter]
             y_win = y.iloc[iter - beta_window:iter]

             try:
               beta = linregress(x_win, y_win).slope
             except Exception:
               continue
             beta_series.append(beta)
             bn_series.append(  y.iloc[iter] - beta * x.iloc[iter]  )
             idx.append(aligned.index[iter])
           if not bn_series:
            continue
           series = pd.Series(bn_series, index=idx)
           std = series.std()

           if pd.notna(std) and std > 0:
              series = series / std

           bn_returns[sym] = self._last_4y(series)

           beta_stats[sym] = { "mean_beta": float(np.mean(beta_series)),"beta_vol": float(np.std(beta_series)),  "observations": len(beta_series),    }
           #print(type(data))
           #print(data.keys())

           #for sym, df in data.items():
            #print("Symbol:", sym)
            #print("Type:", type(df))
            #print("Columns:", df.columns.tolist())
            #break


        # ==================================================
        # PORTFOLIO (TIME SAFE)
        # ==================================================
        #aligned = pd.concat(bn_returns, axis=1).fillna(0)

        #vol = aligned.std() + 1e-8
        #weights = 1.0 / vol
        #weights = weights / weights.sum()

        #portfolio = aligned.dot(weights)
        

        # ------------------------------------------
        # PORTFOLIO
        # ------------------------------------------
        aligned = pd.concat(residuals, axis=1).fillna(0)
        vol = aligned.std() + 1e-8
        weights = 1 / vol
        weights = weights / weights.sum()
        portfolio = aligned.dot(weights)
        portfolio_curve = self._last_4y(portfolio.cumsum())
        pnl = self._last_window(portfolio.cumsum())
        benchmark_curve = self._last_window(
            benchmark_ret.reindex(pnl.index).fillna(0).cumsum()
        )
        symbols = list(residuals.keys())

        
        #rpt= fo.report()
        #print(type(rpt))
        #print(rpt.shape)
        #print(rpt.index)
        #print(rpt.columns)
        #print(rpt.head())
        # -------------------------------------------------
        # CHART METRICS
        # -------------------------------------------------
        """
        metric_series = {
            "ic": self._build_metric_series( rpt, symbols,   "intel","ic"  ),
            "hit_ratio": self._build_metric_series( rpt, symbols,  "basic","hit_ratio" ),
            "sharpe": self._build_metric_series( rpt, symbols, "risk","sharpe" ),
            "tstat_alpha": self._build_metric_series( rpt, symbols, "risk","tstat"  ),
            "drawdown": self._build_metric_series( rpt, symbols,"risk","drawdown"  ),
            "corr": self._build_metric_series( rpt, symbols, "basic","corr"   ),
            "cvar": self._build_metric_series( rpt,symbols,"risk","cvar"  ),
            "volatility":  self._build_metric_series( rpt,  symbols,  "risk","volatility" ),
            "alpha" :  self._build_metric_series( rpt, symbols, "alpha","alpha" ),              
            "beta" :  self._build_metric_series( rpt, symbols, "alpha","beta" ),
            "turnover":  self._build_metric_series( rpt, symbols, "execution","turnover" ),
            "slippage":  self._build_metric_series( rpt, symbols, "execution","slippage" ),
            "kelly": self._build_metric_series( rpt, symbols, "portfolio","kelly"),
            "max_drawdown":  self._build_metric_series( rpt, symbols, "risk","max_drawdown" ),
            "transaction_cost": self._build_metric_series( rpt, symbols, "execution","transaction_cost" )
        }    
        #print("metric series ",metric_series)    
        """
        metric_series1 = {
            "ic": self._build_metric_series_fo( fo, symbols,  "intel","ic"  ),
            "hit_ratio": self._build_metric_series_fo( fo, symbols, "basic","hit_ratio" ),
            "sharpe": self._build_metric_series_fo( fo, symbols, "risk","sharpe" ),
            "tstat_alpha": self._build_metric_series_fo( fo, symbols, "risk","tstat_alpha" ),
            "drawdown": self._build_metric_series_fo( fo, symbols, "risk","drawdown" ),
            "corr": self._build_metric_series_fo( fo, symbols, "basic","corr_rm"   ),
            "cvar": self._build_metric_series_fo( fo,symbols,"risk","cvar"   ),
            "volatility":  self._build_metric_series_fo( fo,  symbols,  "risk","volatility"  ),
            "alpha" :  self._build_metric_series_fo( fo, symbols,"alpha","alpha" ),              
            "beta" :  self._build_metric_series_fo( fo, symbols, "alpha","beta" ),
            "turnover":  self._build_metric_series_fo( fo, symbols, "execution","turnover" ),
            "slippage":  self._build_metric_series_fo( fo, symbols, "execution","slippage" ),
            "kelly": self._build_metric_series_fo( fo, symbols, "portfolio","kelly" ),
            "max_drawdown":  self._build_metric_series_fo( fo, symbols, "risk","max_drawdown" ),
            "transaction_cost": self._build_metric_series_fo( fo, symbols, "execution","transaction_cost" )
        }    
        #print("metric series fo ",metric_series1)    

        # ------------------------------------------
        # CHART DATA
        # ------------------------------------------
                    
        chartdata = {
            "pnl": self._clean_series(pnl),
            "benchmark": self._clean_series(benchmark_curve),
            "portfolio_curve": portfolio_curve,
            "price": self._clean_series(price_series),
            "signal": {
                "x": symbols,
                "y": [1 for _ in symbols],
            },
            "residual": {
                "x": symbols,
                "y": [residuals[s].mean() for s in symbols],
            },
            "beta_data": {
                "x": symbols,
                "y": [beta_values.get(s, np.nan) for s in symbols],
            },
            "rebalance_events": [],
            **metric_series1, 
            "assets":  symbols
        }

        # ------------------------------------------
        # BUILD CHART
        # ------------------------------------------

        chart = self.build_chart(
            charttype="line",
            chartmode="lines",
            title=self.get_cfg("title", "Beta Neutral Strategy"),
            chartdata=chartdata,
            series=self.get_cfg("chart", {}).get("series", []),
        )

        # ------------------------------------------
        # RETURN RESULT
        # ------------------------------------------
        
        return StrategyResult(
            name="BetaNeutralStrategy",
            data=chartdata,
            metrics={
                "benchmark": benchmark,
                "assets": len(bn_returns),
                "beta_window": beta_window,
                "beta_stats": beta_stats,
                "portfolio_weights": weights.to_dict()
            },
            signals=bn_returns,
            chart=chart,
        )





# ============================================================
# END OF BETA NEUTRAL STRATEGY
# ============================================================