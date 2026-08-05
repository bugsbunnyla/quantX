# =====================================================================
# IndustryMomentumStrategy : stock and industry strategy in Quant Xpert
# Date: 2026/06/24 
# Author : bugsbunnyla
# Comment : JT and MG paper references of stock and industry momentum 
# applied strategy processing in Quant Xpert
# =====================================================================
import numpy as np
import pandas as pd

from .BaseStrategy import BaseStrategy
from ..StrategyResult import StrategyResult

#
# 1. Momentum Formula 
# Mi​(t)=(Pi​(t)/Pi​(t−formation))​−1 where Pi​ = asset price formation = 252 days by default This is standard 12-month momentum. here  momentum = prices.pct_change(formation)
# 2. Industry strength as smoothed momentum
# ISi​(t)=1/N ​k=0∑N−1​Mi​(t−k) smoothed momentum over time.  industry_strength = (     momentum    .rolling(industry_window)    .mean())
# 3. Ranking  Ranki(t)=Percentile(ISi​(t)) Every date:strongest asset → near 1.0 weakest asset → near 0.0 and a standard crosssection ranking ranks = industry_strength.rank(axis=1,pct=True)
# 4. Top quantile selection Signali(t)=1ifRanki	​(t)>70%  Longest strong asset ranks for top_q as (ranks >= (1-top_q)).astype(float) 
# 5. Portfolio Construction Rp​(t)=1/N​i∑​Signali​(t−1)Ri​(t) Equal-weight momentum portfolio.  portfolio_returns here portfolio_returns =(prices.pct_change()* portfolio_signal).mean(axis=1)
# 6. Portfolio equity curve V(t)=k=1∏t​(1+Rp​(k)) portfolio_and is a standard cumulative return  here momentum = (1+portfolio_returns).cumprod() 
# 7. Benchmark Benchmark(t)=P(0)P(t)​ for chart comparison here benchmark =price / first_price 
# 8  Rebalance events rebalance_events.iloc[::holding] = 1 for chart markers
# IndustryReturn not here IndustryReturnj	​=1/N ​i∈Industryj​∑​Returni  ​then:IndustryMomentumj​=∏(1+IndustryReturnj​)−1 . 
#  Rank industries: Rank(IndustryMomentumj​) then buy stocks only inside winning industries.  no industry grouping 
# As a cross sectional industry momentum - Cross-Sectional Momentum (Stock Momentum) Momentum i	​=Pi(t)/Pi​(t−k)	​−1 Rank all stocks:Rank(Momentumi	​)
# Classic reference:  Narasimhan Jegadeesh Sheridan Titman  Core idea: Buy stocks that have outperformed other stocks and sell stocks that have underperformed other stocks.
# Industry Momentum (Moskowitz & Grinblatt) Paper: Tobias Moskowitz and Mark Grinblatt Published: Do Industries Explain Momentum? 
# Core finding  -> A large portion of stock momentum comes from industry performance rather than firm-specific performance.
# Group stocks by industry.  as Tech energy etc Calculate industry return. IndustryReturnj	​=1/N​i∈j∑​Returni	​ next is Step 4 Buy stocks from winning industries. may Not be  best individual stocks.
# Cross-sectional momentum:  Rank(StockMomentum)  vs Industry momentum: Rank(IndustryMomentum)
# Cross-Sectional Momentum  Pros: Higher alpha More concentrated Captures stock-specific winners Cons: Higher turnover Higher idiosyncratic risk More crashes
# Industry Momentum Pros:More stable Lower turnover Lower stock-specific risk Cons: Slower Less alpha More benchmark-like 
# Feature				Cross-Sectional Momentum		Industry Momentum
# Rank level				Stocks				Industries
# Selection				Strongest stocks			Strongest industries
# Requires industry classification	No				Yes
# Concentration				High				Lower
# Turnover				Higher				Lower
# Alpha potential			Higher				Moderate
# Academic source			Jegadeesh–Titman 		Moskowitz–Grinblatt
#
# 1. Stock Momentum Individual security momentum: Mi,tstock	​=Pi,t/Pi,t−formation	​−1 
# 2. Industry Momentum Industry average momentum:   Mk,tindustry	​=/Nk 	​i∈k∑​Mi,tstock	​where:k = industry Nk	​ = number of stocks in industry
# signals = {     "stock_momentum": {        "AAPL": stock_mom_aapl,        "MSFT": stock_mom_msft    },    "industry_momentum": {        "Technology": tech_momentum,        "Financials": financial_momentum     },
#  "stock_signal": {         "AAPL": signal_aapl,        "MSFT": signal_msft    },    "industry_signal": {        "Technology": tech_signal,        "Financials": fin_signal   }}
# metrics = {     "formation": formation,    "industry_window": industry_window,    "holding": holding,    "top_quantile": top_q,    "stocks": len(stock_momentum),    "industries": len(industry_momentum),
# "avg_stock_momentum":        float(stock_mom_df.mean().mean()),    "avg_industry_momentum":        float(industry_mom_df.mean().mean()),    "top_industry":        best_industry,    "bottom_industry":  worst_industry}
# chart- signals = {     "industry_momentum_index":        industry_momentum_index,    "stock_momentum_index":        stock_momentum_index,    "industry_signal":        industry_signal_series,    
#                    "stock_signal":        stock_signal_series,    "rebalance_events":        rebalance_events}
class IndustryMomentumStrategy(BaseStrategy):

    # ==================================================
    # INDEX NORMALIZER
    # ==================================================
    def _normalize(self, df):

        df = df.copy()

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
            df = df.set_index("date")
        else:
            df.index = pd.to_datetime(df.index, errors="coerce")

        df = df[df.index.notna()]
        return df.sort_index()

    # ==================================================
    # INDUSTRY MAP
    # ==================================================
    def _industry_map(self, cols):

        crypto = {"BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","DOGEUSDT"}
        macro = {"SPY","QQQ","IWM","TLT","GLD"}
        tech = {"AAPL","MSFT","NVDA","AMD"}
        fin = {"JPM","GS","BAC"}

        m = {}
        for s in cols:
            if s in crypto:
                m[s] = "Crypto"
            elif s in macro:
                m[s] = "Macro"
            elif s in tech:
                m[s] = "Tech"
            elif s in fin:
                m[s] = "Financials"
            else:
                m[s] = "Other"
        return m

    # ==================================================
    # MAIN
    # ==================================================
    def run(self):

        cfg = self.cfg
        chart_cfg = cfg.get("chart", {})

        formation = cfg.get("formation", 252)
        top_q = cfg.get("top_quantile", 0.3)

        # --------------------------------------------------
        # CLEAN DATA
        # --------------------------------------------------
        clean = {}

        for sym, df in self.data.items():
            if "close" not in df.columns:
                continue
            clean[sym] = self._normalize(df)

        prices = pd.DataFrame({k: v["close"] for k, v in clean.items()}).sort_index()

        if prices.empty or prices.shape[1] < 2:
            return self.build_result(
                signals={},
                metrics={"error": "insufficient data"},
                chart=None
            )

        master_index = prices.index

        # ==================================================
        # RETURNS
        # ==================================================
        returns = prices.pct_change()

        # ==================================================
        # ================= STOCK MOMENTUM =================
        # Jegadeesh & Titman (1993)
        # ==================================================
        stock_signal_raw = prices.pct_change(formation).shift(1)

        stock_rank = stock_signal_raw.rank(axis=1, pct=True)

        stock_long = (stock_rank > (1 - top_q)).astype(float)
        stock_short = (stock_rank < top_q).astype(float)

        #  correct portfolio construction (FIXED)
        stock_weights = stock_long - stock_short

        stock_port_ret = (returns * stock_weights.shift(1)).mean(axis=1)

        stock_equity = (1 + stock_port_ret.fillna(0)).cumprod()

        stock_signal = (stock_long > 0).astype(int)

        stock_momentum_factor = stock_signal_raw.mean(axis=1)

        # ==================================================
        # =============== INDUSTRY MOMENTUM ===============
        # Moskowitz & Grinblatt (1999)
        # ==================================================
        ind_map = self._industry_map(prices.columns)

        ind_returns = {}

        for ind in set(ind_map.values()):
            members = [s for s in prices.columns if ind_map[s] == ind]
            if not members:
                continue
            ind_returns[ind] = returns[members].mean(axis=1)

        ind_returns = pd.DataFrame(ind_returns)

        #  FIX: use cumulative formation return, NOT rolling mean
        ind_signal_raw = ind_returns.pct_change(formation).shift(1)

        ind_rank = ind_signal_raw.rank(axis=1, pct=True)

        ind_long = (ind_rank > (1 - top_q)).astype(float)
        ind_short = (ind_rank < top_q).astype(float)

        #  correct portfolio construction
        ind_weights = ind_long - ind_short

        ind_port_ret = (ind_returns * ind_weights.shift(1)).mean(axis=1)

        industry_equity = (1 + ind_port_ret.fillna(0)).cumprod()

        industry_signal = (ind_long > 0).astype(int)

        industry_momentum_factor = ind_signal_raw.mean(axis=1)

        # ==================================================
        # BENCHMARK
        # ==================================================
        benchmark = None
        if "SPY" in prices.columns:
            benchmark = (1 + prices["SPY"].pct_change().fillna(0)).cumprod()

        # ==================================================
        # REBALANCE EVENTS
        # ==================================================
        rebalance = (stock_long.sum(axis=1) > 0).astype(int)

        # ==================================================
        # ALIGNMENT
        # ==================================================
        def A(x):
            return x.reindex(master_index) if x is not None else None

        # ==================================================
        # CHARTDATA (PAPER-CORRECT STRUCTURE)
        # ==================================================
        chartdata = {
            "stock_equity": A(stock_equity),
            "industry_equity": A(industry_equity),
            "benchmark": A(benchmark),

            # factors (non-cumulative)
            "stock_momentum": A(stock_momentum_factor),
            "industry_momentum": A(industry_momentum_factor),

            # signals
            "stock_signal": A(stock_signal),
            "industry_signal": A(industry_signal),
            "rebalance_events": A(rebalance),

            "assets": list(prices.columns),
            "industries": list(ind_returns.columns)
        }

        chart = self.build_chart(
            charttype=chart_cfg.get("type", "line"),
            chartmode=chart_cfg.get("mode", "overlay"),
            title=chart_cfg.get("title", "Industry & Stock Momentum (Paper Corrected)"),
            chartdata=chartdata,
            series=chart_cfg.get("series", [])
        )

        return StrategyResult(
     name="IndustryMomentumStrategy",
     data = self.data,
            signals={
                "stock_signal": A(stock_signal),
                "industry_signal": A(industry_signal),
                "rebalance_events": A(rebalance)
            },
            metrics={
                "formation": formation,
                "top_quantile": top_q,
                "stocks": len(prices.columns),
                "industries": len(ind_returns.columns)
            },
            chart=chart
        )
# ===========================================================
# END OF INDUSTRY MOMENTUM
# ===========================================================