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
# Missed Industry Momentum (Moskowitz & Grinblatt) Paper: Tobias Moskowitz and Mark Grinblatt Published: Do Industries Explain Momentum? 
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
class IndustryMomentumStrategyBase(BaseStrategy):

    # ==================================================
    # MAIN STRATEGY
    # ==================================================
    def run(self):

        cfg = self.cfg

        formation = cfg.get("formation", 252)
        holding = cfg.get("holding", 21)
        industry_window = cfg.get("industry_window", 252)
        top_q = cfg.get("top_quantile", 0.30)

        # ==================================================
        # BUILD PRICE MATRIX
        # ==================================================
        prices = pd.DataFrame({
            sym: df["close"]
            for sym, df in self.data.items()
            if "close" in df.columns
        })

        prices = prices.dropna(how="all")

        if prices.empty:

            return StrategyResult(
                name="IndustryMomentumStrategy",
                data=self.data,
                metrics={"error": "no valid price data"},
                signals={}
            )

        # ==================================================
        # MOMENTUM RETURNS
        # ==================================================
        momentum = prices.pct_change(formation)

        # ==================================================
        # INDUSTRY STRENGTH
        # ==================================================
        industry_strength = (
            momentum
            .rolling(industry_window)
            .mean()
        )

        # ==================================================
        # CROSS-SECTIONAL RANKING
        # ==================================================
        ranks = industry_strength.rank(
            axis=1,
            pct=True
        )

        # ==================================================
        # TOP QUANTILE MEMBERSHIP
        # ==================================================
        signal_matrix = (
            ranks >= (1 - top_q)
        ).astype(float)

        # ==================================================
        # PORTFOLIO MOMENTUM
        # ==================================================
        portfolio_signal = signal_matrix.shift(1)

        portfolio_returns = (
            prices.pct_change()
            * portfolio_signal
        ).mean(axis=1)

        portfolio_returns = portfolio_returns.fillna(0)

        portfolio_momentum = (
            1 + portfolio_returns
        ).cumprod()

        # ==================================================
        # AVERAGE INDUSTRY MOMENTUM
        # ==================================================
        avg_strength = industry_strength.mean(axis=1)

        momentum_ma = avg_strength.rolling(
            industry_window
        ).mean()

        # ==================================================
        # REBALANCE EVENTS
        # ==================================================
        rebalance_events = pd.Series(
            0,
            index=portfolio_momentum.index
        )

        rebalance_events.iloc[::holding] = 1

        # ==================================================
        # BENCHMARK
        # ==================================================
        benchmark = None

        benchmark_symbol = (
            cfg.get("chart", {})
               .get("benchmark")
        )

        if benchmark_symbol in prices.columns:

            benchmark = (
                prices[benchmark_symbol]
                / prices[benchmark_symbol].iloc[0]
            )

        # ==================================================
        # PER-ASSET SIGNALS
        # ==================================================
        asset_signals = {}

        for sym in signal_matrix.columns:

            asset_signals[sym] = (
                signal_matrix[sym]
                .fillna(0)
            )

        # ==================================================
        # BUILD CHART
        # ==================================================
        chart = self.build_chart(
            series=cfg.get("chart", {}).get("series"),
            title=cfg.get("title"),
            charttype=cfg.get("chart", {}).get("type"),
            chartmode=cfg.get("chart", {}).get("mode")
        )

        # ==================================================
        # METRICS
        # ==================================================
        metrics = {

            "formation": formation,

            "holding": holding,

            "industry_window": industry_window,

            "top_quantile": top_q,

            "assets": int(prices.shape[1]),

            "avg_industry_strength":
                float(avg_strength.mean())
                if not avg_strength.empty
                else None,

            "latest_strength":
                float(avg_strength.iloc[-1])
                if not avg_strength.empty
                else None,

            "portfolio_return":
                float(
                    portfolio_momentum.iloc[-1] - 1
                )
                if not portfolio_momentum.empty
                else None
        }

        # ==================================================
        # SIGNALS
        # ==================================================
        signals = {

            "industry_strength": avg_strength,

            "momentum_ma": momentum_ma,

            "portfolio_momentum": portfolio_momentum,

            "rebalance_events": rebalance_events,

            "benchmark": benchmark,

            "asset_signals": asset_signals
        }

        # ==================================================
        # RETURN
        # ==================================================
        return StrategyResult(
            name="IndustryMomentumStrategy",
            data=self.data,
            metrics=metrics,
            signals=signals,
            chart=chart
        )


#====================================================== 
# INDUSTRY and STOCK 
#======================================================

import numpy as np
import pandas as pd

from .BaseStrategy import BaseStrategy
from ..StrategyResult import StrategyResult


class IndustryMomentumStrategy(BaseStrategy):

    # ==================================================
    # INDUSTRY MAP (UNCHANGED LOGIC)
    # ==================================================
    def _get_industry_map(self, columns):

        crypto_assets = {
            "BTCUSDT", "ETHUSDT", "SOLUSDT",
            "BNBUSDT", "XRPUSDT", "DOGEUSDT"
        }

        macro_assets = {
            "SPY", "QQQ", "IWM", "TLT", "GLD"
        }

        tech_assets = {"AAPL", "MSFT", "NVDA", "AMD"}
        financial_assets = {"JPM", "GS", "BAC"}

        mapping = {}

        for s in columns:
            if s in crypto_assets:
                mapping[s] = "Crypto"
            elif s in macro_assets:
                mapping[s] = "Macro"
            elif s in tech_assets:
                mapping[s] = "Technology"
            elif s in financial_assets:
                mapping[s] = "Financials"
            else:
                mapping[s] = "Other"

        return mapping


    # ==================================================
    # SAFE NORMALIZER (CRITICAL FIX)
    # ==================================================
    def _normalize_df(self, df):

        df = df.copy()

        # CASE 1: date column exists
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date"])
            df = df.set_index("date")

        # CASE 2: index already exists
        else:
            df.index = pd.to_datetime(df.index, errors="coerce")

        df = df[~df.index.isna()]
        df = df.sort_index()

        return df


    # ==================================================
    # MAIN
    # ==================================================
    def run(self):

        cfg = self.cfg

        formation = cfg.get("formation", 252)
        industry_window = cfg.get("industry_window", 252)
        top_q = cfg.get("top_quantile", 0.30)

        # ==================================================
        # STEP 1: CLEAN INPUT DATA
        # ==================================================
        clean = {}

        for sym, df in self.data.items():
            if "close" not in df.columns:
                continue
            clean[sym] = self._normalize_df(df)

        # ==================================================
        # STEP 2: BUILD PRICE MATRIX (ONLY HERE INDEX IS FIXED)
        # ==================================================
        prices = pd.DataFrame({
            sym: df["close"]
            for sym, df in clean.items()
        }).sort_index()

        # 🚨 DEBUG (CRITICAL)
        #print("\n[DEBUG] prices.index (FINAL SOURCE):")
        #print(prices.index[:5])
        #print("dtype:", prices.index.dtype)

        if prices.empty or prices.shape[1] < 2:
            return StrategyResult(
                name="IndustryMomentumStrategy",
                data=self.data,
                metrics={"error": "insufficient data"},
                signals={}
            )

        # ==================================================
        # RETURNS + MOMENTUM
        # ==================================================
        returns = prices.pct_change()
        stock_momentum = prices.pct_change(formation)

        # ==================================================
        # INDUSTRY GROUPING
        # ==================================================
        industry_map = self._get_industry_map(prices.columns)

        industry_frames = {}

        for ind in set(industry_map.values()):

            members = [s for s in prices.columns if industry_map[s] == ind]

            if not members:
                continue

            industry_frames[ind] = stock_momentum[members].mean(axis=1)

        industry_momentum = pd.DataFrame(industry_frames).fillna(0)

        industry_momentum_smoothed = (
            industry_momentum.rolling(industry_window)
            .mean()
            .fillna(0)
        )

        # ==================================================
        # SIGNALS
        # ==================================================
        stock_rank = stock_momentum.rank(axis=1, pct=True)
        stock_signal = (stock_rank > (1 - top_q)).astype(float)

        industry_rank = industry_momentum_smoothed.rank(axis=1, pct=True)
        industry_signal = (industry_rank > (1 - top_q)).astype(float)

        # ==================================================
        # SERIES
        # ==================================================
        stock_momentum_index = stock_momentum.mean(axis=1).fillna(0)
        industry_strength = industry_momentum_smoothed.mean(axis=1).fillna(0)

        momentum_ma = industry_strength.rolling(industry_window).mean().fillna(0)

        rebalance_events = (stock_signal.sum(axis=1) > 0).astype(float)

        portfolio_momentum = (1 + stock_momentum_index).cumprod()

        benchmark = None
        if "SPY" in prices.columns:
            benchmark = (1 + prices["SPY"].pct_change().fillna(0)).cumprod()

        # ==================================================
        #  PRINT BEFORE CHARTDATA (YOUR REQUEST)
        # ==================================================
        #print("\n[DEBUG BEFORE CHARTDATA]")
        #print("portfolio_momentum:", portfolio_momentum.index[:3])
        #print("industry_strength:", industry_strength.index[:3])
        #print("momentum_ma:", momentum_ma.index[:3])
        #print("stock_momentum_index:", stock_momentum_index.index[:3])

        # ==================================================
        # CHARTDATA (CRITICAL FIX - NO INDEX INFERENCE)
        # ==================================================
        chartdata = pd.DataFrame(index=prices.index)

        chartdata["portfolio_momentum"] = portfolio_momentum
        chartdata["industry_strength"] = industry_strength
        chartdata["momentum_ma"] = momentum_ma
        chartdata["stock_momentum"] = stock_momentum_index
        chartdata["benchmark"] = benchmark
        chartdata["rebalance_events"] = rebalance_events

        chartdata = chartdata.replace([np.inf, -np.inf], np.nan).fillna(0)

        # ==================================================
        # FINAL DEBUG
        # ==================================================
        #print("\n[DEBUG FINAL chartdata]")
        #print(chartdata.index[:5])
        #print("dtype:", chartdata.index.dtype)

        # ==================================================
        # CHART BUILD
        # ==================================================
        chart = self.build_chart(
            chartdata=chartdata,
            series=cfg.get("chart", {}).get("series"),
            title=cfg.get("title"),
            charttype=cfg.get("chart", {}).get("type"),
            chartmode=cfg.get("chart", {}).get("mode")
        )

        return StrategyResult(
            name="IndustryMomentumStrategy",
            data=self.data,
            metrics={
                "formation": formation,
                "industry_window": industry_window,
                "assets": len(prices.columns)
            },
            signals={
                "stock_signal": stock_signal,
                "industry_signal": industry_signal
            },
            chart=chart
        )