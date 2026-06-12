import numpy as np
import pandas as pd
import requests

# ==========================================
# ALPHA LAYER
# ==========================================

from alpha.ts import TSAlpha
from alpha.xs import XSAlpha
from alpha.decomposition import Decompose
from alpha.beta_regression import AlphaBeta

# ==========================================
# RISK LAYER
# ==========================================

from risk.metrics import Risk
from risk.decorrelate import Decorrelate
from risk.wiggle import Wiggle
from risk.cvar import CVaR

# ==========================================
# TRANSFORMS
# ==========================================

from stats.transforms import Transform

# ==========================================
# PORTFOLIO
# ==========================================

from portfolio.construction import Portfolio
from portfolio.risk_parity import RiskParity
from portfolio.kelly import Kelly
from portfolio.entropy import Entropy

# ==========================================
# EXECUTION
# ==========================================

from execution.slippage import Slippage
from execution.impact import Impact
from execution.liquidity import Liquidity

# ==========================================
# REGIME + INTEL
# ==========================================

from regime.detection import Regime
from alpha.ic import IC

# ==========================================
# FINAL OUTPUT ASSEMBLER
# ==========================================


class SymbolAssembler:

    def assemble_symbol_report(
        self,

        symbol,
        price,
        ret,
        volume,

        alpha_ts,
        alpha_xs,
        alpha_pure,
        beta,
        residual,

        volatility,
        sharpe,
        drawdown,
        cvar,
        decorrelation,
        wiggle,

        rank,
        zscore,
        winsor,
        tanh,
        detrend,

        weight,
        risk_parity,
        kelly,
        entropy,

        regime,
        liquidity_adj_vol,

        slippage,
        impact,
        turnover,

        ic,
        score,
        signal
    ):

        return {

            # ==================================
            # MARKET
            # ==================================

            ("market","symbol"): symbol,
            ("market","price"): float(price),
            ("market","return"): float(ret),
            ("market","volume"): float(volume),

            # ==================================
            # ALPHA
            # ==================================

            ("alpha","ts"): float(alpha_ts),
            ("alpha","xs"): float(alpha_xs),
            ("alpha","pure"): float(alpha_pure),
            ("alpha","beta"): float(beta),
            ("alpha","residual"): float(residual),

            # ==================================
            # RISK
            # ==================================

            ("risk","volatility"): float(volatility),
            ("risk","sharpe"): float(sharpe),
            ("risk","drawdown"): float(drawdown),
            ("risk","cvar"): float(cvar),
            ("risk","decorrelation"): float(decorrelation),
            ("risk","wiggle"): float(wiggle),

            # ==================================
            # TRANSFORMS
            # ==================================

            ("transform","rank"): float(rank),
            ("transform","zscore"): float(zscore),
            ("transform","winsor"): float(winsor),
            ("transform","tanh"): float(tanh),
            ("transform","detrend"): float(detrend),

            # ==================================
            # PORTFOLIO
            # ==================================

            ("portfolio","weight"): float(weight),
            ("portfolio","risk_parity"): float(risk_parity),
            ("portfolio","kelly"): float(kelly),
            ("portfolio","entropy"): float(entropy),

            # ==================================
            # MARKET STRUCTURE
            # ==================================

            ("market_structure","regime"): regime,
            ("market_structure","liq_adj_vol"): float(liquidity_adj_vol),

            # ==================================
            # EXECUTION
            # ==================================

            ("execution","slippage"): float(slippage),
            ("execution","impact"): float(impact),
            ("execution","turnover"): float(turnover),

            # ==================================
            # INTELLIGENCE
            # ==================================

            ("intel","ic"): float(ic),

            # ==================================
            # DECISION
            # ==================================

            ("decision","score"): float(score),
            ("decision","signal"): signal
        }


# ==========================================
# MAIN ENGINE
# ==========================================

class QuantX:

    """
    ==========================================================
    QUANT XPERT X
    ==========================================================

    Full institutional-style research engine

    Includes:
        - TS alpha
        - XS alpha
        - beta regression
        - risk modeling
        - CVaR
        - decorrelation
        - portfolio construction
        - Kelly sizing
        - execution modeling
        - liquidity adjustments
        - IC tracking
        - deterministic BUY/SELL/HOLD

    OUTPUT:
        MultiIndex tensor report
    ==========================================================
    """

    def __init__(self, symbols):

        #self.symbols = self.getSymbols(6)
        self.symbols = symbols if symbols is not None else self.getSymbols(6)

        if self.symbols is None:
           raise ValueError("getSymbols() returned None")

        # ==================================
        # ALPHA
        # ==================================

        self.ts = TSAlpha()
        self.xs = XSAlpha()
        self.dec = Decompose()
        self.ab = AlphaBeta()

        # ==================================
        # RISK
        # ==================================

        self.risk = Risk()
        self.decorr = Decorrelate()
        self.wiggle = Wiggle()
        self.cvar = CVaR()

        # ==================================
        # TRANSFORMS
        # ==================================

        self.tr = Transform()

        # ==================================
        # PORTFOLIO
        # ==================================

        self.port = Portfolio()
        self.rp = RiskParity()
        self.kelly = Kelly()
        self.entropy = Entropy()

        # ==================================
        # EXECUTION
        # ==================================

        self.slip = Slippage()
        self.impact = Impact()
        self.liq = Liquidity()

        # ==================================
        # INTEL
        # ==================================

        self.regime = Regime()
        self.ic = IC()

        # ==================================
        # ASSEMBLER
        # ==================================

        self.assembler = SymbolAssembler()

    # =========================================================
    # PAD and NAN
    # =========================================================
    def pad_nan( self,x, target_len):
        if x is None:
           return [np.nan] * target_len

        if not isinstance(x, (list, np.ndarray)):
           x = list(x)

        # trim if too long (important)
        if len(x) > target_len:
           x = x[:target_len]

        # pad with NaN (NOT zero)
        if len(x) < target_len:
           x = x + [np.nan] * (target_len - len(x))

        return x

    # ==========================================================
    # MOCK DATA ENGINE
    # ==========================================================

    def data_mock(self):

        prices = pd.DataFrame(
            np.cumprod(
                1 + np.random.randn(252, len(self.symbols))*0.01,
                axis=0
            ) * 100,
            columns=self.symbols
        )

        volume = pd.DataFrame(
            np.random.randint(
                100000,
                5000000,
                size=(252, len(self.symbols))
            ),
            columns=self.symbols
        )
        #print ("data [prices,volume] => ", prices, volume)
        return prices, volume

    def  getSymbols(self,howMany) :
        # use api/v3/ticker/24hr for top list
        url = "https://api.binance.us/api/v3/exchangeInfo"
        info = requests.get(url).json()

        # check for error response
        if "symbols" not in info:
          raise Exception(f"Binance error: {info}")

        tickers = list(dict.fromkeys(
                     s["symbol"].strip().upper()
                     for s in info["symbols"]
                         if s["status"] == "TRADING"
                           and s["quoteAsset"] == "USDT"
        ))[:howMany]
        #print("symbols:", len(tickers))
        return tickers

    def getSPV(self,sym, interval="1m"):
       #url = "https://api.binance.us/api/v3/ticker/24hr"

       BASE = "https://api.binance.us/api/v3/klines"

       if not isinstance(sym, str):
         return [], []

       sym = sym.strip().upper()

       if not sym.isalnum():
         raise ValueError(f"Bad symbol: {sym}")
           
       params = {
        "symbol": sym,
        "interval": interval,
        #"limit": 1  # only latest candle
       }

       responsePV = requests.get(BASE, params=params).json()

       if not isinstance(responsePV, list):
          raise ValueError(responsePV)

       closes = [float(c[4]) for c in responsePV]
       volumes = [float(c[5]) for c in responsePV]

       return closes, volumes
    def data(self):

        prices_dict = {}
        volumes_dict = {}
        valid_symbols = []

        max_len = 0

        for sym in self.symbols:
           try:
              closes, volumes = self.getSPV(sym)

              if len(closes) == 0:
                continue

              prices_dict[sym] = closes
              volumes_dict[sym] = volumes

              valid_symbols.append(sym)

              max_len = max(max_len, len(closes))

           except Exception:
              continue

        self.symbols = valid_symbols   # 🔥 CRITICAL FIX

        for k in prices_dict:
           prices_dict[k] = self.pad_nan(prices_dict[k], max_len)

        for k in volumes_dict:
           volumes_dict[k] = self.pad_nan(volumes_dict[k], max_len)

        P = pd.DataFrame.from_dict(prices_dict, orient="index").T
        V = pd.DataFrame.from_dict(volumes_dict, orient="index").T

        P = P.interpolate(limit=5)
        V = V.interpolate(limit=5)

        return P, V

    def dataLastAfterPrior(self):

        prices_dict = {}
        volumes_dict = {}

        max_len = 0

        for sym in self.symbols:
            try:
               closes, volumes = self.getSPV(sym)

               prices_dict[sym] = closes
               volumes_dict[sym] = volumes

               max_len = max(max_len, len(closes))

            except Exception:
               continue

            # SAFE PAD (NaN ONLY)
            for k in prices_dict:
                prices_dict[k] = self.pad_nan(prices_dict[k], max_len)

            for k in volumes_dict:
                volumes_dict[k] = self.pad_nan(volumes_dict[k], max_len)

            # BUILD DATAFRAME (NO SILENT INDEX SHIFT)
            P = pd.DataFrame.from_dict(prices_dict, orient="index").T
            V = pd.DataFrame.from_dict(volumes_dict, orient="index").T

            # CLEAN
            P = P.replace([np.inf, -np.inf], np.nan)
            V = V.replace([np.inf, -np.inf], np.nan)

            # INTERPOLATE (NOT FILL ZERO)
            P = P.interpolate(limit=5)
            V = V.interpolate(limit=5)

            return P, V
    def dataPrior(self):

       prices_dict = {}
       volumes_dict = {}

       for sym in self.symbols:
         try:
            closes, volumes = self.getSPV(sym)

            prices_dict[sym] = closes
            volumes_dict[sym] = volumes
            #print( " volume=" , volumes, " sym=" , sym)
            #print( "len(price_dict) ", sym, len(prices_dict[sym]))
            #print( "len(volume_dict) ", sym, len(volumes_dict[sym]))
            
            # STEP 1: find max length (your logic style)
            max_len = 0
            for v in prices_dict.values():
               if len(v) > max_len:
                 max_len = len(v)

            # STEP 2: pad to max length
            for k in prices_dict:
                prices_dict[k] += [0] * (max_len - len(prices_dict[k]))

            for k in volumes_dict:
                volumes_dict[k] += [0] * (max_len - len(volumes_dict[k]))
         except Exception as e:
               continue
       # STEP 3: build matrix
       P = pd.DataFrame({k: pd.Series(v) for k, v in prices_dict.items()}).ffill().bfill()
       V = pd.DataFrame({k: pd.Series(v) for k, v in volumes_dict.items()}).ffill().bfill()
       #print("P V", P, V)

       P = P.replace([np.inf, -np.inf], np.nan).dropna()
       V = V.replace([np.inf, -np.inf], np.nan).dropna()
       return P, V

    def data1(self):
       #tickerSym = getSymbols(20);
       prices_dict = {}
       volume_dict = {}
      
       for sym in self.symbols:
         #print("symbol = ", sym)
         prices,volumes = self.getSPV(sym)
         prices_dict[sym] = prices
         volume_dict[sym] = volumes

       P = pd.DataFrame(prices_dict).ffill().bfill()
       V = pd.DataFrame(volume_dict).ffill().bfill()

       return P, V


#
#    # ==========================================================
#    # SYMBOL ANALYSIS - market
#    #  Price Formula = Pt​  --> raw spot price
#    #  Return Formula = rt​=Pt​−Pt−1/Pt−1​ ​​or rt​=ln(Pt​/Pt−1​) They are not consistent with “HIGH_VOL regime” unless this is ultra-high frequency data, If daily, these are too small for crypto, If       #intraday, fine—but then volatility must be time-scaled consistently (see risk section)
#          #  Volume Formula = Vt
#    # SYMBOL ANALYSIS - alpha
#    #  Time-series alpha (TS) Formula αts =f(xt)→return prediction from pa
#    #  Cross-sectional alpha (XS) Formula  αxs =f(relative ranking across assets) -->Pure alpha Often: αpure=αts −market factor exposure or regression residual form: rt=βrm+α+ϵ
#    #  Purity (pure) Formula  αpure=αts−β⋅rm  [Note Pure not equals TS or XS or regression residual, it should not equal TS exactly unless:  beta correction is zero, OR xs is ignored, 
#          #  Beta Formula = β=Cov(ri,rm) / Var(rm)
#          #  Residual [noise] Formula ϵ=ri−βrm If beta is non-trivial, residual ≠ 0 almost never. [Note residual 0 is not good]
#    # SYMBOL ANALYSIS - risk
#    #  Volatility Formula σ= sqrt(E[(r−μ)**2])
#    #  Sharpe Formula Sharpe=E[r]​/σ  --> computed on strategy returns, not market returns BUT labeled under market block OR: returns are not aligned (mean not computed over same window as #volatility)
#    #  Drawdown Formula DDt​= (Pt​−max(P0:t​))​/max(P0:t​) extremely tiny Sharpe magnitude → suggests wrong aggregation horizon mismatch
#    #  CVar Formula CVaRα​=E[r∣r≤VaRα​] fat-tailed small-return distribution
#    #  Decorrelation Formula = 1−∣ρ∣ or 1/(1−ρ)​ --> decorrelation should be asset-specific unless:computed at portfolio level and broadcasted incorrectly
#    # SYMBOL ANALYSIS - transform
#    #  zScore Formula z=(x−μ​)/σ
#    #  Winsor Formula ? winsorized values should be bounded  --> transform pipeline bypassed or not applied ???
#    #  Tanh Formula  tanh(x)∈(−1,1)  --> transform pipeline bypassed or not applied???
#    #  Detrend Formula xdetrended​=x−x^trend​  --> wrong sign trend regression OR mixing assets in detrending step
#    # SYMBOL ANALYSIS - portfolio 
#    #  Weight Formula ?  possible -1, 1 --> devise a real optimization output and not hard override
#    #  Kelly criterion Formula [ note f∗ here must be f* notated superscript]  f*  =μ​/σ**2 --> should be extremely negative,different return unit (maybe normalized signal returns, not market #returns)
#    #  Entropy Formula H=−∑pi log pi ​--> 0.0 is degenerate distribution or placeholder
#    # SYMBOL ANALYSIS - Intel Decision
#    #  IC Formula IC=corr(r^,r)  note here ^ is cap on r on first parameter of Correlaton
#    #  Score Formula score=w1​⋅α+w2​⋅IC+w3​⋅z --> score magnitudes do NOT match alpha scale, suggests nonlinear compression or missing normalization
#    # ========================================================== 
#
    def analyze_symbol(self, P, V, R, s):
        EPS = 1e-12

        R = R.fillna(0.0)
        V = V.fillna(0.0)

        R = R.clip(-1, 1)
        V = V.clip(lower=EPS)
        
        price = P[s]
        volume = V[s]
        ret = R[s]
        #cross sectional mean original code 
        #market = R.mean(axis=1)

        # pre check market and return
        #mask = market.notna() & ret.notna()

        #x = market[mask]
        #y = ret[mask]
        # market Formula = Rs​=α+β⋅market+ϵ
        market = R.mean(axis=1).rolling(10).mean()
        # ==================================
        # ALPHA
        # ==================================

        ts = self.ts.momentum(ret)

        xs = self.xs.cross_section(R)

        pure = self.dec.purify(ts, xs)

        # beta Formula = Ri​=α+βM+ϵ
        alpha, beta, residual = self.ab.fit(
            ret,
            market
        )

        # ==================================
        # RISK
        # ==================================

        vol = self.risk.volatility(ret)

        sharpe = self.risk.sharpe(ret)

        dd = self.risk.drawdown(price)

        corr = self.risk.correlation(R)

        decor = self.decorr.score(corr)

        wiggle = self.wiggle.compute(R)

        cvar = self.cvar.compute(ret)

        # ==================================
        # TRANSFORMS
        # ==================================

        rank = self.tr.rank(ret)

        z = self.tr.zscore(ret)

        win = self.tr.winsorize(ret)

        tanh = self.tr.tanh(ret)

        detrend = self.tr.detrend(ret)

        # ==================================
        # PORTFOLIO
        # ==================================

        weight = self.port.weights(
            ret.mean(),
            vol.mean()
        )

        rp = self.rp.weights(vol.mean())

        kelly = self.kelly.size(
            ret.mean(),
            np.var(ret)
        )

        entropy = self.entropy.compute(
            np.array([weight])
        )

        # ==================================
        # EXECUTION
        # ==================================

        slip = self.slip.model(
            vol.mean()
        )

        impact = self.impact.model(
            weight
        )

        liq_adj = self.liq.adjust(
            vol.mean(),
            volume.mean()
        )

        turnover = np.abs(weight) * 0.01

        # ==================================
        # REGIME
        # ==================================

        regime = self.regime.detect(ret)

        # ==================================
        # INTEL
        # ==================================

        future_returns = ret.shift(-1).fillna(0)

        ic = self.ic.compute(
            z.fillna(0),
            future_returns
        )

        # ==================================
        # FINAL SCORE
        # ==================================

        score = (
            alpha
            + sharpe
            + ic
            - cvar
            - abs(dd.iloc[-1])
        )

        # ==================================
        # SIGNAL
        # ==================================

        if score > 0.5:
            signal = "BUY"

        elif score < -0.5:
            signal = "SELL"

        else:
            signal = "HOLD"

        # ==================================
        # ASSEMBLE FINAL REPORT
        # ==================================

        return self.assembler.assemble_symbol_report(

            symbol=s,

            price=price.iloc[-1],
            ret=ret.iloc[-1],
            volume=volume.iloc[-1],

            alpha_ts=ts.iloc[-1],
            alpha_xs=xs[s].iloc[-1],
            alpha_pure=pure.iloc[-1],

            beta=beta,
            residual=np.mean(residual),

            volatility=vol.iloc[-1],
            sharpe=sharpe,
            drawdown=dd.iloc[-1],
            cvar=cvar,
            decorrelation=decor,
            wiggle=wiggle,

            rank=rank.iloc[-1],
            zscore=z.iloc[-1],
            winsor=win.iloc[-1],
            tanh=tanh.iloc[-1],
            detrend=detrend.iloc[-1],

            weight=weight,
            risk_parity=rp,
            kelly=kelly,
            entropy=entropy,

            regime=regime,
            liquidity_adj_vol=liq_adj,

            slippage=slip,
            impact=impact,
            turnover=turnover,

            ic=ic,

            score=score,
            signal=signal
        )

    # ==========================================================
    # RUN ENGINE
    # ==========================================================

    def run(self):

        P, V = self.data()

        R = P.pct_change().fillna(0)

        rows = []

        for s in self.symbols:

            row = self.analyze_symbol(
                P,
                V,
                R,
                s
            )

            rows.append(row)

        df = pd.DataFrame(rows)

        df.columns = pd.MultiIndex.from_tuples(
            df.columns
        )

        return df