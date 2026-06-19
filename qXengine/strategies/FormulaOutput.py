import numpy as np
import pandas as pd


class FormulaOutput:

    def __init__(self, data: pd.DataFrame):

        self.data = data
        self.cache = {}

        self.catalog = self.FORMULA_CATALOG

    # =====================================================
    # CALL INTERFACE
    # =====================================================

    def __call__(self, name: str):
        return self.get(name)

    def get(self, name: str):

        if name in self.cache:
            return self.cache[name]

        if name not in self.catalog:
            raise ValueError(f"Unknown formula: {name}")

        category, ftype, formula, depends, dtype = self.catalog[name]

        # resolve dependencies first
        resolved = {d: self.get(d) for d in depends}

        value = self._compute(name, resolved)

        self.cache[name] = value
        return value

    # =====================================================
    # CORE COMPUTATION ENGINE
    # =====================================================

    def _compute(self, name, dep):

        r = self.data["close"].pct_change().replace([np.inf, -np.inf], 0).fillna(0)

        # -------------------------
        # BASICS
        # -------------------------

        if name == "close":
            return self.data

        if name == "returns":
            return r

        if name == "mu":
            return r.mean()

        if name == "sigma":
            return r.std()

        if name == "var":
            return r.var()

        if name == "std":
            return np.sqrt(dep["var"])

        if name == "t_stat":
            return np.sqrt(len(self.data)) * dep["mu"] / (dep["std"] + 1e-12)

        # -------------------------
        # TS
        # -------------------------

        if "ts_mom" in name:
            return r.rolling(5).mean()

        # -------------------------
        # XS
        # -------------------------

        if name == "xs":
            return r.sub(r.mean(axis=1), axis=0)

        if name == "xs_z":
            xs = dep["xs"]
            return (xs - xs.mean()) / (xs.std() + 1e-12)

        # -------------------------
        # PURE
        # -------------------------

        if name == "pure":
            return dep["ts_mom"] + dep["xs"]

        if name == "pure_ir":
            p = dep["pure"]
            return p.mean() / (p.std() + 1e-12)

        # -------------------------
        # RISK
        # -------------------------

        if name == "volatility":
            return r.rolling(10).std()

        if name == "sharpe":
            return dep["mu"] / (dep["sigma"] + 1e-12)

        if name == "drawdown":
            eq = (1 + r).cumprod()
            return eq / eq.cummax() - 1

        if name == "cvar":
            q = r.quantile(0.05)
            return r[r <= q].mean()

        if name == "corr":
            return r.corr()

        if name == "decorrelation":
            return 1 / (np.mean(np.abs(dep["corr"])) + 1e-12)

        if name == "wiggle":
            return dep["corr"].diff()

        # -------------------------
        # TRANSFORM
        # -------------------------

        if name == "zscore":
            return (r - dep["mu"]) / (dep["sigma"] + 1e-12)

        if name == "winsor":
            return r.clip(r.quantile(0.05), r.quantile(0.95))

        if name == "tanh":
            return np.tanh(r)

        if name == "detrend":
            return r - r.rolling(5).mean()

        if name == "rank":
            return r.rank()

        # -------------------------
        # PORTFOLIO
        # -------------------------

        if name == "weight":
            return dep["mu"] / (dep["sigma"] + 1e-12)

        if name == "kelly":
            return dep["mu"] / (dep["var"] + 1e-12)

        if name == "risk_parity":
            return 1 / (dep["sigma"] + 1e-12)

        if name == "entropy":
            w = np.abs(dep["weight"])
            p = w / (w.sum() + 1e-12)
            return -np.sum(p * np.log(p + 1e-12))

        # -------------------------
        # EXECUTION / MARKET / INTEL / DECISION
        # -------------------------

        if name == "slippage":
            return 0.0001 * np.sqrt(dep["volatility"].mean().mean())

        if name == "impact":
            return np.sqrt(np.abs(dep["weight"]))

        if name == "turnover":
            return dep["weight"].diff()

        if name == "liq_adj_vol":
            return dep["volatility"] / (self.data["volume"] + 1e-12)

        if name == "regime":
            v = dep["volatility"].mean().mean()
            return "HIGH" if v > 0 else "NORMAL"

        if name == "beta":
            return r.cov() / (r.var() + 1e-12)

        if name == "ic":
            return np.corrcoef(dep["zscore"].values.flatten(), r.shift(-1).fillna(0).values.flatten())[0, 1]

        if name == "score":
            return dep["sharpe"] + dep["ic"] - abs(dep["cvar_05"]) - abs(dep["drawdown"].iloc[-1].mean())

        if name == "signal":
            s = dep["score"]
            return "BUY" if s > 0.5 else "SELL" if s < -0.5 else "HOLD"

        if name == "spread":
            return r.iloc[:, 0] - r.iloc[:, 1]

        if name == "beta_reg":
            return np.cov(r.T) / (np.var(r) + 1e-12)

        if name == "meta_alpha":
            return dep["signal"]

        raise ValueError(f"No compute rule for {name}")

    FORMULA_CATALOG = {

        # =====================================================
        # 0 BASICS (ATOMIC LAYER)
        # =====================================================

        "close":        ("basics","close","close",[], "DataFrame"),
        "returns":      ("basics","returns","close.pct_change()",["close"], "DataFrame"),

        "mu":           ("basics","mu","mean(returns)",["returns"], "Series"),
        "sigma":        ("basics","sigma","std(returns)",["returns"], "Series"),
        "var":          ("basics","var","var(returns)",["returns"], "Series"),
        "std":          ("basics","std","sqrt(var)",["var"], "Series"),

        "t_stat":       ("basics","t_stat","sqrt(n)*mu/std",["mu","std"], "Series"),

        # =====================================================
        # 1 ALPHA_TS
        # =====================================================

        "ts_mom":       ("alpha_ts","ts_mom","rolling_mean(returns,5)",["returns"], "DataFrame"),
        "ts_mom_10":    ("alpha_ts","ts_mom_10","rolling_mean(returns,10)",["returns"], "DataFrame"),
        "ts_vol_adj":   ("alpha_ts","ts_vol_adj","ts_mom/sigma",["ts_mom","sigma"], "DataFrame"),

        # =====================================================
        # 2 ALPHA_XS
        # =====================================================

        "xs":           ("alpha_xs","xs","returns-mean(returns,axis=1)",["returns"], "DataFrame"),
        "xs_z":         ("alpha_xs","xs_z","(xs-mean(xs))/std(xs)",["xs"], "DataFrame"),

        # =====================================================
        # 3 ALPHA_PURE
        # =====================================================

        "pure":         ("alpha_pure","pure","ts_mom+xs",["ts_mom","xs"], "DataFrame"),
        "pure_ir":      ("alpha_pure","pure_ir","mean(pure)/std(pure)",["pure"], "Series"),

        # =====================================================
        # 4 RISK
        # =====================================================

        "volatility":   ("risk","volatility","rolling_std(returns,10)",["returns"], "DataFrame"),
        "sharpe":       ("risk","sharpe","mu/sigma",["mu","sigma"], "Series"),

        "drawdown":     ("risk","drawdown","equity/peak-1",["returns"], "DataFrame"),

        "var_05":       ("risk","var_05","quantile(returns,0.05)",["returns"], "Series"),
        "cvar":         ("risk","cvar","mean(returns|returns<var_05)",["returns","var_05"], "Series"),

        "corr":         ("risk","corr","corr(returns)",["returns"], "DataFrame"),
        "decorrelation":("risk","decorrelation","1/mean(abs(corr))",["corr"], "float"),
        "wiggle":       ("risk","wiggle","corr.diff()",["corr"], "DataFrame"),

        # =====================================================
        # 5 TRANSFORM
        # =====================================================

        "zscore":       ("transform","zscore","(returns-mu)/sigma",["returns","mu","sigma"], "DataFrame"),
        "winsor":       ("transform","winsor","clip(returns,5%,95%)",["returns"], "DataFrame"),
        "tanh":         ("transform","tanh","tanh(returns)",["returns"], "DataFrame"),
        "detrend":      ("transform","detrend","returns-rolling_mean",["returns"], "DataFrame"),
        "rank":         ("transform","rank","rank(returns)",["returns"], "DataFrame"),
     
        # =====================================================
        # 6 PORTFOLIO
        # =====================================================

        "weight":       ("portfolio","weight","mu/sigma",["mu","sigma"], "Series"),
        "kelly":        ("portfolio","kelly","mu/var",["mu","var"], "Series"),
  
        "risk_parity":  ("portfolio","risk_parity","1/sigma normalized",["sigma"], "Series"),
        "entropy":      ("portfolio","entropy","-sum(p log p)",["weight"], "float"),

        # =====================================================
        # 7 EXECUTION
        # =====================================================

        "slippage":     ("execution","slippage","k*sqrt(volatility)",["volatility"], "float"),
        "impact":       ("execution","impact","k*sqrt(weight)",["weight"], "Series"),
        "turnover":     ("execution","turnover","abs(diff(weight))",["weight"], "Series"),
        "liq_adj_vol":  ("execution","liq_adj_vol","volatility/volume",["volatility"], "Series"),
 
        # =====================================================
        # 8 MARKET STRUCTURE
        # =====================================================

        "regime":       ("market_structure","regime","vol percentile",["volatility"], "str"),
        "beta":         ("market_structure","beta","cov(asset,mkt)/var(mkt)",["returns"], "Series"),

        # =====================================================
        # 9 INTEL
        # =====================================================

        "ic":           ("intel","ic","corr(zscore,fwd_returns)",["zscore"], "float"),

        # =====================================================
        # 10 DECISION
        # =====================================================

        "score":        ("decision","score","sharpe+ic-cvar-drawdown",["sharpe","ic","cvar_05","drawdown"], "float"),
        "signal":       ("decision","signal","threshold(score)",["score"], "str"),

        # =====================================================
        # 11 PAIRS
        # =====================================================

        "spread":       ("pairs","spread","A-B",["returns"], "Series"),
        "beta_reg":     ("pairs","beta_reg","cov/var",["returns"], "Series"),

        # =====================================================
        # 12 META
        # =====================================================

        "meta_alpha":   ("meta","meta_alpha","weighted(signal,ic,sharpe)",["signal","ic","sharpe"], "float"),

     }