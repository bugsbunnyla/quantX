# ===============================================================
# FormulaOutput class defines the BaseStrategy structured data
# Date: 2026/06/22 
# Author : bugsbunnyla
# Comment : Core formulas for the QuantXpert structure data model
# ===============================================================
import numpy as np
import pandas as pd
import ast

class FormulaOutput:
    STATIC_STORE = []

    def __init__(self, data: dict):
        self.data = data
        self.symbols = list(data.keys())

        # CRITICAL FIX: compute returns once and inject into data layer
        self.ret = self._returns()
        self._inject_returns()

        self.outputs = self.assemble()
        FormulaOutput.STATIC_STORE.append(self.outputs)

    # =====================================================
    # SAFE CAST
    # =====================================================
    def _f(self, x, default=0.0):
        try:
            if isinstance(x, pd.Series):
                x = x.replace([np.inf, -np.inf], np.nan).fillna(0)
                return float(x.iloc[0]) if len(x) else default
            return float(x)
        except:
            return default

    # =====================================================
    # RETURNS (CORE PRIMITIVE)
    # =====================================================
    def _returns(self):
        closes = {
            s: pd.to_numeric(self.data[s]["close"], errors="coerce")
            for s in self.symbols
        }

        r = pd.DataFrame(closes).pct_change()

        return r.replace([np.inf, -np.inf], np.nan).fillna(0)

    #  CRITICAL FIX: expose ret into original dataset
    def _inject_returns(self):
        for s in self.symbols:
            self.data[s]["ret"] = self.ret[s]

    # =====================================================
    # CORE METRICS
    # =====================================================
    def _volatility(self, r):
        return r.std().replace(0, 1e-9)

    def _sharpe(self, r):
        return np.sqrt(252) * r.mean() / self._volatility(r)

    def _cvar(self, r):
        q = r.quantile(0.05)
        return pd.Series({
            c: r[c][r[c] <= q[c]].mean() if len(r[c]) else 0
            for c in r.columns
        }).fillna(0)

    def _ic(self, r):
        fwd = r.shift(-1).fillna(0)
        return pd.Series({
            c: r[c].corr(fwd[c]) if r[c].std() > 0 else 0
            for c in r.columns
        }).fillna(0)

    def _corr(self, r, s, mkt_ret=None):

      if mkt_ret is None:
        mkt_ret = r.mean(axis=1)

      series = r[s]

      if series.std() == 0 or mkt_ret.std() == 0:
        return 0.0

      return float(series.corr(mkt_ret))

    # =====================================================
    # ALPHA
    # =====================================================
    def _alpha_ts(self, r):
        return r.rolling(10, min_periods=1).mean().iloc[-1].fillna(0)

    def _alpha_xs(self, r):
        return (r.sub(r.mean(axis=1), axis=0)).iloc[-1].fillna(0)

    # =====================================================
    # TRANSFORMS
    # =====================================================
    def _rank(self, r):
        return r.iloc[-1].rank().fillna(0)

    def _zscore(self, r):
        return ((r - r.mean()) / (r.std() + 1e-9)).iloc[-1].fillna(0)

    def _winsor(self, r):
        return r.clip(r.quantile(0.05), r.quantile(0.95), axis=1).iloc[-1].fillna(0)

    def _tanh(self, r):
        return pd.Series(np.tanh(r.iloc[-1]), index=r.columns).fillna(0)

    def _detrend(self, r):
        return (r - r.rolling(20, min_periods=1).mean()).iloc[-1].fillna(0)

    # =====================================================
    # PORTFOLIO LAYER
    # =====================================================
    def _weight(self, r):
        return pd.Series(1 / len(r.columns), index=r.columns)

    def _risk_parity(self, r):
        inv = 1 / (r.std() + 1e-9)
        return inv / inv.sum()

    def _kelly(self, r):
        return r.mean() / (r.var() + 1e-9)

    def _inv_vol(self, r):
        v = r.std() + 1e-9
        return 1 / v / (1 / v).sum()

    def _mvo(self, r):
        mu = r.mean().values
        cov = r.cov().values
        inv = np.linalg.pinv(cov + np.eye(len(cov)) * 1e-6)
        w = inv @ mu
        w = np.maximum(w, 0)
        return pd.Series(w / (w.sum() + 1e-9), index=r.columns)

    def _entropy(self, w):
        w = np.clip(w, 1e-9, 1)
        return float(-np.sum(w * np.log(w)))

    # =====================================================
    # MARKET STRUCTURE
    # =====================================================
    def detect_regime(self, r):
        vol = r.std()
        if vol.mean() > np.percentile(vol, 75):
            return "HIGH_VOL"
        elif vol.mean() < np.percentile(vol, 25):
            return "LOW_VOL"
        return "NORMAL"

    def _liq_adj_vol(self, vol, volume):
        return vol / (volume + 1e-9)

    # =====================================================
    # BETA / RESIDUAL
    # =====================================================
    def _beta(self, r, benchmark):
        cov = r.cov().iloc[:, 0]
        return cov / (benchmark.var() + 1e-9)

    def _residual(self, r, benchmark):
        return r.sub(benchmark, axis=0).mean()

    
    # =====================================================
    # MAIN ASSEMBLY
    # =====================================================
    def assemble(self):

        r = self.ret  # use injected canonical returns

        benchmark = r.mean(axis=1)

        weight = self._weight(r)
        risk_parity = self._risk_parity(r)

        volatility = self._volatility(r)
        sharpe = self._sharpe(r)
        cvar = self._cvar(r)

        ic = self._ic(r)

        alpha_ts = self._alpha_ts(r)
        alpha_xs = self._alpha_xs(r)

        rank = self._rank(r)
        zscore = self._zscore(r)
        winsor = self._winsor(r)
        tanh = self._tanh(r)
        detrend = self._detrend(r)

        slippage = r.std().mean()
        impact = slippage * 0.5
        turnover = r.diff().abs().mean().mean()

        future = r.shift(-1)
        hit_ratio = (np.sign(r) == np.sign(future)).mean().fillna(0)

        reports = {}

        for s in self.symbols:

            price = self._f(self.data[s]["close"].iloc[-1])
            volume = self._f(self.data[s]["volume"].iloc[-1])

            ats = self._f(alpha_ts[s])
            axs = self._f(alpha_xs[s])
            apure = ats + axs

            beta = self._f(self._beta(r, benchmark)[s])
            residual = self._f(self._residual(r, benchmark)[s])

            vol = self._f(volatility[s])
            regime = self.detect_regime(r[s])
            liq_adj = self._liq_adj_vol(vol, volume)

            rp = self._f(risk_parity[s])
            kv = self._f(self._kelly(r)[s])
            iv = self._f(self._inv_vol(r)[s])

            portfolio_signal = 0.4 * rp + 0.3 * kv + 0.3 * iv

            score = (
                0.4 * apure +
                0.2 * self._f(sharpe[s]) +
                0.2 * self._f(ic[s]) -
                0.1 * self._f(cvar[s]) +
                0.3 * portfolio_signal
            )

            signal = "BUY" if score > 0 else "SELL"

            reports[s] = {

                # ================= CORE MARKET =================
                ("market", "symbol"): s,
                ("market", "price"): price,
                ("market", "volume"): volume,

                # FIXED: REQUIRED CORE PRIMITIVE
                ("market", "ret"): self._f(self.ret[s].iloc[-1]),

                # ================= ALPHA =================
                ("alpha", "ts"): ats,
                ("alpha", "xs"): axs,
                ("alpha", "pure"): apure,
                ("alpha", "beta"): beta,
                ("alpha", "residual"): residual,

                # ================= RISK =================
                ("risk", "volatility"): vol,
                ("risk", "sharpe"): self._f(sharpe[s]),
                ("risk", "cvar"): self._f(cvar[s]),
                ("risk", "drawdown"): 0.0,

                # ================= TRANSFORM =================
                ("transform", "rank"): self._f(rank[s]),
                ("transform", "zscore"): self._f(zscore[s]),
                ("transform", "winsor"): self._f(winsor[s]),
                ("transform", "tanh"): self._f(tanh[s]),
                ("transform", "detrend"): self._f(detrend[s]),

                # ================= PORTFOLIO =================
                ("portfolio", "weight"): self._f(weight[s]),
                ("portfolio", "risk_parity"): rp,
                ("portfolio", "kelly"): kv,
                ("portfolio", "inv_vol"): iv,
                ("portfolio", "mvo"): self._f(self._mvo(r)[s]),
                ("portfolio", "entropy"): self._entropy(weight.values),

                # ================= EXECUTION =================
                ("execution", "slippage"): slippage,
                ("execution", "impact"): impact,
                ("execution", "turnover"): turnover,

                # ================= INTEL =================
                ("intel", "ic"): self._f(ic[s]),

                # ================= BASIC =================
                ("basic", "r_squared"): float(r.corr().mean().mean()),
                ("basic", "tstat"): float(r.mean().mean() / (r.std().mean() + 1e-9)),
                ("basic", "hit_ratio"): self._f(hit_ratio[s]),                
                ("basic", "corr"):  self._corr(r, s, r.mean(axis=1)),

                # ================= STRUCTURE =================
                ("market_structure", "liq_adj_vol"): liq_adj,
                ("market_structure", "regime"): regime,

                # ================= DECISION =================
                ("decision", "score"): score,
                ("decision", "signal"): signal,
            }

        df = pd.DataFrame(reports)
        df.index = pd.MultiIndex.from_tuples(df.index, names=["category", "metric"])
        return df

    def get(self, key):
        return self.outputs.get(key, None)


    # =====================================================
    # FORMULA CONFIG (UNCHANGED)
    # =====================================================

    FORMULA_CONFIG = {

      # =========================
      # MARKET (RAW DATA ONLY)
      # =========================
      "symbol":     {"category":"market","type":"symbol","formula":"raw_symbol","depends":[],"dtype":"Series","return_result":"symbol"},
      "price":      {"category":"market","type":"price","formula":"price","depends":[],"dtype":"Series","return_result":"price"},
      "volume":     {"category":"market","type":"volume","formula":"volume","depends":[],"dtype":"Series","return_result":"volume"},
      "mkt_price":  {"category":"market","type":"mkt_price","formula":"benchmark_price","depends":[],"dtype":"Series","return_result":"mkt_price"},
      "mkt_ret":    {"category":"market","type":"mkt_ret","formula":"pct_change(mkt_price)","depends":["mkt_price"],"dtype":"Series","return_result":"mkt_ret"},

      # =========================
      # BASIC (ATOMIC + PURE DERIVATIONS)
      # =========================

      # returns
      "ret":        {"category":"basic","type":"ret","formula":"pct_change(price)","depends":["price"],"dtype":"Series","return_result":"ret"},
      "log_ret":    {"category":"basic","type":"log_ret","formula":"log1p(ret)","depends":["ret"],"dtype":"Series","return_result":"log_ret"},

      # simple stats
      "mean_ret":   {"category":"basic","type":"mean","formula":"mean(ret)","depends":["ret"],"dtype":"float","return_result":"mean_ret"},
      "std_ret":    {"category":"basic","type":"std","formula":"std(ret)","depends":["ret"],"dtype":"float","return_result":"std_ret"},
      "var_ret":    {"category":"basic","type":"var","formula":"var(ret)","depends":["ret","mean_ret"],"dtype":"float","return_result":"var_ret"},

      "cov_rm":     {"category":"basic","type":"cov_rm","formula":"cov(ret,mkt_ret)","depends":["ret","mkt_ret","mean_ret"],"dtype":"float","return_result":"cov_rm"},
      "corr_rm":    {"category":"basic","type":"corr_rm","formula":"corr(ret,mkt_ret)","depends":["cov_rm","std_ret"],"dtype":"float","return_result":"corr_rm"},

      # rolling primitives
      "rolling_mean": {"category":"basic","type":"rolling_mean","formula":"mean(window)","depends":["ret"],"dtype":"Series","return_result":"rolling_mean"},
      "rolling_std":  {"category":"basic","type":"rolling_std","formula":"std(window)","depends":["ret"],"dtype":"Series","return_result":"rolling_std"},
      "rolling_var":  {"category":"basic","type":"rolling_var","formula":"var(window)","depends":["ret"],"dtype":"Series","return_result":"rolling_var"},

      # cross-section primitives
      "cs_mean":   {"category":"basic","type":"cs_mean","formula":"mean(axis=1)","depends":["ret"],"dtype":"DataFrame","return_result":"cs_mean"},
      "cs_std":    {"category":"basic","type":"cs_std","formula":"std(axis=1)","depends":["ret"],"dtype":"DataFrame","return_result":"cs_std"},
      "cs_rank":   {"category":"basic","type":"cs_rank","formula":"rank(axis=1)","depends":["ret"],"dtype":"DataFrame","return_result":"cs_rank"},
      "cs_zscore": {"category":"basic","type":"cs_zscore","formula":"(x-cs_mean)/cs_std","depends":["ret","cs_mean","cs_std"],"dtype":"DataFrame","return_result":"cs_zscore"},

      # regression primitives
      "ols_beta":   {"category":"basic","type":"ols_beta","formula":"cov/var","depends":["cov_rm","var_ret"],"dtype":"float","return_result":"ols_beta"},
      "ols_alpha":  {"category":"basic","type":"ols_alpha","formula":"mean(ret)-beta*mean(mkt_ret)","depends":["mean_ret","ols_beta"],"dtype":"float","return_result":"ols_alpha"},
      "fitted":     {"category":"basic","type":"fitted","formula":"alpha+beta*mkt_ret","depends":["ols_alpha","ols_beta","mkt_ret"],"dtype":"Series","return_result":"fitted"},
      "residual":   {"category":"basic","type":"residual","formula":"ret-fitted","depends":["ret","fitted"],"dtype":"Series","return_result":"residual"},
      "sse":        {"category":"basic","type":"sse","formula":"sum(residual^2)","depends":["residual"],"dtype":"float","return_result":"sse"},
      "sst":        {"category":"basic","type":"sst","formula":"sum((ret-mean)^2)","depends":["ret","mean_ret"],"dtype":"float","return_result":"sst"},

      # transforms
      "rank":     {"category":"basic","type":"rank","formula":"rank(x)","depends":["ret"],"dtype":"DataFrame","return_result":"rank"},
      "zscore":   {"category":"basic","type":"zscore","formula":"(x-mean)/std","depends":["ret","mean_ret","std_ret"],"dtype":"DataFrame","return_result":"zscore"},
      "winsor":   {"category":"basic","type":"winsor","formula":"clip(5%,95%)","depends":["ret"],"dtype":"DataFrame","return_result":"winsor"},
      "tanh":     {"category":"basic","type":"tanh","formula":"tanh(x)","depends":["ret"],"dtype":"DataFrame","return_result":"tanh"},
      "detrend":  {"category":"basic","type":"detrend","formula":"x-rolling_mean(x)","depends":["ret","rolling_mean"],"dtype":"DataFrame","return_result":"detrend"},

      # time ops
      "diff":     {"category":"basic","type":"diff","formula":"x_t-x_{t-1}","depends":["ret"],"dtype":"Series","return_result":"diff"},
      "lag":      {"category":"basic","type":"lag","formula":"shift(x)","depends":["ret"],"dtype":"Series","return_result":"lag"},
      "cumprod":  {"category":"basic","type":"cumprod","formula":"product(1+x)","depends":["ret"],"dtype":"Series","return_result":"cumprod"},

      # portfolio primitives
      "weight":     {"category":"basic","type":"weight","formula":"raw_weight","depends":[],"dtype":"Series","return_result":"weight"},
      "norm_weight":{"category":"basic","type":"norm_weight","formula":"w/sum(w)","depends":["weight"],"dtype":"Series","return_result":"norm_weight"},

      # =========================
      # RISK (DERIVED LAYER)
      # =========================
      "volatility": {"category":"risk","type":"volatility","formula":"rolling_std(ret)","depends":["rolling_std"],"dtype":"float","return_result":"volatility"},
      "sharpe":     {"category":"risk","type":"sharpe","formula":"mean/std * np.sqrt(252)","depends":["mean_ret","std_ret"],"dtype":"float","return_result":"sharpe"},

      "equity":     {"category":"risk","type":"equity","formula":"cumprod(1+ret)","depends":["ret"],"dtype":"Series","return_result":"equity"},
      "peak":       {"category":"risk","type":"peak","formula":"cummax(equity)","depends":["equity"],"dtype":"Series","return_result":"peak"},
      "drawdown":   {"category":"risk","type":"drawdown","formula":"equity/peak-1","depends":["equity","peak"],"dtype":"Series","return_result":"drawdown"},

      "cvar_95":    {"category":"risk","type":"cvar","formula":"mean(ret<=q05)","depends":["ret"],"dtype":"float","return_result":"cvar_95"},

      "decorrelation":{"category":"risk","type":"decorrelation","formula":"1-mean(abs(corr_matrix))","depends":["corr_rm"],"dtype":"float","return_result":"decorrelation"},
      "wiggle":      {"category":"risk","type":"wiggle","formula":"diff(corr_matrix)","depends":["corr_rm"],"dtype":"float","return_result":"wiggle"},

      # =========================
      # ALPHA
      # =========================
      "alpha_ts":   {"category":"alpha","type":"ts","formula":"rolling_mean(ret,10)","depends":["ret"],"dtype":"Series","return_result":"alpha_ts"},
      "alpha_xs":   {"category":"alpha","type":"xs","formula":"ret-cs_mean","depends":["ret","cs_mean"],"dtype":"DataFrame","return_result":"alpha_xs"},
      "alpha_pure": {"category":"alpha","type":"pure","formula":"alpha_ts+alpha_xs","depends":["alpha_ts","alpha_xs"],"dtype":"DataFrame","return_result":"alpha_pure"},
      "beta_alpha": {"category":"alpha","type":"beta","formula":"cov/var","depends":["cov_rm","var_ret"],"dtype":"float","return_result":"beta_alpha"},

      # =========================
      # PORTFOLIO
      # =========================
      "inv_vol":     {"category":"portfolio","type":"inv_vol","formula":"1/std","depends":["std_ret"],"dtype":"Series","return_result":"inv_vol"},
      "risk_parity": {"category":"portfolio","type":"risk_parity","formula":"inv_vol/sum(inv_vol)","depends":["inv_vol"],"dtype":"Series","return_result":"weights"},
      "kelly":       {"category":"portfolio","type":"kelly","formula":"mean/var","depends":["mean_ret","var_ret"],"dtype":"Series","return_result":"weights"},
      "mvo":         {"category":"portfolio","type":"mvo","formula":"mu/sigma^2","depends":["mean_ret","var_ret"],"dtype":"Series","return_result":"weights"},
      "entropy":     {"category":"portfolio","type":"entropy","formula":"-sum(w log w)","depends":["weight"],"dtype":"float","return_result":"entropy"},

      # =========================
      # EXECUTION
      # =========================
      "slippage": {"category":"execution","type":"slippage","formula":"sqrt(volatility)","depends":["volatility"],"dtype":"Series","return_result":"slippage"},
      "impact":   {"category":"execution","type":"impact","formula":"sqrt(weight)","depends":["weight"],"dtype":"Series","return_result":"impact"},
      "turnover": {"category":"execution","type":"turnover","formula":"abs(w_t-w_t-1)","depends":["weight"],"dtype":"Series","return_result":"turnover"},

      # =========================
      # MARKET STRUCTURE REGIME
      # =========================
      "regime": {"category": "market_structure", "type": "regime", "formula": "detect(r)", "depends": ["returns"], "dtype": "string", "return_result": True}, 
      "liquidity_adj_vol": {"category": "market_structure", "type": "liquidity_adj_vol", "formula": "adjust(volatility, volume)", "depends":["volatility", "volume"], "dtype": "float", "return_result": True},

      # =========================
      # INTEL
      # =========================
      "ic": {"category":"intel","type":"ic","formula":"corr(signal,future_ret)","depends":["ret"],"dtype":"float","return_result":"ic"},

      # =========================
      # DECISION
      # =========================
      "score":  {"category":"decision","type":"score","formula":"alpha+sharpe+ic-cvar","depends":["alpha_pure","sharpe","ic","cvar_95"],"dtype":"Series","return_result":"score"},
      "signal": {"category":"decision","type":"signal","formula":"threshold(score)","depends":["score"],"dtype":"Series","return_result":"signal"},

      "coefficient":{"category":"basic","type":"coefficient","formula":"ols_beta","depends":["ret","mkt_ret"],"dtype":"float","return_result":"cient"},
      "intercept":{"category":"basic","type":"intercept","formula":"ols_alpha","depends":["ret","mkt_ret","coefficient"],"dtype":"float","return_result":"intercept"},
      "fitted":{"category":"basic","type":"fitted","formula":"intercept+coefficient*mkt_ret","depends":["intercept","coefficient","mkt_ret"],"dtype":"Series","return_result":"fitted"},
      "residual":{"category":"basic","type":"residual","formula":"ret-fitted","depends":["ret","fitted"],"dtype":"Series","return_result":"residual"},
      "sse":{"category":"basic","type":"sse","formula":"sum(residual^2)","depends":["residual"],"dtype":"float","return_result":"sse"},
      "sst":{"category":"basic","type":"sst","formula":"sum((ret-mean_ret)^2)","depends":["ret","mean_ret"],"dtype":"float","return_result":"sst"},
      "ssr":{"category":"basic","type":"ssr","formula":"sst-sse","depends":["sst","sse"],"dtype":"float","return_result":"ssr"},
      "adj_r_squared":{"category":"basic","type":"adj_r_squared","formula":"1-(1-r2)*(n-1)/(n-k-1)","depends":["r_squared"],"dtype":"float","return_result":"adj_r_squared"},
      "mse":{"category":"basic","type":"mse","formula":"sse/(n-k)","depends":["sse"],"dtype":"float","return_result":"mse"},
      "rmse":{"category":"basic","type":"rmse","formula":"sqrt(mse)","depends":["mse"],"dtype":"float","return_result":"rmse"},
      "mae":{"category":"basic","type":"mae","formula":"mean(abs(residual))","depends":["residual"],"dtype":"float","return_result":"mae"},
      "stderr_beta":{"category":"basic","type":"stderr_beta","formula":"sqrt(var(beta))","depends":["coefficient","mse"],"dtype":"float","return_result":"stderr_beta"},
      "stderr_alpha":{"category":"basic","type":"stderr_alpha","formula":"sqrt(var(alpha))","depends":["intercept","mse"],"dtype":"float","return_result":"stderr_alpha"},
      "tstat_iid":{"category":"basic","type":"tstat_iid","formula":"coefficient/stderr_beta","depends":["coefficient","stderr_beta"],"dtype":"float","return_result":"tstat_iid"},
      "tstat_hac":{"category":"basic","type":"tstat_hac","formula":"coefficient/hac_stderr","depends":["coefficient","residual"],"dtype":"float","return_result":"tstat_hac"},
      "r_squared":{"category":"basic","type":"r_squared","formula":"1-sse/sst","depends":["sse","sst"],"dtype":"float","return_result":"r_squared"},
      "tstat_neweywest":{"category":"basic","type":"tstat_neweywest","formula":"coefficient/neweywest_stderr","depends":["coefficient","residual"],
                         "dtype":"float","return_result":"tstat_neweywest"},
      "tstat_alpha":{"category":"basic","type":"tstat_alpha","formula":"intercept/stderr_alpha","depends":["intercept","stderr_alpha"],"dtype":"float","return_result":"tstat_alpha"},
      "f_stat":{"category":"basic","type":"f_stat","formula":"(ssr/1)/(sse/(n-2))","depends":["ssr","sse"],"dtype":"float","return_result":"f_stat"},
      "n_obs":{"category":"basic","type":"n_obs","formula":"count(ret)","depends":["ret"],"dtype":"int","return_result":"n_obs"},
      "hit_ratio" : { "category": "performance",  "type": "hit_ratio", "formula": "(np.sign(preds) == np.sign(rets)).mean()", "depends": ["preds", "rets"],  "dtype": "float",
    "return_result": True},
      "preds" : { "category": "signal", "type": "preds", "formula": "predictions",  "depends": ["raw_signal"], "dtype": "array", "return_result": False},
    }

# =================================================================
# END OF FORMULAOUTPUT
# =================================================================