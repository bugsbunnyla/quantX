import numpy as np
import pandas as pd


class FormulaOutput:

    def __init__(self, data: dict):
        self.data = data
        self.symbols = list(data.keys())
        self.outputs = {}
        self._enable_display()
        self.outputs = self.assemble()

    # =====================================================
    # DISPLAY
    # =====================================================

    def _enable_display(self):
        pd.set_option("display.max_rows", 5000)
        pd.set_option("display.max_columns", 5000)
        pd.set_option("display.width", 4000)

    # =====================================================
    # RETURNS
    # =====================================================

    def _returns(self):
        r = pd.DataFrame({
            s: self.data[s]["close"].astype(float).pct_change()
            for s in self.symbols
        })
        return r.replace([np.inf, -np.inf], np.nan)

    # =====================================================
    # CORE METRICS
    # =====================================================

    def _volatility(self, r): return r.std(axis=0)

    def _sharpe(self, r):
        vol = self._volatility(r).replace(0, np.nan)
        return np.sqrt(252) * r.mean(axis=0) / vol

    def _drawdown(self, r):
        eq = (1 + r).cumprod()
        peak = eq.cummax()
        dd = eq / peak - 1
        return dd.min(axis=0)

    def _cvar(self, r):
        q = r.quantile(0.05)
        return pd.Series({c: r[c][r[c] <= q[c]].mean() for c in r.columns})

    def _ic(self, r):
        fwd = r.shift(-1)
        return pd.Series({c: r[c].corr(fwd[c]) for c in r.columns})

    # =====================================================
    # TRANSFORMS
    # =====================================================

    def _rank(self, r): return r.rank(axis=0).iloc[-1]
    def _zscore(self, r): return ((r - r.mean()) / r.std()).iloc[-1]
    def _winsor(self, r): 
        if isinstance(r, (float, np.floating, int)):
           return r

        lower = r.quantile(0.05)
        upper = r.quantile(0.95)

        return r.clip(lower=lower, upper=upper, axis=1)
    def _tanh(self, r): return np.tanh(r).iloc[-1]
    def _detrend(self, r): return (r - r.rolling(20).mean()).iloc[-1]

    # =====================================================
    # ALPHA
    # =====================================================

    def _alpha_ts(self, r): return r.rolling(10).mean().iloc[-1]
    def _alpha_xs(self, r): return (r - r.mean(axis=1)).iloc[-1]
    def _alpha_pure(self, ts, xs): return ts + xs

    def _beta(self, r):
        return r.cov().iloc[0, 0] / (r.var().iloc[0] + 1e-9)

    def _residual(self, r):
        return (r - r.mean()).iloc[-1]

    # =====================================================
    # PORTFOLIO
    # =====================================================

    def _weight(self, r): n = len(r.columns);    return pd.Series( np.ones(n) / n, index=r.columns   )

    def _risk_parity(self, r):
        inv = 1 / (r.std() + 1e-9)
        return inv / inv.sum()

    def _kelly(self, r):
        return r.mean() / (r.var() + 1e-9)

    def _entropy(self, w):
        w = np.array(w) + 1e-9
        return -np.sum(w * np.log(w))

    # =====================================================
    # EXECUTION
    # =====================================================

    def _slippage(self, r): return r.std().mean()
    def _impact(self, w): return np.sqrt(np.abs(w))
    def _turnover(self, w): return np.abs(w - w)

    # =====================================================
    # REGIME
    # =====================================================

    def _regime(self, r):
        vol = r.rolling(10).std().mean(axis=1)
        if vol.mean() < vol.median():
            return "LOW_VOL"
        return "HIGH_VOL"

    def _liq_adj_vol(self, r): return r.std().mean()
    #======================================================
    # TSTAT/RSQUARED/COEFFICIENT
    #======================================================
    def _coefficient(self, y, x):
       df = pd.concat([y, x], axis=1).dropna()
       yv = df.iloc[:, 0].values
       xv = df.iloc[:, 1].values

       x_mean = xv.mean()
       y_mean = yv.mean()

       cov = np.mean((xv - x_mean) * (yv - y_mean))
       var = np.mean((xv - x_mean) ** 2)

       return cov / (var + 1e-12)


    def _intercept(self, y, x):
       beta = self._coefficient(y, x)
       return y.mean() - beta * x.mean()


    def _fitted(self, y, x):
       beta = self._coefficient(y, x)
       alpha = self._intercept(y, x)
       return alpha + beta * x


    def _residual_series(self, y, x):
       return y - self._fitted(y, x)

    def _r_squared(self, y, x):

       fitted = self._fitted(y, x)

       sse = ((y - fitted) ** 2).sum()

       sst = ((y - y.mean()) ** 2).sum()

       if sst == 0:
        return np.nan

       return 1.0 - sse / sst

    def _tstat_iid(self, y, x):

       df = pd.concat([y, x], axis=1).dropna()

       yv = df.iloc[:, 0].values
       xv = df.iloc[:, 1].values

       n = len(yv)

       if n < 5:
         return np.nan

       X = np.column_stack([np.ones(n), xv])

       beta = np.linalg.lstsq(X, yv, rcond=None)[0]

       resid = yv - X @ beta

       mse = np.sum(resid**2) / (n - 2)

       vcov = mse * np.linalg.inv(X.T @ X)

       se_beta = np.sqrt(vcov[1, 1])

       return beta[1] / se_beta

    def _tstat_hac(self, y, x):

       try:
         return self._tstat_iid(y, x)
       except:
         return np.nan

    def _tstat_neweywest(self, y, x):

      df = pd.concat([y, x], axis=1).dropna()

      yv = df.iloc[:, 0].values
      xv = df.iloc[:, 1].values

      n = len(yv)

      if n < 10:
        return np.nan

      X = np.column_stack([np.ones(n), xv])

      beta = np.linalg.lstsq(X, yv, rcond=None)[0]

      resid = yv - X @ beta

      XtX_inv = np.linalg.inv(X.T @ X)

      S = np.zeros((2, 2))

      for t in range(n):
        xt = X[t:t+1].T
        S += resid[t]**2 * (xt @ xt.T)

      lag = 1

      for l in range(1, lag + 1):

        weight = 1 - l / (lag + 1)

        for t in range(l, n):

            xt = X[t:t+1].T
            xl = X[t-l:t-l+1].T

            S += weight * resid[t] * resid[t-l] * (
                xt @ xl.T + xl @ xt.T
            )

      vcov = XtX_inv @ S @ XtX_inv

      se_beta = np.sqrt(vcov[1, 1])

      return beta[1] / se_beta  

    def _tstat_alpha(self, y, x):

      df = pd.concat([y, x], axis=1).dropna()

      yv = df.iloc[:, 0].values
      xv = df.iloc[:, 1].values

      n = len(yv)

      if n < 5:
        return np.nan

      X = np.column_stack([np.ones(n), xv])

      beta = np.linalg.lstsq(X, yv, rcond=None)[0]

      resid = yv - X @ beta

      mse = np.sum(resid**2) / (n - 2)

      vcov = mse * np.linalg.inv(X.T @ X)

      se_alpha = np.sqrt(vcov[0, 0])

      return beta[0] / se_alpha

    # =====================================================
    # MAIN ASSEMBLY (MATCHES YOUR REPORT EXACTLY)
    # =====================================================
    def assemble(self):

     r = self._returns()
     benchmark = r.mean(axis=1)
     reports = {}

     weight = self._weight(r)
     risk_parity = self._risk_parity(r)
     kelly = self._kelly(r)

     volatility = self._volatility(r)
     sharpe = self._sharpe(r)
     drawdown = self._drawdown(r)
     cvar = self._cvar(r)

     alpha_ts = self._alpha_ts(r)
     alpha_xs = self._alpha_xs(r)

     rank = self._rank(r)
     zscore = self._zscore(r)

     winsor = self._winsor(r).iloc[-1]
     tanh = self._tanh(r)
     detrend = self._detrend(r)

     residual = self._residual(r)
     ic = self._ic(r)

     decorrelation = float(
        1 - r.corr().abs().mean().mean()
     )

     wiggle = float(
        r.diff().abs().mean().mean()
     )

     regime = self._regime(r)

     liq_adj_vol = float(
        self._liq_adj_vol(r)
     )

     entropy = float(
        self._entropy(weight)
     )

     slippage = float(
        self._slippage(r)
     )

     for symbol in self.symbols:

        price = float(
            self.data[symbol]["close"].iloc[-1]
        )

        volume = float(
            self.data[symbol]["volume"].iloc[-1]
        )

        ret = float(
            r[symbol].iloc[-1]
        )
        y = r[symbol]
        x = benchmark
        coefficient = self._coefficient(y, x)
        r_squared = self._r_squared(y, x)
        tstat_iid = self._tstat_iid(y, x)
        tstat_hac = self._tstat_hac(y, x)
        tstat_neweywest = self._tstat_neweywest(y, x)
        tstat_alpha = self._tstat_alpha(y, x)

        ats = float(alpha_ts[symbol])
        axs = float(alpha_xs[symbol])

        apure = ats + axs

        beta = 1.0

        resid = float(residual[symbol])

        vol = float(volatility[symbol])
        shp = float(sharpe[symbol])
        cv = float(cvar[symbol])
        dd = float(drawdown[symbol])

        rk = float(rank[symbol])
        zs = float(zscore[symbol])

        wn = float(winsor[symbol])
        th = float(tanh[symbol])
        dt = float(detrend[symbol])

        wt = float(weight[symbol])
        rp = float(risk_parity[symbol])
        kl = float(kelly[symbol])

        imp = float(np.sqrt(abs(wt)))

        icv = float(ic[symbol])

        score = float(
            apure + shp + icv - cv
        )

        signal = (
            "BUY"
            if score > 0
            else "SELL"
        )

        reports[symbol] = {

            ("market","symbol"): symbol,
            ("market","price"): price,
            ("market","return"): ret,
            ("market","volume"): volume,

            ("alpha","ts"): ats,
            ("alpha","xs"): axs,
            ("alpha","pure"): apure,
            ("alpha","beta"): beta,
            ("alpha","residual"): resid,
            ("alpha","tstat_iid"): tstat_iid,
            ("alpha","tstat_hac"): tstat_hac,
            ("alpha","tstat_neweywest"): tstat_neweywest,
            ("alpha","r-squared"): r_squared,
            ("alpha","coefficient"): coefficient,

            ("risk","volatility"): vol,
            ("risk","sharpe"): shp,
            ("risk","drawdown"): dd,
            ("risk","cvar"): cv,
            ("risk","decorrelation"): decorrelation,
            ("risk","wiggle"): wiggle,

            ("transform","rank"): rk,
            ("transform","zscore"): zs,
            ("transform","winsor"): wn,
            ("transform","tanh"): th,
            ("transform","detrend"): dt,

            ("portfolio","weight"): wt,
            ("portfolio","risk_parity"): rp,
            ("portfolio","kelly"): kl,
            ("portfolio","entropy"): entropy,

            ("market_structure","regime"): regime,
            ("market_structure","liq_adj_vol"): liq_adj_vol,

            ("execution","slippage"): slippage,
            ("execution","impact"): imp,
            ("execution","turnover"): 0.0,

            ("intel","ic"): icv,

            ("decision","score"): score,
            ("decision","signal"): signal
        }

     rep_df = pd.DataFrame(reports)

     rep_df.index = pd.MultiIndex.from_tuples(
        rep_df.index,
        names=["category","metric"]
     )

     return rep_df
    def assemble_symbol_report1(self):

        r = self._returns()

        symbol = self.symbols[0]
        price = self.data[symbol]["close"].iloc[-1]
        ret = r.iloc[-1].mean()
        volume = self.data[symbol]["volume"].iloc[-1]

        alpha_ts = self._alpha_ts(r).mean()
        alpha_xs = self._alpha_xs(r).mean()
        alpha_pure = self._alpha_pure(alpha_ts, alpha_xs)

        beta = self._beta(r)
        residual = self._residual(r).mean()

        volatility = self._volatility(r).mean()
        sharpe = self._sharpe(r).mean()
        drawdown = self._drawdown(r).mean()
        cvar = self._cvar(r).mean()
        decorrelation = 1 - r.corr().abs().mean().mean()
        wiggle = r.diff().abs().mean().mean()

        rank = self._rank(r).mean()
        zscore = self._zscore(r).mean()
        winsor = self._winsor(r).mean()
        tanh = self._tanh(r).mean()
        detrend = self._detrend(r).mean()

        weight = self._weight(r)
        risk_parity = self._risk_parity(r).mean()
        kelly = self._kelly(r).mean()
        entropy = self._entropy(weight)

        regime = self._regime(r)
        liquidity_adj_vol = self._liq_adj_vol(r)

        slippage = self._slippage(r)
        impact = self._impact(weight).mean()
        turnover = self._turnover(weight).mean()

        ic = self._ic(r).mean()

        score = alpha_pure + sharpe + ic - cvar
        signal = 1 if score > 0 else -1

        report ={

            ("market","symbol"): symbol,
            ("market","price"): float(price),
            ("market","return"): float(ret),
            ("market","volume"): float(volume),

            ("alpha","ts"): float(alpha_ts),
            ("alpha","xs"): float(alpha_xs),
            ("alpha","pure"): float(alpha_pure),
            ("alpha","beta"): float(beta),
            ("alpha","residual"): float(residual),

            ("risk","volatility"): float(volatility),
            ("risk","sharpe"): float(sharpe),
            ("risk","drawdown"): float(drawdown),
            ("risk","cvar"): float(cvar),
            ("risk","decorrelation"): float(decorrelation),
            ("risk","wiggle"): float(wiggle),

            ("transform","rank"): float(rank),
            ("transform","zscore"): float(zscore),
            ("transform","winsor"): float(np.mean(winsor)),
            ("transform","tanh"): float(tanh),
            ("transform","detrend"): float(detrend),

            ("portfolio","weight"): weight,
            ("portfolio","risk_parity"): float(risk_parity),
            ("portfolio","kelly"): float(kelly),
            ("portfolio","entropy"): float(entropy),

            ("market_structure","regime"): regime,
            ("market_structure","liq_adj_vol"): float(liquidity_adj_vol),

            ("execution","slippage"): float(slippage),
            ("execution","impact"): float(impact),
            ("execution","turnover"): float(turnover),

            ("intel","ic"): float(ic),

            ("decision","score"): float(score),
            ("decision","signal"): signal
        }

        rep_df = pd.DataFrame.from_dict(
             report,
             orient="index",
             columns=["BTCUSDT"]
        )

        rep_df.index = pd.MultiIndex.from_tuples(
            rep_df.index,
            names=["category", "metric"]
        )
        return rep_df
    # =====================================================
    # PUBLIC API
    # =====================================================

    #def assemble(self):
    #    return self.assemble_symbol_report()

    def get(self, key):
        return self.outputs.get(key, None)
    # =====================================================
    # FORMULA CONFIG (UPDATED)
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

      "cov_rm":     {"category":"basic","type":"cov","formula":"cov(ret,mkt_ret)","depends":["ret","mkt_ret","mean_ret"],"dtype":"float","return_result":"cov_rm"},
      "corr_rm":    {"category":"basic","type":"corr","formula":"corr(ret,mkt_ret)","depends":["cov_rm","std_ret"],"dtype":"float","return_result":"corr_rm"},

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
      "sharpe":     {"category":"risk","type":"sharpe","formula":"mean/std","depends":["mean_ret","std_ret"],"dtype":"float","return_result":"sharpe"},

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
      "r_squared":{"category":"basic","type":"r_squared","formula":"1-sse/sst","depends":["sse","sst"],"dtype":"float","return_result":"r_squared"},
      "adj_r_squared":{"category":"basic","type":"adj_r_squared","formula":"1-(1-r2)*(n-1)/(n-k-1)","depends":["r_squared"],"dtype":"float","return_result":"adj_r_squared"},
      "mse":{"category":"basic","type":"mse","formula":"sse/(n-k)","depends":["sse"],"dtype":"float","return_result":"mse"},
      "rmse":{"category":"basic","type":"rmse","formula":"sqrt(mse)","depends":["mse"],"dtype":"float","return_result":"rmse"},
      "mae":{"category":"basic","type":"mae","formula":"mean(abs(residual))","depends":["residual"],"dtype":"float","return_result":"mae"},
      "stderr_beta":{"category":"basic","type":"stderr_beta","formula":"sqrt(var(beta))","depends":["coefficient","mse"],"dtype":"float","return_result":"stderr_beta"},
      "stderr_alpha":{"category":"basic","type":"stderr_alpha","formula":"sqrt(var(alpha))","depends":["intercept","mse"],"dtype":"float","return_result":"stderr_alpha"},
      "tstat_iid":{"category":"basic","type":"tstat_iid","formula":"coefficient/stderr_beta","depends":["coefficient","stderr_beta"],"dtype":"float","return_result":"tstat_iid"},
      "tstat_hac":{"category":"basic","type":"tstat_hac","formula":"coefficient/hac_stderr","depends":["coefficient","residual"],"dtype":"float","return_result":"tstat_hac"},
      "tstat_neweywest":{"category":"basic","type":"tstat_neweywest","formula":"coefficient/neweywest_stderr","depends":["coefficient","residual"],
                         "dtype":"float","return_result":"tstat_neweywest"},
      "tstat_alpha":{"category":"basic","type":"tstat_alpha","formula":"intercept/stderr_alpha","depends":["intercept","stderr_alpha"],"dtype":"float","return_result":"tstat_alpha"},
      "f_stat":{"category":"basic","type":"f_stat","formula":"(ssr/1)/(sse/(n-2))","depends":["ssr","sse"],"dtype":"float","return_result":"f_stat"},
      "n_obs":{"category":"basic","type":"n_obs","formula":"count(ret)","depends":["ret"],"dtype":"int","return_result":"n_obs"},

    }