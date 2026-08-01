# ===============================================================
# FormulaInfo class defines the BaseStrategy structured data
# Date: 2026/06/22 
# Author : bugsbunnyla
# Comment : Core formulas for the QuantXpert structure data model
# ===============================================================
import numpy as np
import pandas as pd
import ast
import numexpr 
import bottleneck


class RegressionEngine:

    def __init__(self, formula_config):
        self.config = formula_config
        self.cache = {}
        self.data = {}
        self.master_index = []

    # =====================================================
    # SAFE SCOPE (UNCHANGED)
    # =====================================================
    def _scope(self, context=None):
        if context is None:
            context = {}

        scope = {
            "np": np,
            "pd": pd,
            "min": self.min_,
            "max": self.max_,
            "len": self.len_,
            "mean": self.mean,
            "sum": self.sum_,
            "sqrt": self.sqrt,
            "log": self.log,
            "abs": self.abs_,
            "var": self.var,
            "cov": self.cov,
            "corr": self.corr,
            "std": self.std,
            "inv": self.inv,
            "rolling_mean": self.rolling_mean,
            "rolling_std": self.rolling_std,
            "rolling_var": self.rolling_var,
            "window": 20,
            "lag": 1,
            "lookback": 5,
            "n": lambda x=None: len(self.master_index),
            "N": len(self.master_index),
            "k": len(self.master_index),
        }
        scope.update(self.data)
        scope.update(context)
        return scope

    # =====================================================
    # ENTRY POINT (FIXED reg_ SAFE ROUTING ONLY)
    # =====================================================
    def run(self, key, context=None):
        if context is None:
            context = {}
        if key in self.cache:
            return self.cache[key]
        spec = self.config[key]
        context = self._apply_transforms(spec, context)
        if self._needs_ols(spec):
            context["_ols_result"] = self._ols_from_context(spec, context)
        if key.startswith("reg_"):
            result = self._evaluate_reg(spec["formula"], context)
        else:
            result = self._evaluate(spec["formula"], context)
        self.cache[key] = result
        return result

    # =====================================================
    # NEW: REG SAFE EVALUATOR (NO CORE CHANGES)
    # =====================================================
    def _evaluate_reg(self, formula, context):
        scope = self._scope(context)
        ols = context.get("_ols_result", None)
        if ols is not None:
            scope["_ols"] = ols
            scope["_ols_result"] = ols
            if hasattr(ols, "residuals"):
                scope["residuals"] = ols.residuals
            if hasattr(ols, "residual"):
                scope["residual"] = ols.residual
        return eval(formula, {"__builtins__": {}}, scope)

    # =====================================================
    # OLS RESOLUTION (UNCHANGED LOGIC)
    # =====================================================
    def _ols_from_context(self, spec, context):
        depends = spec.get("depends", {})
        y_key = depends["y"]
        X_keys = depends["X"]
        y = self._lookup(y_key, context, reg_only=True)
        if isinstance(X_keys, list):
            X = [self._lookup(k, context, reg_only=True) for k in X_keys]
        else:
            X = self._lookup(X_keys, context, reg_only=True)
        return self._ols(y, X)

    # =====================================================
    # LOOKUP (UNCHANGED)
    # =====================================================
    def _lookup(self, key, context, reg_only=False):
        if reg_only:
            if key in self.cache:
                return self.cache[key]
            if key in context:
                return context[key]
            if key in self.data:
                return self.data[key]
        if key in context:
            return context[key]
        if key in self.data:
            return self.data[key]
        if key in self.cache:
            return self.cache[key]
        raise NameError(f"{key} not found in context/data/cache")

    # =====================================================
    # OLS CORE (FIXED NUMPY SAFETY ONLY)
    # =====================================================
    def _ols(self, y, X):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        mask = np.isfinite(X).all(axis=1) & np.isfinite(y).all(axis=1)
        X = X[mask]
        y = y[mask]
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"OLS row mismatch: X={X.shape}, y={y.shape}")
        XtX = X.T @ X
        XtY = X.T @ y
        beta = np.linalg.pinv(XtX) @ XtY
        fitted = X @ beta
        residuals = y - fitted
        mse = np.mean(residuals ** 2)
        yvar = np.var(y)
        r2 = 1.0 - mse / (yvar + 1e-12)
        alpha = float(np.mean(residuals))
        return type(
            "OLSResult",
            (),
            {
                "beta": beta,
                "alpha": alpha,
                "fitted": fitted,
                "residuals": residuals,
                "residual": residuals,
                "mse": mse,
                "r2": r2,
                "nobs": X.shape[0],
                "k": X.shape[1],
            },
        )

    # =====================================================
    # TRANSFORMS (UNCHANGED)
    # =====================================================
    def _apply_transforms(self, spec, context):
        for new_key, rule in spec.get("transforms", {}).items():
            if rule == "transpose(alpha)":
                context[new_key] = context["alpha"].T
            elif rule == "normalize":
                x = context[new_key]
                context[new_key] = (x - np.mean(x)) / (np.std(x) + 1e-8)
            elif rule == "log":
                context[new_key] = np.log(context[new_key])
        return context

    # =====================================================
    # EVAL (UNCHANGED)
    # =====================================================
    def _evaluate(self, formula, context):
        scope = self._scope(context)
        return eval(formula, {"__builtins__": {}}, scope)

    # =====================================================
    # NEEDS OLS (UNCHANGED)
    # =====================================================
    def _needs_ols(self, spec):
        return "_ols" in spec["formula"]

    # =====================================================
    # MATH HELPERS (UNCHANGED)
    # =====================================================
    def sum_(self, x): return np.sum(x)
    def mean(self, x): return np.mean(x)
    def sqrt(self, x): return np.sqrt(x)
    def log(self, x): return np.log(x)
    def abs_(self, x): return np.abs(x)
    def var(self, x): return np.var(x)
    def std(self, x): return np.std(x)
    def len_(self, x): return len(x)
    def min_(self, x): return min(x)
    def max_(self, x): return max(x)

    def cov(self, x, y): return np.cov(x, y)[0, 1]
    def corr(self, x, y):
        if isinstance(x, pd.Series):
            return x.corr(y)
        if isinstance(x, pd.DataFrame):
            if isinstance(y, pd.Series):
                return x.corrwith(y)
            if isinstance(y, pd.DataFrame):
                return x.corrwith(y)
        return np.corrcoef(np.asarray(x), np.asarray(y))[0, 1]

    def inv(self, x): return 1 / x
    def rolling_mean(self, x, w=20): return pd.Series(x).rolling(w).mean().values
    def rolling_std(self, x, w=20): return pd.Series(x).rolling(w).std().values
    def rolling_var(self, x, w=20): return pd.Series(x).rolling(w).var().values


class FormulaInfo:

    def __init__(self, data):
        self.data = data
        self.symbols = list(data.keys())
        self.build_master_index()
        self.RegressionEngine = RegressionEngine(self.FORMULA_CONFIG)
        self.cache = {}
        # FIX: build base scope once (avoid recomputation)
        self.base_scope = self._scope()

    # =====================================================
    # MASTER INDEX
    # =====================================================
    def build_master_index(self):
        idx = None
        for df in self.data.values():
            if isinstance(df, pd.DataFrame):
                idx = df.index if idx is None else idx.union(df.index)
        self.master_index = idx.sort_values()

    # =====================================================
    # RAW MARKET BUILDERS
    # =====================================================
    def _build_field(self, field):
        series_dict = {}
        for s, df in self.data.items():
            if df is None:
                continue
            if field not in df.columns:
                continue
            series_dict[s] = pd.to_numeric(df[field], errors="coerce")
        if len(series_dict) == 0:
            raise ValueError(f"Field '{field}' not found in any symbol DataFrame")
        return pd.concat(series_dict, axis=1).sort_index()

    # =====================================================
    # GLOBAL SCOPE
    # =====================================================
    def _scopep(self):
        data_scope = {
            "close": self._build_field("close"),
            "open": self._build_field("open"),
            "high": self._build_field("high"),
            "low": self._build_field("low"),
            "volume": self._build_field("volume"),
            "symbol": pd.DataFrame(
                {s: [s] * len(self.master_index) for s in self.symbols},
                index=self.master_index
            )
        }
        func_scope = {
            "tuple": tuple,
            "list": list,
            "range": range,
            "len": len,
            "float": float,
            "int": int,
            "bool": bool,
            "slice": slice,
            "min": self.min_,
            "max": self.max_,
            "len": self.len_,
            "mean": self.mean,
            "sum": self.sum_,
            "sqrt": self.sqrt,
            "log": self.log,
            "abs": self.abs_,
            "var": self.var,
            "cov": self.cov,
            "corr": self.corr,
            "std": self.std,
            "count": self.count,
            "inv": self.inv,
            "rolling_mean": self.rolling_mean,
            "rolling_std": self.rolling_std,
            "rolling_var": self.rolling_var,
            "build_X": self.build_X,
            "transpose_": self.transpose_,
        }
        scope = {
            "np": np,
            "pd": pd,
            "window": 20,
            "lag": 1,
            "lookback": 5,
            "k": 2,
            "n": len(self.master_index),
            **func_scope,
            **data_scope,
            **self.data,
        }
        return scope

    def _scope(self):
        data_scope = {
            "close": self._build_field("close"),
            "open": self._build_field("open"),
            "high": self._build_field("high"),
            "low": self._build_field("low"),
            "volume": self._build_field("volume"),
            "symbol": pd.DataFrame(
                {s: [s] * len(self.master_index) for s in self.symbols},
                index=self.master_index
            )
        }
        func_scope = {
            "tuple": tuple,
            "list": list,
            "range": range,
            "len": len,
            "float": float,
            "int": int,
            "bool": bool,
            "slice": slice,
            "min": self.min_,
            "max": self.max_,
            "len_": self.len_,
            "mean": self.mean,
            "sum": self.sum_,
            "sqrt": self.sqrt,
            "log": self.log,
            "abs": self.abs_,
            "var": self.var,
            "cov": self.cov,
            "corr": self.corr,
            "std": self.std,
            "count": self.count,
            "inv": self.inv,
            "rolling_mean": self.rolling_mean,
            "rolling_std": self.rolling_std,
            "rolling_var": self.rolling_var,
            "build_X": self.build_X,
            "transpose_": self.transpose_,
        }
        cost_scope = {
            "TCOST_BUY": 0.10,
            "TCOST_SELL": 0.10,
            "SERVICE_FEE": 0.05,
            "RISK_FEE": 0.05,
            "EXCHANGE_FEE": 0.03,
            "DELTA_UNK_FEE": 0.08,
            "TCOST": 0.41,
            "DELTA": 0.08
        }
        scope = {
            "np": np,
            "pd": pd,
            "window": 20,
            "lag": 1,
            "lookback": 5,
            "k": 2,
            "n": len(self.master_index),
            **func_scope,
            **data_scope,
            **self.data,
            **cost_scope,
        }
        return scope

    # =====================================================
    # SAFE OPS (FIXED CRASHES)
    # =====================================================
    def _is_vector(self, x):
        return isinstance(x, (pd.Series, pd.DataFrame, np.ndarray))

    def mean(self, x):
        return x.mean() if self._is_vector(x) else x

    def sum_(self, x):
        return x.sum() if self._is_vector(x) else x

    def var(self, x):
        return x.var() if self._is_vector(x) else 0.0

    def std(self, x):
        return x.std() if self._is_vector(x) else 0.0

    def count(self, x):
        return x.count() if self._is_vector(x) else 1

    def sqrt(self, x):
        return np.sqrt(x)

    def abs_(self, x):
        return np.abs(x)

    def len_(self, x):
        return len(x)

    def min_(self, x):
        return x.min() if self._is_vector(x) else x

    def max_(self, x):
        return x.max() if self._is_vector(x) else x

    def log(self, x):
        return np.log(x)

    def inv(self, x):
        if isinstance(x, pd.DataFrame):
            x = x.values
        return np.linalg.pinv(x)

    def cov(self, x, y, window=20):
        return x.rolling(window, min_periods=1).cov(y)

    def corr(self, x, y, window=20):
        return x.rolling(window, min_periods=1).corr(y)

    def rolling_mean_(self, x, window=20, min_periods=1):
        return x.rolling(window, min_periods).mean()

    def rolling_mean(self, x, window=20):
        return x.rolling(window).mean()

    def rolling_std(self, x, w=20):
        return x.rolling(w).std(ddof=0)

    def rolling_var(self, x, w=20):
        return x.rolling(w).var(ddof=0)

    # =====================================================
    # DAG COMPUTE (FIXED)
    # =====================================================
    def compute(self, prop, visited=None):
        if visited is None:
            visited = set()
        if prop in self.cache:
            return self.cache[prop]
        if prop in visited:
            raise ValueError(f"Circular dependency: {prop}")
        visited.add(prop)
        try:
            cfg = self.FORMULA_CONFIG[prop]
            formula = cfg["formula"]
            depends = cfg.get("depends", [])
            for dep in depends:
                if dep not in self.cache:
                    self.compute(dep, visited)
            scope = dict(self.base_scope)
            scope.update(self.cache)
            for dep in depends:
                if dep not in scope:
                    raise KeyError(f"{prop}: missing dependency '{dep}'")
            try:
                print(formula, depends, prop)
                val = eval(formula, {"__builtins__": {}}, scope)
                print(val, formula, depends, prop)
            except Exception:
                print("formula:", formula)
                for dep in depends:
                    x = scope[dep]
                    print(dep, type(x), getattr(x, "shape", None), getattr(x, "index", None), getattr(x, "columns", "<no columns>"))
                raise
            if prop == "alpha":
                self.cache[prop] = self.extract_tsx(val)
            else:
                self.cache[prop] = val
            return val
        finally:
            visited.remove(prop)

    def extract_tsx(self, val):
        """
        Normalize formula outputs into cross-sectional symbol Series.
        Expected final shape:
        index = self.symbols
        values = signal values
        Handles:
        - None
        - scalar
        - time-series Series
        - cross-sectional Series
        - DataFrame (time x symbols or symbols x time)
        - numpy arrays
        """
        if val is None:
            return pd.Series(np.nan, index=self.symbols)
        if isinstance(val, pd.Series):
            if set(self.symbols).issubset(val.index):
                return val.reindex(self.symbols)
            if isinstance(val.index, pd.DatetimeIndex):
                return pd.Series(val.iloc[-1], index=self.symbols)
            if len(val) == len(self.symbols):
                return pd.Series(val.values, index=self.symbols)
            return pd.Series(val.iloc[-1], index=self.symbols)
        if isinstance(val, pd.DataFrame):
            val = val.replace([np.inf, -np.inf], np.nan)
            if set(self.symbols).issubset(val.columns):
                return val.iloc[-1].reindex(self.symbols)
            if set(self.symbols).issubset(val.index):
                return val.iloc[:, -1].reindex(self.symbols)
            if len(val):
                return val.iloc[-1].reindex(self.symbols)
            return pd.Series(np.nan, index=self.symbols)
        if isinstance(val, np.ndarray):
            if val.ndim == 0:
                return pd.Series(val.item(), index=self.symbols)
            return pd.Series(val.flatten(), index=self.symbols)
        if np.isscalar(val):
            return pd.Series(val, index=self.symbols)
        return pd.Series(np.nan, index=self.symbols)

    # =====================================================
    # SCALAR SAFE
    # =====================================================
    def scalar(self, x):
        if isinstance(x, pd.Series):
            return x.iloc[-1] if len(x) else np.nan
        if isinstance(x, pd.DataFrame):
            return x.iloc[-1].iloc[-1] if len(x) else np.nan
        return x

    def transpose_(self, x, y=None):
        scope = self._scope()
        df = self.compute("ret")
        if y == x:
            retXT = df.transpose()
            retX = df
            response = retXT @ retX
            return response
        else:
            retXT = df.transpose()
            response = retXT @ df
            return response

    # =====================================================
    # BUILD X
    # =====================================================
    def build_X(self):
        ret = self.compute("ret")
        if isinstance(ret, pd.DataFrame):
            ret = ret.iloc[:, 0]
        if isinstance(ret, pd.Series):
            ret = ret.fillna(0)
        ret = np.asarray(ret)
        ret = ret.reshape(-1, 1)
        ones = np.ones((ret.shape[0], 1))
        X = np.hstack([ones, ret])
        return X

    # =====================================================
    # assemble
    # =====================================================
    def assemble(self):
        for prop in self.FORMULA_CONFIG:
            if prop.startswith("reg_"):
                continue
            else:
                self.compute(prop)

    # =====================================================
    # REPORT function to provide simple output
    # =====================================================
    def reporting(self, mode="both"):
        """
        Build report from cached metrics.
        Parameters
        ----------
        mode : {"both", "multi", "easy"}
         both  -> return (df, df_easy)
         multi -> return df
         easy  -> return df_easy

        Returns
        -------
        df :
         MultiIndex rows (category, metric), columns = symbols.
        df_easy :
         Same values as df, but with a flat index:
             market_price
             alpha_ts
             risk_volatility
             ...
        """
        import numpy as np
        import pandas as pd
        ALLOWED_KEYS = {
            ("market", "symbol"): ("market", "symbol"),
            ("market", "price"): ("market", "price"),
            ("market", "volume"): ("market", "volume"),
            ("market", "mkt_price"): ("market", "mkt_price"),
            ("market", "mkt_ret"): ("market", "mkt_ret"),
            ("market", "ret"): ("market", "ret"),
            ("alpha", "ts"): ("alpha", "ts"),
            ("alpha", "xs"): ("alpha", "xs"),
            ("alpha", "pure"): ("alpha", "pure"),
            ("alpha", "alpha"): ("alpha", "alpha"),
            ("alpha", "beta"): ("alpha", "beta"),
            ("alpha", "residual"): ("alpha", "residual"),
            ("basic", "corr"): ("basic", "corr"),
            ("basic", "corr_rm"): ("basic", "corr_rm"),
            ("basic", "r_squared"): ("basic", "r_squared"),
            ("basic", "hit_ratio"): ("basic", "hit_ratio"),
            ("basic", "tstat"): ("basic", "tstat_alpha"),
            ("basic", "tstat_alpha"): ("basic", "tstat_alpha"),
            ("basic", "tstat_beta"): ("basic", "tstat_beta"),
            ("basic", "norm_weight"): ("basic", "norm_weight"),
            ("execution", "slippage"): ("execution", "slippage"),
            ("execution", "impact"): ("execution", "impact"),
            ("execution", "turnover"): ("execution", "turnover"),
            ("execution", "pturnover"): ("execution", "pturnover"),
            ("execution", "tcost"): ("execution", "transaction_cost"),
            ("execution", "transaction_cost"): ("execution", "transaction_cost"),
            ("intel", "ic"): ("intel", "ic"),
            ("portfolio", "weight"): ("portfolio", "weight"),
            ("portfolio", "risk_parity"): ("portfolio", "risk_parity"),
            ("portfolio", "kelly"): ("portfolio", "kelly"),
            ("portfolio", "inv_vol"): ("portfolio", "inv_vol"),
            ("portfolio", "mvo"): ("portfolio", "mvo"),
            ("portfolio", "entropy"): ("portfolio", "entropy"),
            ("portfolio", "net_ret"): ("portfolio", "net_ret"),
            ("portfolio", "executed_weight"): ("portfolio", "executed_weight"),
            ("risk", "volatility"): ("risk", "volatility"),
            ("risk", "cvar"): ("risk", "cvar"),
            ("risk", "sharpe"): ("risk", "sharpe"),
            ("risk", "drawdown"): ("risk", "drawdown"),
            ("risk", "max_drawdown"): ("risk", "max_drawdown"),
            ("transform", "rank"): ("transform", "rank"),
            ("transform", "zscore"): ("transform", "zscore"),
            ("transform", "winsor"): ("transform", "winsor"),
            ("transform", "tanh"): ("transform", "tanh"),
            ("transform", "detrend"): ("transform", "detrend"),
            ("market_structure", "liq_adj_vol"): ("market_structure", "liquidity_adj_vol"),
            ("market_structure", "liquidity_adj_vol"): ("market_structure", "liquidity_adj_vol"),
            ("market_structure", "regime"): ("market_structure", "regime"),
            ("decision", "score"): ("decision", "score"),
            ("decision", "signal"): ("decision", "signal"),
            ("decision", "psignal"): ("decision", "psignal"),
        }

        if not self.cache:
            self.assemble()

        rows = {}
        for symbol in self.symbols:
            row = {}
            for report_key, cache_key in ALLOWED_KEYS.items():
                cached = self.cache.get(cache_key)
                if cached is None:
                    row[report_key] = np.nan
                    continue
                try:
                    row[report_key] = self.extract_value_for_symbol(
                        cached,
                        symbol,
                        cache_key,
                    )
                except Exception:
                    row[report_key] = np.nan
            rows[symbol] = row

        rpt = pd.DataFrame.from_dict(rows, orient="index")
        rpt.index.name = "symbol"
        rpt.columns = pd.MultiIndex.from_tuples(
            rpt.columns,
            names=["category", "metric"],
        )
        df = rpt.T
        df.index = pd.MultiIndex.from_tuples(
            df.index,
            names=["category", "metric"],
        )
        df_easy = df.copy()
        df_easy.index = pd.Index(
            [
                f"{category}_{metric}"
                for category, metric in df.index
            ],
            name="category_metric",
        )
        return df, df_easy

    def report(self, mode="both"):
        """
        Return assembled metric series indexed by symbol.
        Supports:
          report("category", "metric")
          report(metric="metric")
        Does not alter:
        - assemble()
        - ALLOWED_KEYS
        """
        ALLOWED_KEYS = {
            ("market", "symbol"): ("market", "symbol"),
            ("market", "price"): ("market", "price"),
            ("market", "volume"): ("market", "volume"),
            ("market", "mkt_price"): ("market", "mkt_price"),
            ("market", "mkt_ret"): ("market", "mkt_ret"),
            ("market", "ret"): ("basic", "ret"),
            ("alpha", "ts"): ("alpha", "ts"),
            ("alpha", "xs"): ("alpha", "xs"),
            ("alpha", "pure"): ("alpha", "pure"),
            ("alpha", "alpha"): ("alpha", "alpha"),
            ("alpha", "beta"): ("alpha", "beta"),
            ("alpha", "residual"): ("alpha", "residual"),
            ("basic", "corr"): ("basic", "corr"),
            ("basic", "corr_rm"): ("basic", "corr_rm"),
            ("basic", "r_squared"): ("basic", "r_squared"),
            ("basic", "hit_ratio"): ("basic", "hit_ratio"),
            ("basic", "ret"): ("basic", "ret"),
            ("basic", "tstat"): ("basic", "tstat_alpha"),
            ("basic", "tstat_alpha"): ("basic", "tstat_alpha"),
            ("basic", "tstat_beta"): ("basic", "tstat_beta"),
            ("basic", "norm_weight"): ("basic", "norm_weight"),
            ("execution", "slippage"): ("execution", "slippage"),
            ("execution", "impact"): ("execution", "impact"),
            ("execution", "turnover"): ("execution", "turnover"),
            ("execution", "pturnover"): ("execution", "pturnover"),
            ("execution", "tcost"): ("execution", "transaction_cost"),
            ("execution", "transaction_cost"): ("execution", "transaction_cost"),
            ("intel", "ic"): ("intel", "ic"),
            ("portfolio", "weight"): ("portfolio", "weight"),
            ("portfolio", "risk_parity"): ("portfolio", "risk_parity"),
            ("portfolio", "kelly"): ("portfolio", "kelly"),
            ("portfolio", "inv_vol"): ("portfolio", "inv_vol"),
            ("portfolio", "mvo"): ("portfolio", "mvo"),
            ("portfolio", "entropy"): ("portfolio", "entropy"),
            ("portfolio", "net_ret"): ("portfolio", "net_ret"),
            ("portfolio", "executed_weight"): ("portfolio", "executed_weight"),
            ("risk", "volatility"): ("risk", "volatility"),
            ("risk", "cvar"): ("risk", "cvar"),
            ("risk", "sharpe"): ("risk", "sharpe"),
            ("risk", "drawdown"): ("risk", "drawdown"),
            ("risk", "max_drawdown"): ("risk", "max_drawdown"),
            ("transform", "rank"): ("transform", "rank"),
            ("transform", "zscore"): ("transform", "zscore"),
            ("transform", "winsor"): ("transform", "winsor"),
            ("transform", "tanh"): ("transform", "tanh"),
            ("transform", "detrend"): ("transform", "detrend"),
            ("market_structure", "liq_adj_vol"): ("market_structure", "liquidity_adj_vol"),
            ("market_structure", "liquidity_adj_vol"): ("market_structure", "liquidity_adj_vol"),
            ("market_structure", "regime"): ("market_structure", "regime"),
            ("decision", "score"): ("decision", "score"),
            ("decision", "signal"): ("decision", "signal"),
            ("decision", "psignal"): ("decision", "psignal"),
        }
        rows = {}
        for symbol in self.symbols:
            row = {}
            for (category, metric), alias in ALLOWED_KEYS.items():
                cached = self.cache.get(alias)
                if cached is None:
                    row[(category, metric)] = np.nan
                    continue
                row[(category, metric)] = self.extract_value_for_symbol(
                    cached,
                    symbol,
                    alias
                )
            rows[symbol] = row

        rpt = pd.DataFrame(rows).T
        rpt.index.name = "symbol"
        rpt.columns = pd.MultiIndex.from_tuples(
            rpt.columns,
            names=["category", "metric"]
        )
        return rpt

    def extract_value_for_symbol(self, cached, symbol, alias_metric=None):
        try:
            if isinstance(cached, dict):
                return cached.get(symbol, np.nan)
            if isinstance(cached, pd.Series):
                if symbol in cached.index:
                    return cached.loc[symbol]
                return np.nan
            if isinstance(cached, pd.DataFrame):
                df = cached
                if symbol in df.index:
                    val = df.loc[symbol]
                    if isinstance(val, pd.Series):
                        return val.iloc[0]
                    return val
                if symbol in df.columns:
                    val = df[symbol]
                    if len(val) == 1:
                        return val.iloc[0]
                    return val.iloc[-1]
                if len(df.index) == 1:
                    if symbol in df.columns:
                        return df.iloc[0][symbol]
                if len(df.columns) == 1:
                    if symbol in df.iloc[:, 0].values:
                        return df[df.iloc[:, 0] == symbol].iloc[0, 0]
            return np.nan
        except Exception as e:
            print(
                "extract failed:",
                alias_metric,
                symbol,
                type(cached),
                e
            )
            return np.nan

    def _extract_report_value(self, category, metric):
        """
        Resolve alias category/metric and extract symbol -> value series.
        Returns:
          pd.Series indexed by symbol
        """
        val = metric
        if val is None:
            return pd.Series(index=self.symbols, dtype=object)
        if isinstance(val, pd.DataFrame):
            if set(self.symbols).issubset(val.columns):
                return val[self.symbols].apply(
                    lambda x: x.dropna().iloc[-1]
                    if len(x.dropna())
                    else np.nan
                )
            if set(self.symbols).issubset(val.index):
                return val.loc[self.symbols]
        if isinstance(val, pd.Series):
            if set(self.symbols).issubset(val.index):
                return val.reindex(self.symbols)
        if isinstance(val, dict):
            return pd.Series(val).reindex(self.symbols)
        return pd.Series(
            val,
            index=self.symbols
        )

    def get(self, key, default=None):
        """
        Return a computed metric.
        Automatically computes the metric if it has not
        already been assembled.
        """
        if key not in self.cache:
            if key not in self.FORMULA_CONFIG:
                if default is not None:
                    return default
                raise KeyError(f"Unknown metric '{key}'")
            self.compute(key)
        return self.cache[key]

    FORMULA_CONFIG = {
        "mkt_price": {"category": "market", "type": "mkt_price", "formula": "close", "depends": [], "dtype": "Series", "return_result": "mkt_price"},
        "mkt_price_low": {"category": "market", "type": "mkt_price_low", "formula": "low", "depends": [], "dtype": "Series", "return_result": "mkt_price_low"},
        "mkt_price_high": {"category": "market", "type": "mkt_price_high", "formula": "high", "depends": [], "dtype": "Series", "return_result": "mkt_price_high"},
        "mkt_ret": {"category": "market", "type": "mkt_ret", "formula": "ret.mean(axis=1)", "depends": ["ret"], "dtype": "Series", "return_result": "mkt_ret"},
        "symbol": {"category": "market", "type": "symbol", "formula": "symbol", "depends": [], "dtype": "categorical", "return_result": "symbol"},
        "price": {"category": "market", "type": "price", "formula": "mkt_price", "depends": ["mkt_price"], "dtype": "Series", "return_result": "price"},
        "volume": {"category": "market", "type": "volume", "formula": "volume", "depends": [], "dtype": "Series", "return_result": "volume"},
        "ret": {"category": "basic", "type": "ret", "formula": "mkt_price.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)", "depends": ["mkt_price"], "dtype": "DataFrame", "return_result": "ret"},
        "log_ret": {"category": "basic", "type": "log_ret", "formula": "np.log(mkt_price).diff().fillna(0)", "depends": ["mkt_price"], "dtype": "Series", "return_result": "log_ret"},
        "mean_ret": {"category": "basic", "type": "mean_ret", "formula": "ret.rolling(window).mean()", "depends": ["ret"], "dtype": "Series", "return_result": "mean_ret"},
        "std_ret": {"category": "basic", "type": "std_ret", "formula": "ret.rolling(window).std(ddof=0).replace(0, 1e-9)", "depends": ["ret"], "dtype": "Series", "return_result": "std_ret"},
        "var_ret": {"category": "basic", "type": "var_Ret", "formula": "std_ret ** 2", "depends": ["std_ret"], "dtype": "Series", "return_result": "var_ret"},
        "cov_rm": {"category": "risk", "type": "cov_rm", "formula": "ret.rolling(window).cov(benchmark)", "depends": ["ret", "benchmark"], "dtype": "DataFrame", "return_result": "cov_rm"},
        "corr_rm": {"category": "risk", "type": "corr_rm", "formula": "ret.rolling(window).corr(benchmark)", "depends": ["ret", "benchmark"], "dtype": "DataFrame", "return_result": "corr_rm"},
        "rolling_mean": {"category": "basic", "type": "rolling_mean", "formula": "ret.rolling(window=20, min_periods=1).mean()", "depends": ["ret"], "dtype": "DataFrame", "return_result": "rolling_mean"},
        "rolling_std": {"category": "basic", "type": "rolling_std", "formula": "ret.rolling(window=20, min_periods=1).std(ddof=0).fillna(0)", "depends": ["ret"], "dtype": "DataFrame", "return_result": "rolling_std"},
        "rolling_var": {"category": "basic", "type": "rolling_var", "formula": "rolling_std ** 2", "depends": ["rolling_std"], "dtype": "Series", "return_result": "rolling_var"},
        "ts": {"category": "alpha", "type": "ts", "formula": "ret.rolling(window=5).mean().fillna(0)", "depends": ["ret"], "dtype": "Series", "return_result": "ts"},
        "xs": {"category": "alpha", "type": "xs", "formula": "ret.sub(ret.mean(axis=1), axis=0).div(ret.std(axis=1), axis=0)", "depends": ["ret"], "dtype": "Series", "return_result": "xs"},
        "pure": {"category": "alpha", "type": "pure", "formula": "ts + xs - xs.mean(axis=1).to_numpy()[:, None]", "depends": ["ts", "xs"], "dtype": "Series", "return_result": "pure"},
        "residual": {"category": "alpha", "type": "residual", "formula": "(ts.sub(ts.mean(axis=1), axis=0).div(ts.std(axis=1), axis=0))+(xs.sub(xs.mean(axis=1), axis=0).div(xs.std(axis=1), axis=0))", "depends": ["ts", "xs"], "dtype": "DataFrame", "return_result": "residual"},
        "cs_mean": {"category": "basic", "type": "cs_mean", "formula": "ret.mean(axis=1)", "depends": ["ret"], "dtype": "Series", "return_result": "cs_mean"},
        "cs_std": {"category": "basic", "type": "cs_std", "formula": "ret.std(axis=1).replace(0, 1e-9)", "depends": ["ret"], "dtype": "Series", "return_result": "cs_std"},
        "cs_rank": {"category": "basic", "type": "cs_rank", "formula": "ret.rank(axis=1, method='average')", "depends": ["ret"], "dtype": "DataFrame", "return_result": "cs_rank"},
        "cs_zscore": {"category": "basic", "type": "cs_zscore", "formula": "(ret.T - cs_mean).T.div(cs_std, axis=0)", "depends": ["ret", "cs_mean", "cs_std"], "dtype": "DataFrame", "return_result": "cs_zscore"},
        "clean_ret": {"category": "alpha", "type": "clean_ret", "formula": "pd.DataFrame(np.clip(future_ret, np.nanquantile(future_ret, 0.01, axis=0), np.nanquantile(future_ret, 0.99, axis=0)), index=future_ret.index, columns=future_ret.columns).fillna(0)", "depends": ["future_ret"], "dtype": "DataFrame", "return_result": "clean_ret"},
        "clean_benchmark": {"category": "alpha", "type": "clean_benchmark", "formula": "pd.Series(np.clip(benchmark, np.quantile(benchmark, 0.01), np.quantile(benchmark, 0.99)), index=benchmark.index)", "depends": ["benchmark"], "dtype": "Series", "return_result": "clean_benchmark"},
        "beta": {"category": "alpha", "type": "beta", "formula": "(clean_ret.sub(clean_ret.mean()).mul(clean_benchmark.sub(clean_benchmark.mean()), axis=0).mean()/clean_benchmark.sub(clean_benchmark.mean()).pow(2).mean())", "depends": ["clean_ret", "clean_benchmark"], "dtype": "Series", "return_result": "beta"},
        "alpha": {"category": "alpha", "type": "alpha", "formula": "clean_ret.mean() - beta * clean_benchmark.mean()", "depends": ["clean_ret", "beta", "clean_benchmark"], "dtype": "Series", "return_result": "alpha"},
        "fitted": {"category": "basic", "type": "fitted", "formula": "pd.DataFrame({s: alpha.loc[s] + beta.loc[s] * benchmark for s in alpha.index.intersection(beta.index)})", "depends": ["alpha", "beta", "benchmark"], "dtype": "DataFrame", "return_result": "fitted"},
        "sse": {"category": "basic", "type": "sse", "formula": "(residual.fillna(0)**2).sum()", "depends": ["residual"], "dtype": "Series", "return_result": "sse"},
        "sst": {"category": "basic", "type": "sst", "formula": "((future_ret - future_ret.mean())**2).fillna(0).sum()", "depends": ["future_ret"], "dtype": "Series", "return_result": "sst"},
        "ssr": {"category": "basic", "type": "ssr", "formula": "sst - sse", "depends": ["sst", "sse"], "dtype": "Series", "return_result": "ssr"},
        "r_squared": {"category": "basic", "type": "r_squared", "formula": "1 - (sse / (sst + 1e-12))", "depends": ["sse", "sst"], "dtype": "Series", "return_result": "r_squared"},
        "adj_r_squared": {"category": "basic", "type": "adj_r_squared", "formula": "1 - (1 - r_squared) * (n - 1) / (n - k - 1)", "depends": ["r_squared"], "dtype": "Series", "return_result": "adj_r_squared"},
        "mse": {"category": "basic", "type": "mse", "formula": "sse / n", "depends": ["sse"], "dtype": "Series", "return_result": "mse"},
        "rmse": {"category": "basic", "type": "rmse", "formula": "np.sqrt(mse)", "depends": ["mse"], "dtype": "Series", "return_result": "rmse"},
        "mae": {"category": "basic", "type": "mae", "formula": "mean(abs(residual))", "depends": ["residual"], "dtype": "Series", "return_result": "mae"},
        "X": {"category": "input", "type": "design_matrix", "builder": "build_X", "formula": "build_X", "depends": ["future_ret"], "dtype": "Matrix", "return_result": "X"},
        "stderr_beta": {"category": "basic", "type": "stderr_beta", "formula": "np.sqrt(mse * np.linalg.pinv(transpose_(X, X))[1, 1])", "depends": ["mse", "X"], "dtype": "Series", "return_result": "stderr_beta"},
        "stderr_alpha": {"category": "basic", "type": "stderr_alpha", "formula": "np.sqrt(mse * np.linalg.pinv(transpose_(X, X))[0, 0])", "depends": ["mse", "X"], "dtype": "Series", "return_result": "stderr_alpha"},
        "f_stat": {"category": "basic", "type": "f_stat", "formula": "(ssr / k) / (sse / (n - k - 1))", "depends": ["ssr", "sse"], "dtype": "Series", "return_result": "f_stat"},
        "tstat_beta": {"category": "risk", "type": "tstat_beta", "formula": "beta / stderr_beta", "depends": ["beta", "stderr_beta"], "dtype": "Series", "return_result": "tstat_beta"},
        "tstat_alpha": {"category": "risk", "type": "tstat_alpha", "formula": "alpha / stderr_alpha", "depends": ["alpha", "stderr_alpha"], "dtype": "Series", "return_result": "tstat_alpha"},
        "n_obs": {"category": "basic", "type": "n_obs", "formula": "count(ret)", "depends": ["ret"], "dtype": "int", "return_result": "n_obs"},
        "hit_ratio": {"category": "basic", "type": "hit_ratio", "formula": "(np.sign(preds) == np.sign(future_ret)).where(future_ret.notna()).mean()", "depends": ["preds", "future_ret"], "dtype": "Series", "return_result": "hit_ratio"},
        "preds": {"category": "basic", "type": "preds", "formula": "signal", "depends": ["signal"], "dtype": "array", "return_result": "preds"},
        "candidate_signal": {"category": "basic", "type": "candidate_signal", "formula": "0.4*ret.rolling(30).mean()+0.2*(price/price.rolling(100).mean()-1)-0.2*ret.rolling(20).std()+0.2*(volume/volume.rolling(30).mean())", "depends": ["ret", "price", "volume"], "dtype": "Series", "return_result": "candidate_signal"},
        "rank": {"category": "transform", "type": "rank", "formula": "ret.rank(axis=1, method='average')", "depends": ["ret"], "dtype": "DataFrame", "return_result": "rank"},
        "zscore": {"category": "transform", "type": "zscore", "formula": "ret.sub(ret.mean(axis=1), axis=0).div(ret.std(axis=1).replace(0, 1e-9), axis=0)", "depends": ["ret"], "dtype": "DataFrame", "return_result": "zscore"},
        "winsor": {"category": "transform", "type": "winsor", "formula": "ret.apply(lambda r: r.clip(r.quantile(0.05), r.quantile(0.95)), axis=1)", "depends": ["ret"], "dtype": "DataFrame", "return_result": "winsor"},
        "tanh": {"category": "transform", "type": "tanh", "formula": "np.tanh(ret)", "depends": ["ret"], "dtype": "DataFrame", "return_result": "tanh"},
        "detrend": {"category": "transform", "type": "detrend", "formula": "ret - rolling_mean", "depends": ["ret", "rolling_mean"], "dtype": "DataFrame", "return_result": "detrend"},
        "diff": {"category": "transform", "type": "diff", "formula": "ret.diff().fillna(0)", "depends": ["ret"], "dtype": "DataFrame", "return_result": "diff"},
        "lag": {"category": "transform", "type": "lag", "formula": "ret.shift(1).fillna(0)", "depends": ["ret"], "dtype": "DataFrame", "return_result": "lag"},
        "cumprod": {"category": "transform", "type": "cumprod", "formula": "(1 + ret.replace([np.inf, -np.inf], 0).fillna(0)).cumprod()", "depends": ["ret"], "dtype": "DataFrame", "return_result": "cumprod"},
        "volatility": {"category": "risk", "type": "volatility", "formula": "np.sqrt(sum((ret - mean(ret))**2) / len(ret))", "depends": ["ret"], "dtype": "Series", "return_result": "volatility"},
        "sharpe": {"category": "basic", "type": "sharpe", "formula": "mean(daily_ret)/std(daily_ret)*np.sqrt(252)", "depends": ["daily_ret"], "dtype": "Series", "return_result": "sharpe"},
        "reg_sharpe": {"category": "risk", "type": "reg_sharpe", "formula": "mean(ret_all) / (np.sqrt(252) * std(ret_all))", "depends": ["ret_all"], "dtype": "Series", "return_result": "reg_sharpe"},
        "equity": {"category": "risk", "type": "equity", "formula": "(1 + net_ret).cumprod()", "depends": ["net_ret"], "dtype": "DataFrame", "return_result": "equity"},
        "peak": {"category": "risk", "type": "peak", "formula": "equity.cummax()", "depends": ["equity"], "dtype": "Series", "return_result": "peak"},
        "drawdown": {"category": "risk", "type": "drawdown", "formula": "equity / peak - 1", "depends": ["equity", "peak"], "dtype": "DataFrame", "return_result": "drawdown"},
        "cvar": {"category": "risk", "type": "cvar", "formula": "ret[ret <= ret.quantile(0.05)].mean()", "depends": ["ret"], "dtype": "Series", "return_result": "cvar"},
        "q05": {"category": "risk", "type": "quantile", "formula": "ret.quantile(0.05)", "depends": ["ret"], "dtype": "Series", "return_result": "q05"},
        "cvar_95": {"category": "risk", "type": "cvar", "formula": "ret[ret <= q05].mean()", "depends": ["ret", "q05"], "dtype": "Series", "return_result": "cvar_95"},
        "decorrelation": {"category": "risk", "type": "decorrelation", "formula": "1 - np.mean(np.abs(corr_rm))", "depends": ["corr_rm"], "dtype": "float", "return_result": "decorrelation"},
        "wiggle": {"category": "risk", "type": "wiggle", "formula": "np.mean(np.abs(np.diff(corr_rm)))", "depends": ["corr_rm"], "dtype": "float", "return_result": "wiggle"},
        "max_drawdown": {"category": "risk", "type": "max_drawdown", "formula": "drawdown.min()", "depends": ["drawdown"], "dtype": "Series", "return_result": "max_drawdown"},
        "alpha_ts": {"category": "alpha", "type": "ts", "formula": "0.2*ret.rolling(10).mean()+0.5*ret.rolling(30).mean()+0.3*ret.rolling(90).mean()", "depends": ["ret"], "dtype": "Series", "return_result": "alpha_ts"},
        "alpha_xs": {"category": "alpha", "type": "xs", "formula": "ret.sub(cs_mean, axis=0)", "depends": ["ret", "cs_mean"], "dtype": "DataFrame", "return_result": "alpha_xs"},
        "alpha_pure": {"category": "alpha", "type": "pure", "formula": "alpha_ts+alpha_xs", "depends": ["alpha_ts", "alpha_xs"], "dtype": "DataFrame", "return_result": "alpha_pure"},
        "beta_alpha": {"category": "alpha", "type": "beta", "formula": "cov_rm/var_ret", "depends": ["cov_rm", "var_ret"], "dtype": "float", "return_result": "beta_alpha"},
        "inv_vol": {"category": "portfolio", "type": "inv_vol", "formula": "1/std_ret", "depends": ["std_ret"], "dtype": "Series", "return_result": "inv_vol"},
        "risk_parity": {"category": "portfolio", "type": "risk_parity", "formula": "(1/ret.rolling(20).std()).div((1/ret.rolling(20).std()).sum(axis=1), axis=0)", "depends": ["ret"], "dtype": "Series", "return_result": "risk_parity"},
        "kelly": {"category": "portfolio", "type": "kelly", "formula": "(ret.mean()/ret.var()*0.25).clip(-2, 2)", "depends": ["mean_ret", "var_ret"], "dtype": "Series", "return_result": "kelly"},
        "mvo": {"category": "portfolio", "type": "mvo", "formula": "mean_ret/var_ret", "depends": ["mean_ret", "var_ret"], "dtype": "Series", "return_result": "mvo"},
        "entropy": {"category": "portfolio", "type": "entropy", "formula": "-sum((weight) * (log(np.abs(weight) + 1e-12)))", "depends": ["weight"], "dtype": "float", "return_result": "entropy"},
        "risk_aversion": {"category": "parameter", "type": "risk_aversion", "formula": "1.0", "depends": [], "dtype": "float", "return_result": "risk_aversion"},
        "ret_centered": {"category": "parameter", "type": "ret_centered", "formula": "ret - ret.mean(axis=0)", "depends": ["ret"], "dtype": "DataFrame"},
        "cov_matrix": {"category": "basic", "type": "cov_matrix", "formula": "(ret_centered.T @ ret_centered) / (ret_centered.shape[0] - 1)", "depends": ["ret_centered"], "dtype": "DataFrame", "return_result": "cov_matrix"},
        "weight": {"category": "portfolio", "type": "weight", "formula": "pd.Series(np.linalg.solve(cov_matrix.values + 0.1*np.eye(len(cov_matrix)), ret.mean(axis=0).values), index=cov_matrix.columns)", "depends": ["cov_matrix", "ret", "risk_aversion"], "dtype": "Series", "return_result": "weight"},
        "norm_weight": {"category": "basic", "type": "norm_weight", "formula": "weight.div(weight.abs().sum()+1e-12)", "depends": ["weight"], "dtype": "Series", "return_result": "norm_weight"},
        "executed_weight": {"category": "portfolio", "type": "executed_weight", "formula": "pd.DataFrame([norm_weight]*len(ret), index=ret.index, columns=ret.columns).shift(1).fillna(0)", "depends": ["norm_weight", "ret"], "dtype": "DataFrame", "return_result": "executed_weight"},
        "strategy_ret": {"category": "portfolio", "type": "strategy_ret", "formula": "(ret.mul(executed_weight, axis=1)).sum(axis=1)", "depends": ["executed_weight", "ret"], "dtype": "DataFrame", "return_result": "strategy_ret"},
        "regime": {"category": "market_structure", "type": "regime", "formula": "pd.Series(np.where(ret.rolling(20).std().mean(axis=1)>ret.rolling(20).std().mean(axis=1).quantile(.75), 'HIGH_VOL', np.where(price.mean(axis=1)>price.mean(axis=1).rolling(100).mean(), 'BULL', 'BEAR')), index=ret.index)", "depends": ["ret", "price"], "dtype": "Series", "return_result": "regime"},
        "liquidity_adj_vol": {"category": "market_structure", "type": "liquidity_adj_vol", "formula": "daily_ret.rolling(20).std() / np.log(volume.mean(axis=1)+1)",
    "depends": ["daily_ret", "volume"], "dtype": "Series", "return_result": "liquidity_adj_vol"},
        "slippage": {"category": "execution", "type": "slippage", "formula": "0.5*(mkt_price_high-mkt_price_low)/mkt_price", "depends": ["mkt_price", "mkt_price_high", "mkt_price_low"], "dtype": "Series", "return_result": "slippage"},
        "impact": {"category": "execution", "type": "impact", "formula": "slippage*(norm_weight.abs()**2).sum()", "depends": ["slippage", "norm_weight"], "dtype": "Series", "return_result": "impact"}, 
        "turnover": {"category": "execution", "type": "turnover", "formula": "executed_weight.sub(executed_weight.shift(1)).abs().sum(axis=1)", "depends": ["executed_weight"], "dtype": "Series", "return_result": "turnover"},
        "pturnover": {"category": "execution", "type": "pturnover", "formula": "executed_weight.diff().abs().sum(axis=1)", "depends": ["executed_weight"], "dtype": "Series", "return_result": "pturnover"},
        "transaction_cost": {"category": "execution", "type": "transaction_cost", "formula": "pturnover *(0.001 + ret.rolling(20, min_periods=10).std().mean(axis=1).fillna(0))*TCOST/100.0", "depends": ["pturnover", "ret"], "dtype": "Series", "return_result": "transaction_cost"},
        "net_ret": {"category": "portfolio", "type": "net_ret", "formula": "strategy_ret.sub(transaction_cost, axis=0)", "depends": ["strategy_ret", "transaction_cost"], "dtype": "DataFrame", "return_result": "net_ret"},
        "daily_ret": {"category": "portfolio", "type": "daily_ret", "formula": "net_ret.copy()", "depends": ["net_ret"], "dtype": "DataFrame", "return_result": "daily_ret"},

        "future_ret": {"category": "intel", "type": "future_ret", "formula": "ret.shift(-1)", "depends": ["ret"], "dtype": "DataFrame", "return_result": "future_ret"},
        "ic": {"category": "intel", "type": "ic", "formula": "candidate_signal.corrwith(future_ret)", "depends": ["candidate_signal", "future_ret"], "dtype": "Series", "return_result": "ic"},
        "portfolio_signal": {"category": "portfolio", "type": "portfolio_signal", "formula": "0.4*risk_parity.div(risk_parity.mean(axis=1), axis=0)+0.3*kelly.div(kelly.abs().mean(), axis=0)+0.3*inv_vol.div(inv_vol.mean(axis=1), axis=0)", "depends": ["risk_parity", "kelly", "inv_vol"], "dtype": "Series", "return_result": "portfolio_signal"},
        "score": {"category": "decision", "type": "score", "formula": "np.nanmean(pure.to_numpy(), axis=0)* 0.4 + np.nanmean(ic.to_numpy(), axis=0) * 0.2 + -np.nanmean(cvar.to_numpy(), axis=0) * 0.1 + np.nanmean(portfolio_signal.to_numpy(), axis=0) * 0.3", "depends": ["pure", "sharpe", "ic", "cvar", "portfolio_signal"], "dtype": "Series", "return_result": "score"},
        "signal": {"category": "decision", "type": "signal", "formula": "future_ret", "depends": ["future_ret"], "dtype": "Series", "return_result": "signal"},
        "psignal": {"category": "decision", "type": "psignal", "formula": "np.where((m := np.mean(np.where((score + future_ret - 0.41) <= 0.04, 0, np.where((score + future_ret - 0.41) <= 0.06, 1, 2)), axis=0)) < 0.5, 'BUY', np.where(m < 1.5, 'HOLD', 'SELL'))", "depends": ["score", "future_ret"], "dtype": "Series", "return_result": "psignal"},
        "strategy_returns": {"category": "portfolio", "type": "strategy_returns", "formula": "signal.shift(1)*ret", "depends": ["signal", "ret"], "dtype": "Series", "return_result": "strategy_returns"},
        "benchmark": {"category": "market", "type": "benchmark", "formula": "(ret['BTCUSDT'] if 'BTCUSDT' in ret.columns else (ret['SPY'] if 'SPY' in ret.columns else ret.mean(axis=1)))", "depends": ["ret"], "dtype": "Series", "return_result": "benchmark"},
        "reg_beta": {"category": "regression", "type": "reg_beta", "depends": {"y": "ret", "X": ["mkt_ret"]}, "transforms": {}, "formula": "_ols(y=ret, X=[mkt_ret]).beta", "dtype": "float", "return_result": "reg_beta"},
        "reg_alpha": {"category": "regression", "type": "reg_alpha", "depends": {"y": "ret", "X": ["mkt_ret"]}, "transforms": {}, "formula": "_ols(y=ret, X=[mkt_ret]).alpha", "dtype": "float", "return_result": "reg_alpha"},
        "reg_r2": {"category": "regression", "type": "reg_r2", "depends": {"y": "ret", "X": ["mkt_ret"]}, "transforms": {}, "formula": "np.nanmean(_ols(y=ret, X=[mkt_ret]).r2)", "dtype": "float", "return_result": "reg_r2"},
        "reg_residual": {"category": "regression", "type": "reg_residual", "depends": {"y": "ret", "X": ["mkt_ret"]}, "transforms": {}, "formula": "_ols(y=ret, X=[mkt_ret]).residuals", "dtype": "Series", "return_result": "reg_residual"},
        "reg_tstat_alpha": {"category": "regression", "type": "reg_tstat_alpha", "depends": {"y": "ret", "X": ["mkt_ret"]}, "transforms": {}, "formula": "np.nanmean(_ols(y=ret, X=[mkt_ret]).alpha_tstat)", "dtype": "float", "return_result": "reg_tstat_alpha"},
        "reg_fitted": {"category": "regression", "type": "reg_fitted", "depends": {"y": "ret", "X": ["mkt_ret"]}, "transforms": {}, "formula": "_ols(y=ret, X=[mkt_ret]).fitted", "dtype": "DataFrame", "return_result": "reg_fitted"},
        "reg_tstat_beta": {"category": "regression", "type": "reg_tstat_beta", "depends": {"y": "ret", "X": ["mkt_ret"]}, "transforms": {}, "formula": "np.nanmean(_ols(y=ret, X=[mkt_ret]).tstat_beta)", "dtype": "float", "return_result": "reg_tstat_beta"},
        "reg_beta_multi": {"category": "regression", "type": "reg_beta_multi", "depends": {"y": "ret", "X": ["mkt_ret", "alpha"]}, "transforms": {}, "formula": "_ols(y=ret, X=[mkt_ret, transpose(alpha)]).beta", "dtype": "Series", "return_result": "reg_beta_multi"},
        "reg_adj_r2": {"category": "regression", "type": "reg_adj_r2", "depends": ["adj_r_squared"], "transforms": {}, "formula": "adj_r_squared", "dtype": "float", "return_result": "reg_adj_r2"},
        "reg_mse": {"category": "regression", "type": "reg_mse", "depends": ["mse"], "transforms": {}, "formula": "mse", "dtype": "float", "return_result": "reg_mse"},
        "reg_rmse": {"category": "regression", "type": "reg_rmse", "depends": ["rmse"], "transforms": {}, "formula": "rmse", "dtype": "float", "return_result": "reg_rmse"},
        "reg_return": {"category": "risk", "type": "reg_return", "depends": ["strategy_returns"], "transforms": {}, "formula": "strategy_returns.mean()", "dtype": "float", "return_result": "reg_return"},
        "reg_total_return": {"category": "risk", "type": "reg_total_return", "depends": ["strategy_returns"], "transforms": {}, "formula": "(1 + strategy_returns).prod() - 1", "dtype": "float", "return_result": "reg_total_return"},
        "reg_volatility": {"category": "risk", "type": "reg_volatility", "depends": ["strategy_returns"], "transforms": {}, "formula": "strategy_returns.std()", "dtype": "float", "return_result": "reg_volatility"},
        "reg_sharpe": {"category": "risk", "type": "reg_sharpe", "depends": ["strategy_returns"], "transforms": {}, "formula": "np.nanmean(strategy_returns.mean() / strategy_returns.std())", "dtype": "float", "return_result": "reg_sharpe"},
        "reg_max_drawdown": {"category": "risk", "type": "reg_max_drawdown", "depends": ["drawdown"], "formula": "np.nanmean(np.abs(np.nanmin(drawdown)))", "dtype": "float", "return_result": "reg_max_drawdown"},
        "reg_cvar": {"category": "risk", "type": "reg_cvar", "depends": ["cvar"], "transforms": {}, "formula": "np.nanmean(cvar)", "dtype": "float", "return_result": "reg_cvar"},
        "reg_cagr": {"category": "risk", "type": "reg_cagr", "depends": ["strategy_returns"], "transforms": {}, "formula": "(1 + strategy_returns).prod() ** (252 / len(strategy_returns)) - 1", "dtype": "float", "return_result": "reg_cagr"},
        "reg_turnover": {"category": "execution", "type": "reg_turnover", "depends": ["turnover"], "transforms": {}, "formula": "np.nanmean(turnover)", "dtype": "float", "return_result": "reg_turnover"},
        "reg_slippage": {"category": "execution", "type": "reg_slippage", "depends": ["slippage"], "transforms": {}, "formula": "np.nanmean(slippage)", "dtype": "float", "return_result": "reg_slippage"},
        "reg_impact": {"category": "execution", "type": "reg_impact", "depends": ["impact"], "transforms": {}, "formula": "np.nanmean(impact)", "dtype": "float", "return_result": "reg_impact"},
        "reg_ic": {"category": "intel", "type": "reg_ic", "depends": {"y": "ret", "X": ["signal"]}, "transforms": {}, "formula": "np.nanmean(_ols(y=ret, X=[signal]).beta)", "dtype": "float", "return_result": "reg_ic"},
        "reg_signal_strength": {"category": "regression", "type": "reg_signal_strength", "depends": ["reg_beta"], "transforms": {}, "formula": "reg_beta", "dtype": "float", "return_result": "reg_signal_strength"},
        "reg_hit_ratio": {"category": "performance", "type": "reg_hit_ratio", "formula": "np.nanmean(hit_ratio)", "depends": ["hit_ratio"], "dtype": "float", "return_result": "reg_hit_ratio"},
        "reg_score": {"category": "decision", "type": "reg_score", "depends": ["score", "reg_sharpe", "reg_r2", "reg_hit_ratio", "reg_ic", "reg_tstat_alpha", "reg_tstat_beta", "reg_max_drawdown"], "transforms": {}, "formula": "score + reg_sharpe + reg_r2 + reg_hit_ratio + reg_ic + reg_tstat_alpha + (-1) * reg_max_drawdown", "dtype": "float", "return_result": "reg_score"}
    }

# =================================================================
# END OF FORMULAOUTPUT
# =================================================================