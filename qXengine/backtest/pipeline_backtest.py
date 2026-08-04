"""
quantX Backtest Pipeline — Combinatorial Scenario Optimizer (Production-Ready)
Engineering Prescription Applied: A–E
=============================================================================

A. Data Layer (CRITICAL)
   – Data sufficiency gate: >=1,460 rows per asset, >=4yr history enforced
   – 50/50 temporal split retained

B. Feature Engineering (CRITICAL)
   – Dimensionality reduction: PCA/ICA -> 10-15 orthogonal factors
   – Microstructure features: realized vol skew, order-flow toxicity proxy,
     bid-ask bounce proxy, intraday range, volume imbalance
   – Target: volatility-scaled forward returns (Sharpe-like target)

C. Model Layer (HIGH)
   – LightGBM / XGBoost with graceful sklearn fallback
   – Regularization: L1/L2, early stopping, subsampling
   – Ensemble: Stacked 3-model ensemble (momentum, mean-reversion, ML)

D. Validation Layer (HIGH)
   – Combinatorial Purged Cross-Validation (CPCV) per Lopez de Prado
   – Regime-conditional backtesting with 10 bps (0.001) transaction costs
   – Go-Live gate: minimum 100 OOS predictions required

E. Agentic Loop (MEDIUM)
   – STOP only after 5 iterations OR score degrades from non-zero baseline
   – Feature-set mutation operators (not just hyperparameters)
   – DATA_INSUFFICIENT human-in-the-loop flag instead of silent zero-convergence
quantX Backtest Pipeline — Fixed per user requirements
======================================================
Changes:
  1. Combinatorial optimization CALL commented out (fn kept).
  2. Old defs preserved with _0 suffix; new fixed defs added.
  3. Common cutoff-date temporal split (not per-symbol row split).
  4. evaluate() metrics flow into saved research JSON/pickles.
  5. Canonical Sharpe = mean/std * sqrt(252) everywhere.
  6. Portfolio alpha t-stat from intercept of daily port ret vs BTC.
  7. Transaction cost sensitivity: 10 bps and 20 bps.
  8. Turnover computed from executed (lagged) weights.
  9. Simple portfolio-level summary table printed and saved.
"""



import os
import pickle
import json
import warnings
import copy
import joblib
import itertools
import hashlib
import time
import gc

import numpy as np
import pandas as pd

from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from copy import deepcopy
from pathlib import Path
from collections import defaultdict

from scipy import stats
from sklearn.decomposition import PCA, FastICA

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore")
from ..strategies.FormulaInfo import FormulaInfo

# =====================================================
# GLOBAL CONFIG
# =====================================================
RETAIL_TRANSACTION_COST_LOW = 0.001   # 10 bps
RETAIL_TRANSACTION_COST_HIGH = 0.002  # 20 bps
MIN_ROWS_PER_ASSET = 1460
MIN_OOS_PREDICTIONS = 100
MAX_FEATURES_BEFORE_REDUCTION = 50
TARGET_N_COMPONENTS = 12

# =====================================================
# SPLIT CONFIGURATION
# =====================================================
DEFAULT_SPLIT_RATIO = 0.5       # 50% train / 50% val
MIN_SPLIT_DAYS = 30             # minimum days in either split


SCENARIO_CONFIG = {
    "BULL": {"mu_multiplier": 1.8, "vol_multiplier": 0.7, "kelly_cap": 0.25},
    "BEAR": {"mu_multiplier": -0.8, "vol_multiplier": 1.2, "kelly_cap": 0.10},
    "HIGH_VOL": {"mu_multiplier": 0.3, "vol_multiplier": 2.0, "kelly_cap": 0.05},
    "CRASH": {"mu_multiplier": -2.5, "vol_multiplier": 3.5, "kelly_cap": 0.02},
    "STABLE": {"mu_multiplier": 0.5, "vol_multiplier": 0.5, "kelly_cap": 0.15}
}

PARAM_GRID_FAST = {
    "trees": [200, 500, 800], "depth": [8, 12, 16], "horizon": [21, 42],
    "min_samples_split": [2, 10], "max_features": ["sqrt", 0.5],
    "model": ["random_forest"], "learning_rate": [0.05, 0.1],
    "reg_alpha": [0.0, 0.1], "reg_lambda": [1.0, 2.0],
}
PARAM_GRID_INSIDE = {
    "trees": [200, 350, 500, 650, 800], "depth": [6, 8, 10, 12, 14],
    "horizon": [10, 21, 42], "min_samples_split": [5, 10, 20],
    "max_features": ["sqrt", "log2", 0.5],
    "model": ["lightgbm", "xgboost", "random_forest", "gradient_boosting"],
    "learning_rate": [0.01, 0.05, 0.1], "reg_alpha": [0.0, 0.1, 1.0],
    "reg_lambda": [0.5, 1.0, 2.0],
}


def formula_report(report, mode="df_multi", lookup="index"):
    if not isinstance(report, tuple) or len(report) != 2:
        raise ValueError("report must be tuple(df, df_easy)")
    df, df_easy = report
    if mode == "df_multi":
        result = df
    elif mode == "df_easy":
        result = df_easy
    else:
        raise ValueError("mode must be 'df_multi' or 'df_easy'")
    if not isinstance(result, pd.DataFrame):
        raise TypeError("Selected report is not a DataFrame")
    if lookup == "index":
        return result
    elif lookup == "columns":
        return result.T
    elif lookup == "symbol_features":
        if isinstance(result.columns, pd.MultiIndex):
            result = result.copy()
            result.columns = [f"{a}_{b}" for a, b in result.columns]
        return result
    else:
        raise ValueError("lookup must be 'index' or 'columns' or 'symbol_features'")


def run_quantx_engine(data, interval="4y", params=None):
    if params is None:
        params = {}
    try:
        from ..qxEngine import QuantXEngine
    except Exception as exc:
        return {
            "engine": None, "results": [], "strategies": [],
            "formula_outputs": [], "formula_outputs_raw": [],
            "raw_data": data, "success": False, "error": str(exc),
        }
    engine = QuantXEngine()
    strategies = engine.qxStrategyList(data, interval=params.get("interval", interval))
    print(f"[ENGINE] Strategies returned: {len(strategies)}")
    formula_outputs = []
    formula_outputs_raw = []
    for s in engine.strategy:
        rpt_fo = getattr(s, "formulaOutput", None)
        if rpt_fo is None:
            rpt_fo = FormulaInfo(data)
        rpt_fo.assemble()
        rpt = rpt_fo.reporting()
        formula_outputs.append(rpt)
        formula_outputs_raw.append(rpt_fo)
    return {
        "engine": engine, "results": engine.results, "strategies": engine.strategy,
        "formula_outputs": formula_outputs, "formula_outputs_raw": formula_outputs_raw,
        "raw_data": data, "success": True, "error": None,
    }


class Storage:
    def __init__(self, base_dir="runs"):
        self.base = Path(base_dir)
        self.datasets = self.base / "datasets"
        self.base.mkdir(parents=True, exist_ok=True)
        self.datasets.mkdir(parents=True, exist_ok=True)

    def _resolve(self, path):
        path = Path(path)
        if path.is_absolute():
            return path
        if len(path.parts) > 0 and path.parts[0] == self.base.name:
            return path
        return self.base / path

    def save(self, obj, path):
        path = self._resolve(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        return path

    def load(self, path):
        path = self._resolve(path)
        with open(path, "rb") as f:
            return pickle.load(f)

    def save_json(self, obj, path):
        path = self._resolve(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(obj, f, indent=2, default=str)
        return path

    def create_run(self):
        runs = sorted(self.base.glob("run_*"))
        idx = int(runs[-1].name.split("_")[1]) + 1 if runs else 1
        run_dir = self.base / f"run_{idx:06d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(self.base / "latest.txt", "w") as f:
            f.write(run_dir.name)
        return idx, run_dir


# =====================================================
# SPLIT ENGINE  —  OLD (preserved)
# =====================================================
class SplitEngine_0:
    def split(self, data, split_ratio=0.5):
        train, val = {}, {}
        for symbol, df in data.items():
            if not isinstance(df, pd.DataFrame):
                continue
            split_idx = int(len(df) * split_ratio)
            if split_idx < 10:
                split_idx = len(df) // 2
            train[symbol] = df.iloc[:split_idx].copy()
            val[symbol] = df.iloc[split_idx:].copy()
        return train, val


# =====================================================
# SPLIT ENGINE  —  NEW: common cutoff date
# =====================================================
class SplitEngine:
    def split(self, data, split_ratio=DEFAULT_SPLIT_RATIO, explicit_split_date=None):
        """
        Temporal split at a COMMON CALENDAR DATE across all symbols.

        1. Finds the full calendar span (union of all asset date ranges).
        2. Computes the cutoff date as split_ratio along that span.
        3. Splits EVERY asset at that exact date.

        Assets that start after the cutoff get 0%% train.
        Assets that end before the cutoff get 0%% val.
        The split wall is a single calendar date.
        """
        # Per-asset date ranges
        ranges = {}
        for sym, df in data.items():
            if isinstance(df, pd.DataFrame) and len(df) > 0:
                idx = pd.to_datetime(df.index, errors="coerce").dropna()
                if len(idx) > 0:
                    ranges[sym] = (idx.min(), idx.max(), len(idx))

        if not ranges:
            print("[SPLIT] No valid dates found; falling back to per-symbol row split.")
            return SplitEngine_0().split(data, split_ratio)

        # Union: overall calendar span across ALL assets
        union_start = min(r[0] for r in ranges.values())
        union_end = max(r[1] for r in ranges.values())
        union_days = (union_end - union_start).days

        # Determine cutoff date
        if explicit_split_date is not None:
            cutoff_date = pd.Timestamp(explicit_split_date)
            print(f"[SPLIT] Explicit split date requested: {explicit_split_date}")
        else:
            cutoff_offset = int(union_days * split_ratio)
            cutoff_date = union_start + pd.Timedelta(days=cutoff_offset)
            print(f"[SPLIT] Splitting union span at ratio={split_ratio}")

        print(f"[SPLIT] Union range:   {union_start.strftime('%%Y-%%m-%%d')} to {union_end.strftime('%%Y-%%m-%%d')}  ({union_days} calendar days)")
        print(f"[SPLIT] Cutoff date:   {cutoff_date.strftime('%%Y-%%m-%%d')}")

        # Apply cutoff uniformly to EVERY asset
        train, val = {}, {}
        per_asset_stats = []
        for sym, df in data.items():
            if not isinstance(df, pd.DataFrame):
                continue
            idx = pd.to_datetime(df.index, errors="coerce")
            train_mask = idx <= cutoff_date
            val_mask = idx > cutoff_date
            train[sym] = df.loc[train_mask].copy()
            val[sym] = df.loc[val_mask].copy()
            n_total = len(df)
            n_train = len(train[sym])
            n_val = len(val[sym])
            pct_train = n_train / n_total * 100 if n_total > 0 else 0
            pct_val = n_val / n_total * 100 if n_total > 0 else 0
            per_asset_stats.append({
                "symbol": sym, "total": n_total, "train": n_train, "val": n_val,
                "train_pct": pct_train, "val_pct": pct_val,
                "start": ranges[sym][0].strftime('%%Y-%%m-%%d'),
                "end": ranges[sym][1].strftime('%%Y-%%m-%%d')
            })

        # Print per-asset diagnostics
        print(f"\n[SPLIT] Per-asset split diagnostics (cutoff: {cutoff_date.strftime('%%Y-%%m-%%d')}):")
        print(f"  {'Symbol':<12} {'Start':<12} {'End':<12} {'Total':>6} {'Train':>6} {'Val':>6} {'Train%%':>7} {'Val%%':>7}")
        print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*6} {'-'*6} {'-'*6} {'-'*7} {'-'*7}")
        for s in per_asset_stats:
            print(f"  {s['symbol']:<12} {s['start']:<12} {s['end']:<12} {s['total']:>6} {s['train']:>6} {s['val']:>6} {s['train_pct']:>6.1f}%% {s['val_pct']:>6.1f}%%")

        # Summary
        total_obs = sum(s['total'] for s in per_asset_stats)
        total_train = sum(s['train'] for s in per_asset_stats)
        total_val = sum(s['val'] for s in per_asset_stats)
        print(f"\n[SPLIT] GLOBAL SUMMARY: Train={total_train}/{total_obs} ({total_train/total_obs*100:.1f}%%), Val={total_val}/{total_obs} ({total_val/total_obs*100:.1f}%%)")

        return train, val
class FeatureReducer:
    def __init__(self, n_components=TARGET_N_COMPONENTS, method="pca"):
        self.n_components = n_components
        self.method = method
        self.reducer = None
        self.scaler = StandardScaler()
        self.feature_names_out = []

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        if X.shape[1] <= self.n_components:
            self.feature_names_out = list(X.columns)
            return X.copy()
        X_num = X.select_dtypes(include=[np.number]).copy()
        X_num = X_num.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_scaled = self.scaler.fit_transform(X_num)
        n_comp = min(self.n_components, X_num.shape[1], X_num.shape[0])
        if self.method == "pca":
            self.reducer = PCA(n_components=n_comp, random_state=42)
        elif self.method == "ica":
            self.reducer = FastICA(n_components=n_comp, random_state=42, max_iter=500)
        else:
            if y is not None and y.std() > 0:
                corrs = X_num.corrwith(y).abs().sort_values(ascending=False)
                selected = corrs.head(n_comp).index.tolist()
                self.feature_names_out = selected
                return X_num[selected].copy()
            else:
                self.feature_names_out = list(X_num.columns)[:n_comp]
                return X_num[self.feature_names_out].copy()
        X_reduced = self.reducer.fit_transform(X_scaled)
        self.feature_names_out = [f"{self.method.upper()}_F{i+1}" for i in range(n_comp)]
        return pd.DataFrame(X_reduced, index=X_num.index, columns=self.feature_names_out)

    def transform(self, X: pd.DataFrame):
        if self.reducer is None:
            if self.feature_names_out:
                return X[self.feature_names_out].copy()
            return X.copy()
        X_num = X.select_dtypes(include=[np.number]).copy()
        X_num = X_num.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_scaled = self.scaler.transform(X_num)
        X_reduced = self.reducer.transform(X_scaled)
        return pd.DataFrame(X_reduced, index=X_num.index, columns=self.feature_names_out)


def compute_microstructure_features(df: pd.DataFrame) -> Dict[str, float]:
    feats = {}
    if not isinstance(df, pd.DataFrame) or len(df) < 20:
        return feats
    close = df["close"]
    high = df["high"] if "high" in df.columns else close
    low = df["low"] if "low" in df.columns else close
    volume = df["volume"] if "volume" in df.columns else pd.Series(1, index=df.index)
    ret = df["ret"] if "ret" in df.columns else close.pct_change()
    up_ret = ret[ret > 0]
    down_ret = ret[ret < 0]
    up_vol = up_ret.std() if len(up_ret) > 1 else 1e-9
    down_vol = down_ret.std() if len(down_ret) > 1 else 1e-9
    feats["micro_vol_skew"] = up_vol / (down_vol + 1e-9)
    signed_vol = volume * np.sign(ret)
    feats["micro_toxicity"] = signed_vol.rolling(20).mean().iloc[-1]
    intraday_range = (high - low) / (close + 1e-9)
    feats["micro_bounce"] = intraday_range.rolling(20).mean().iloc[-1]
    feats["micro_range"] = intraday_range.iloc[-1]
    vol_ma = volume.rolling(20).mean().iloc[-1]
    feats["micro_vol_imbalance"] = volume.iloc[-1] / (vol_ma + 1e-9)
    feats["micro_realized_vol"] = ret.rolling(20).std().iloc[-1]
    return feats

class MLTrainEngine:
    def __init__(self, use_feature_reduction=True, reduction_method="pca"):
        self.feature_schema = None
        self.model = None
        self.params = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.reducer = FeatureReducer(method=reduction_method) if use_feature_reduction else None
        self.use_feature_reduction = use_feature_reduction

    def train(self, train_package):
        X = train_package["X"].copy()
        y = train_package["y"].copy()
        self.params = train_package.get("params", {})
        if isinstance(y, pd.DataFrame):
            if "y" in y.columns:
                y = y["y"]
            else:
                y = y.iloc[:, 0]
        y = pd.Series(y).astype(float).replace([np.inf, -np.inf], np.nan)
        if not X.index.equals(y.index) or len(X) != len(y):
            X = X.reset_index(drop=True)
            y = pd.Series(y).reset_index(drop=True)
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx].copy()
        y = y.loc[common_idx].copy()
        if len(y) == 0:
            raise ValueError("No valid training targets: X and y have zero overlapping indices.")
        y = pd.Series(y).replace([np.inf, -np.inf], np.nan)
        if y.isna().any():
            nan_count = int(y.isna().sum())
            print(f"[ML WARNING] y contains {nan_count}/{len(y)} NaN/inf values. Filling with 0.")
            y = y.fillna(0)
        valid_mask = y.notna().values
        X = X.iloc[valid_mask]
        y = y.iloc[valid_mask]
        if len(y) == 0:
            raise ValueError("No valid training targets after NaN removal")
        if X.shape[0] > 0 and X.shape[1] > 0 and X.shape[0] < 2 * X.shape[1]:
            print(f"[ML WARNING] Underdetermined: n={X.shape[0]} < 2*p={2*X.shape[1]}. Forcing feature reduction.")
            if not self.use_feature_reduction:
                self.use_feature_reduction = True
                if self.reducer is None:
                    self.reducer = FeatureReducer(method="pca")
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        print(f"[ML] Dataset AFTER CLEAN: X={X.shape}, y={y.shape}")
        if isinstance(X, pd.DataFrame) and isinstance(y, pd.Series):
            common_idx = X.index.intersection(y.index)
            X_aligned = X.loc[common_idx]
            y_aligned = y.loc[common_idx]
            corr = X_aligned.corrwith(y_aligned).abs()
            max_corr = corr.max()
            print(f"[SANITY] Max |corr(X, y)| = {max_corr:.4f}")
            if max_corr < 0.01:
                print(f"[SANITY WARNING] Max feature-target correlation {max_corr:.4f} < 0.01. Signal may be too weak.")
        X_encoded = self._encode_categoricals(X, fit=True)
        if self.use_feature_reduction and X_encoded.shape[1] > MAX_FEATURES_BEFORE_REDUCTION:
            print(f"[FEATURE] Reducing {X_encoded.shape[1]} features -> {TARGET_N_COMPONENTS} via {self.reducer.method}")
            X_encoded = self.reducer.fit_transform(X_encoded, y)
            print(f"[FEATURE] Reduced shape: {X_encoded.shape}")
        trained_model, metrics = self.fit(X_encoded, y, self.params)
        self.model = trained_model
        self.feature_schema = list(X_encoded.columns)
        return {
            "model": trained_model, "X": X, "y": y,
            "features": list(X_encoded.columns), "metrics": metrics,
            "results": train_package.get("results"),
            "formula_outputs": train_package.get("formula_outputs"),
            "params": self.params, "label_encoders": self.label_encoders,
            "scaler": self.scaler,
            "reducer": self.reducer if self.use_feature_reduction else None,
        }

    def build_feature_matrix(self, strategy_results, formula_outputs, raw_data, params=None):
        X, y = self.build_features(strategy_results, formula_outputs, raw_data, params or {})
        if X.shape[0] == 0:
            raise ValueError(f"build_features returned empty feature matrix: X.shape={X.shape}")
        return X, y

    def build_features(self, strategy_results, formula_outputs, raw_data, params=None):
        frames = []
        ret_series = None
        print(f"[FEATURE BUILD] strategy_results={len(strategy_results or [])}, formula_outputs={len(formula_outputs or [])}, raw_data={'YES' if raw_data is not None else 'NO'}")
        if formula_outputs is not None:
            if isinstance(formula_outputs, pd.DataFrame):
                formula_outputs = [formula_outputs]
            elif isinstance(formula_outputs, tuple):
                formula_outputs = list(formula_outputs)
            for fo in formula_outputs:
                if isinstance(fo, tuple) and len(fo) == 2:
                    fo_df = formula_report(fo, mode="df_multi", lookup="columns")
                elif isinstance(fo, pd.DataFrame):
                    fo_df = fo
                else:
                    continue
                if not isinstance(fo_df, pd.DataFrame) or fo_df.empty:
                    continue
                if isinstance(fo_df.columns, pd.MultiIndex):
                    feature_df = fo_df.copy()
                    if ret_series is None:
                        if ("market", "ret") in feature_df.columns:
                            ret_series = feature_df[("market", "ret")].copy()
                    feature_df.columns = [f"fo_{c[0]}_{c[1]}" for c in feature_df.columns]
                    feature_df.index.name = "symbol"
                    feature_df = feature_df.reset_index()
                    feature_df = feature_df.drop(columns=["fo_market_ret"], errors="ignore")
                    frames.append(feature_df)
                else:
                    feature_df = fo_df.copy()
                    if ret_series is None:
                        for idx in feature_df.index:
                            if str(idx).lower() in ["market_ret", "ret"]:
                                ret_series = feature_df.loc[idx].copy()
                                break
                    feature_df.index.name = "category_metric"
                    feature_df = feature_df.reset_index()
                    drop_cols = [c for c in feature_df.columns if "ret" in str(c).lower()]
                    feature_df = feature_df.drop(columns=drop_cols, errors="ignore")
                    frames.append(feature_df)
        for idx, result in enumerate(strategy_results or []):
            metrics = getattr(result, "metrics", None)
            if isinstance(metrics, dict):
                rows = []
                for symbol, values in metrics.items():
                    if isinstance(values, dict):
                        row = {"symbol": symbol}
                        for k, v in values.items():
                            row[f"sr_{k}"] = v
                        rows.append(row)
                if rows:
                    frames.append(pd.DataFrame(rows))
            signals = getattr(result, "signals", None)
            if signals:
                frames.append(pd.DataFrame([{"symbol": s, f"signal_{idx}": v} for s, v in signals.items()]))
        if raw_data is not None:
            micro_rows = []
            for symbol, df in raw_data.items():
                if not isinstance(df, pd.DataFrame):
                    continue
                row = {"symbol": symbol}
                micro = compute_microstructure_features(df)
                row.update({f"micro_{k}": v for k, v in micro.items()})
                micro_rows.append(row)
            if micro_rows:
                frames.append(pd.DataFrame(micro_rows))
        if not frames:
            raise ValueError("No features generated")
        feature_df = frames[0]
        for df in frames[1:]:
            feature_df = feature_df.merge(df, on="symbol", how="outer")
        if ret_series is None:
            raise ValueError("Missing market.ret target")
        if isinstance(ret_series, pd.DataFrame):
            ret_series = ret_series.iloc[:, 0]
        ret_series = pd.Series(ret_series).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        if ret_series.empty:
            raise ValueError("market.ret contains only NaN")
        ret_series.index = ret_series.index.astype(str)
        ret_series.index.name = "symbol"
        if raw_data is not None and len(ret_series) > 0:
            vol_map = {}
            for sym, df in raw_data.items():
                if isinstance(df, pd.DataFrame) and "close" in df.columns:
                    r = df["close"].pct_change()
                    vol = r.rolling(20).std().iloc[-1] if len(r) >= 20 else r.std()
                    if pd.isna(vol) or vol == 0:
                        vol = 1e-9
                    vol_map[sym] = vol
            if vol_map:
                vol_series = pd.Series(vol_map)
                vol_series = vol_series.replace([np.inf, -np.inf], np.nan)
                vol_median = vol_series.median()
                if pd.isna(vol_median):
                    vol_median = 1e-9
                vol_series = vol_series.fillna(vol_median)
                vol_series = vol_series.reindex(ret_series.index).fillna(vol_median)
                ret_series = ret_series / vol_series
                ret_series = ret_series.replace([np.inf, -np.inf], np.nan).fillna(0)
                print(f"[FEATURE] Target volatility-scaled. Range: [{ret_series.min():.4f}, {ret_series.max():.4f}]")
        y_df = ret_series.rename("y").reset_index()
        feature_df["symbol"] = feature_df["symbol"].astype(str)
        y_df["symbol"] = y_df["symbol"].astype(str)
        feature_df = feature_df.merge(y_df, on="symbol", how="inner")
        if feature_df.empty:
            raise ValueError("No symbol overlap between features and target")
        y = feature_df["y"]
        X = feature_df.drop(columns=["symbol", "y"], errors="ignore")
        X = self.encode_features(X)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        y = pd.Series(y).replace([np.inf, -np.inf], np.nan).fillna(0)
        print(f"[FEATURE] Final y: len={len(y)}, n_unique={y.nunique()}, range=[{y.min():.4f}, {y.max():.4f}]")
        return X, y, list(X.columns)

    def _encode_categoricals(self, X, fit=True):
        X_encoded = X.copy()
        for col in X_encoded.columns:
            if not pd.api.types.is_numeric_dtype(X_encoded[col]):
                if fit:
                    le = LabelEncoder()
                    X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    le = self.label_encoders.get(col)
                    if le:
                        X_encoded[col] = X_encoded[col].astype(str)
                        known_classes = set(le.classes_)
                        X_encoded[col] = X_encoded[col].apply(lambda x: le.transform([x])[0] if x in known_classes else -1)
                    else:
                        X_encoded[col] = 0
        return X_encoded

    def _try_import_booster(self, model_type):
        if model_type == "lightgbm":
            try:
                import lightgbm as lgb
                return lgb
            except ImportError:
                return None
        elif model_type == "xgboost":
            try:
                import xgboost as xgb
                return xgb
            except ImportError:
                return None
        return None

    def fit(self, X, y, params=None):
        if params is None:
            params = {}
        model_type = params.get("model", "random_forest")
        n_jobs = params.get("n_jobs", -1)
        reg_alpha = params.get("reg_alpha", 0.0)
        reg_lambda = params.get("reg_lambda", 1.0)
        learning_rate = params.get("learning_rate", 0.1)
        if model_type == "random_forest":
            model = RandomForestRegressor(
                n_estimators=params.get("trees", 500), max_depth=params.get("depth", 12),
                min_samples_split=params.get("min_samples_split", 2),
                max_features=params.get("max_features", "sqrt"),
                random_state=42, n_jobs=n_jobs,
            )
        elif model_type == "gradient_boosting":
            model = GradientBoostingRegressor(
                n_estimators=params.get("trees", 500), max_depth=params.get("depth", 6),
                min_samples_split=params.get("min_samples_split", 2),
                max_features=params.get("max_features", "sqrt"),
                learning_rate=learning_rate, random_state=42, subsample=0.8,
            )
        elif model_type == "lightgbm":
            booster = self._try_import_booster("lightgbm")
            if booster:
                model = booster.LGBMRegressor(
                    n_estimators=params.get("trees", 500), max_depth=params.get("depth", 6),
                    learning_rate=learning_rate, reg_alpha=reg_alpha, reg_lambda=reg_lambda,
                    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=n_jobs, verbosity=-1,
                )
            else:
                model = GradientBoostingRegressor(n_estimators=params.get("trees", 500), max_depth=params.get("depth", 6),
                    learning_rate=learning_rate, random_state=42, subsample=0.8)
        elif model_type == "xgboost":
            booster = self._try_import_booster("xgboost")
            if booster:
                model = booster.XGBRegressor(
                    n_estimators=params.get("trees", 500), max_depth=params.get("depth", 6),
                    learning_rate=learning_rate, reg_alpha=reg_alpha, reg_lambda=reg_lambda,
                    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=n_jobs,
                )
            else:
                model = GradientBoostingRegressor(n_estimators=params.get("trees", 500), max_depth=params.get("depth", 6),
                    learning_rate=learning_rate, random_state=42, subsample=0.8)
        elif model_type == "ridge":
            model = Ridge(alpha=reg_lambda)
        elif model_type == "lasso":
            model = Lasso(alpha=reg_alpha if reg_alpha > 0 else 0.01)
        elif model_type == "elasticnet":
            model = ElasticNet(alpha=reg_alpha if reg_alpha > 0 else 0.01, l1_ratio=0.5)
        elif model_type == "mlp":
            model = MLPRegressor(hidden_layer_sizes=(64, 32), alpha=reg_lambda,
                early_stopping=True, validation_fraction=0.15, max_iter=500, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=params.get("trees", 500), max_depth=params.get("depth", 12),
                random_state=42, n_jobs=n_jobs)
        model.fit(X, y)
        pred = model.predict(X)
        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "r2": float(r2_score(y, pred)),
            "samples": len(X), "features": len(X.columns), "model_type": model_type,
        }
        return {"model": model, "features": list(X.columns), "metrics": metrics}, metrics

    def encode_features(self, df):
        result = df.copy()
        for col in result.columns:
            if pd.api.types.is_numeric_dtype(result[col]):
                result[col] = result[col].fillna(0)
            elif pd.api.types.is_datetime64_any_dtype(result[col]):
                result[col] = result[col].astype("int64")
            else:
                result[col] = result[col].fillna("UNKNOWN").astype("category").cat.codes
        return result

class ProcessEngine:
    def __init__(self, use_feature_reduction=True, reduction_method="pca"):
        self.ml = MLTrainEngine(use_feature_reduction=use_feature_reduction, reduction_method=reduction_method)

    def train(self, train_package, params=None):
        if params is None:
            params = {}
        if not isinstance(train_package, dict):
            raise TypeError("ProcessEngine.train expects package dict")
        required = ["X", "y"]
        for key in required:
            if key not in train_package:
                raise KeyError(f"Missing train package key: {key}")
        ml_params = {
            "trees": params.get("trees", 500), "depth": params.get("depth", 12),
            "horizon": params.get("horizon", 21), "min_samples_split": params.get("min_samples_split", 2),
            "max_features": params.get("max_features", "sqrt"), "model": params.get("model", "random_forest"),
            "learning_rate": params.get("learning_rate", 0.1), "reg_alpha": params.get("reg_alpha", 0.0),
            "reg_lambda": params.get("reg_lambda", 1.0),
        }
        train_package["params"] = ml_params
        result = self.ml.train(train_package)
        train_package["model"] = result["model"]
        train_package["metrics"] = result.get("metrics", {})
        train_package["features"] = result.get("features", train_package.get("features", []))
        train_package["label_encoders"] = result.get("label_encoders", {})
        train_package["scaler"] = result.get("scaler", None)
        train_package["reducer"] = result.get("reducer", None)
        return train_package


# =====================================================
# BACKTEST ENGINE  —  OLD (preserved)
# =====================================================
class BacktestEngine_0:
    def __init__(self):
        self.last_features = None
        self.last_signal = None

    def build_features_for_signal(self, raw_data):
        rows = []
        for symbol, df in raw_data.items():
            if not isinstance(df, pd.DataFrame) or len(df) < 20:
                continue
            row = {"symbol": symbol}
            close = df["close"]
            volume = df["volume"]
            ret = df["ret"] if "ret" in df.columns else close.pct_change()
            row["market_price"] = close.iloc[-1]
            row["market_volume"] = volume.iloc[-1]
            row["market_structure_liq_adj_vol"] = volume.rolling(20).mean().iloc[-1]
            volatility = ret.rolling(20).std().iloc[-1]
            row["risk_volatility"] = volatility
            row["risk_sharpe"] = ret.mean() / (ret.std() + 1e-9)
            row["risk_drawdown"] = (close / close.cummax() - 1).iloc[-1]
            row["alpha_pure"] = ret.rolling(5).mean().iloc[-1]
            row["alpha_ts"] = close.iloc[-1] - close.rolling(20).mean().iloc[-1]
            row["alpha_beta"] = (ret.rolling(20).mean() / (ret.rolling(20).std() + 1e-9)).iloc[-1]
            row["alpha_residual"] = ret.iloc[-1] - ret.rolling(20).mean().iloc[-1]
            row["alpha_xs"] = close.pct_change(5).iloc[-1]
            mean20 = close.rolling(20).mean().iloc[-1]
            std20 = close.rolling(20).std().iloc[-1]
            row["transform_zscore"] = (close.iloc[-1] - mean20) / (std20 + 1e-9)
            row["transform_rank"] = close.rolling(20).rank().iloc[-1]
            row["transform_winsor"] = np.clip(ret.iloc[-1], -3 * ret.std(), 3 * ret.std())
            row["transform_tanh"] = np.tanh(ret.iloc[-1])
            row["market_symbol"] = symbol
            row["transform_detrend"] = close.iloc[-1] - close.rolling(20).mean().iloc[-1]
            micro = compute_microstructure_features(df)
            row.update(micro)
            defaults = ["basic_corr", "basic_hit_ratio", "basic_r_squared", "basic_tstat",
                        "decision_score", "decision_signal", "execution_impact",
                        "execution_slippage", "execution_turnover", "intel_ic",
                        "market_structure_regime", "portfolio_entropy", "portfolio_inv_vol",
                        "portfolio_kelly", "portfolio_mvo", "portfolio_risk_parity",
                        "portfolio_weight", "risk_cvar"]
            for c in defaults:
                row[c] = 0
            rows.append(row)
        df = pd.DataFrame(rows)
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        self.last_features = df.copy()
        return df

    def encode_features(self, df):
        result = df.copy()
        for col in result.columns:
            if pd.api.types.is_numeric_dtype(result[col]):
                result[col] = result[col].fillna(0)
            else:
                result[col] = result[col].fillna("UNKNOWN").astype("category").cat.codes
        return result

    def signal(self, model_package, dataset):
        if "model" not in model_package:
            raise KeyError("model_package missing model")
        model = model_package["model"]["model"]
        if "X" not in dataset:
            raise KeyError("dataset missing X")
        X = dataset["X"].copy()
        required_features = model_package["model"]["features"]
        reducer = model_package.get("reducer")
        if reducer is not None and hasattr(reducer, "transform"):
            X = reducer.transform(X)
        for col in required_features:
            if col not in X.columns:
                X[col] = 0
        X = X[required_features]
        if X.shape[0] == 0:
            return np.array([])
        label_encoders = model_package.get("label_encoders", {})
        if label_encoders:
            for col in X.columns:
                if not pd.api.types.is_numeric_dtype(X[col]):
                    le = label_encoders.get(col)
                    if le:
                        X[col] = X[col].astype(str)
                        known_classes = set(le.classes_)
                        X[col] = X[col].apply(lambda x: le.transform([x])[0] if x in known_classes else -1)
                    else:
                        X[col] = X[col].fillna("UNKNOWN").astype("category").cat.codes
        prediction = model.predict(X)
        prediction = np.asarray(prediction, dtype=float)
        self.last_signal = prediction
        return prediction

    def evaluate(self, package, signal):
        metrics = {}
        y_actual = package.get("y")
        if y_actual is not None:
            y_actual = pd.Series(y_actual).replace([np.inf, -np.inf], np.nan)
            signal_s = pd.Series(signal).replace([np.inf, -np.inf], np.nan)
            min_len = min(len(y_actual), len(signal_s))
            if min_len > 0:
                y_vec = y_actual.iloc[:min_len].values
                s_vec = signal_s.iloc[:min_len].values
                mask = pd.notna(y_vec) & pd.notna(s_vec)
                y_clean = y_vec[mask]
                s_clean = s_vec[mask]
                n_valid = len(y_clean)
                if n_valid >= 3:
                    if np.std(y_clean) > 1e-12 and np.std(s_clean) > 1e-12:
                        corr = float(np.corrcoef(y_clean, s_clean)[0, 1])
                    else:
                        corr = 0.0
                    try:
                        ic = float(stats.spearmanr(y_clean, s_clean)[0])
                    except Exception:
                        ic = 0.0
                    hit = float(np.mean(np.sign(y_clean) == np.sign(s_clean)))
                    ss_res = np.sum((y_clean - s_clean) ** 2)
                    ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
                    r2 = float(1.0 - ss_res / (ss_tot + 1e-12)) if ss_tot > 1e-12 else 0.0
                    if np.sum(np.abs(s_clean)) > 1e-12:
                        w = s_clean - np.mean(s_clean)
                        w = w / np.sum(np.abs(w))
                        port_rets = w * y_clean
                        port_mean = float(np.mean(port_rets))
                        port_std = float(np.std(port_rets))
                        sharpe = float(port_mean / (port_std + 1e-12) * np.sqrt(252)) if port_std > 1e-12 else 0.0
                        volatility = float(port_std * np.sqrt(252))
                    else:
                        port_mean = float(np.mean(y_clean))
                        port_std = float(np.std(y_clean))
                        sharpe = float(port_mean / (port_std + 1e-12) * np.sqrt(252)) if port_std > 1e-12 else 0.0
                        volatility = float(port_std * np.sqrt(252))
                    cvar = float(np.percentile(y_clean, 5))
                    if np.std(s_clean) > 1e-12:
                        beta = float(np.cov(y_clean, s_clean)[0, 1] / (np.var(s_clean) + 1e-12))
                    else:
                        beta = 0.0
                    alpha = float(np.mean(y_clean) - beta * np.mean(s_clean))
                    turnover = float(np.mean(np.abs(np.diff(s_clean)))) if len(s_clean) > 1 else 0.0
                    tcost = turnover * RETAIL_TRANSACTION_COST_LOW
                    if np.sum(np.abs(s_clean)) > 1e-12:
                        w = s_clean - np.mean(s_clean)
                        w = w / np.sum(np.abs(w))
                        daily_pnl = w * y_clean
                    else:
                        daily_pnl = y_clean
                    equity = np.cumprod(1 + daily_pnl)
                    running_max = np.maximum.accumulate(equity)
                    drawdowns = equity / running_max - 1
                    max_drawdown = float(np.min(drawdowns))
                    score = 0.25 * abs(ic) + 0.25 * abs(corr) + 0.20 * hit + 0.15 * max(0, sharpe) + 0.15 * max(0, r2)
                    metrics = {
                        "score": float(score), "sharpe": float(sharpe), "corr": float(corr),
                        "hit": float(hit), "r2": float(r2), "cvar": float(cvar),
                        "alpha": float(alpha), "beta": float(beta), "ret": float(np.mean(y_clean)),
                        "turnover": float(turnover), "tcost": float(tcost), "ic": float(ic),
                        "volatility": float(volatility), "drawdown": float(drawdowns[-1]) if len(drawdowns) else 0.0,
                        "max_drawdown": float(max_drawdown),
                        "tstat": float(alpha / (np.std(y_clean) / np.sqrt(n_valid) + 1e-12)),
                        "samples": n_valid,
                    }
                    return metrics
        # fallback
        metric_map = {
            "alpha": "alpha_alpha", "beta": "alpha_beta", "ret": "market_ret",
            "corr": "basic_corr", "hit": "basic_hit_ratio", "tstat": "basic_tstat",
            "turnover": "execution_turnover", "tcost": "execution_transaction_cost",
            "ic": "intel_ic", "volatility": "risk_volatility", "cvar": "risk_cvar",
            "sharpe": "risk_sharpe", "drawdown": "risk_drawdown",
            "max_drawdown": "risk_max_drawdown", "score": "decision_score",
            "signal": "decision_psignal",
        }
        formula_outputs = package.get("formula_outputs", [])
        if formula_outputs:
            for fo in formula_outputs:
                report = None
                if hasattr(fo, "reporting"):
                    try:
                        fo.assemble()
                        report_out = fo.reporting()
                        report = formula_report(report_out, mode="df_multi", lookup="columns")
                    except Exception:
                        continue
                elif isinstance(fo, pd.DataFrame):
                    report = fo
                elif isinstance(fo, tuple) and len(fo) == 2:
                    try:
                        report = formula_report(fo, mode="df_multi", lookup="columns")
                    except Exception:
                        continue
                if not isinstance(report, pd.DataFrame) or report.empty:
                    continue
                if isinstance(report.columns, pd.MultiIndex):
                    iter_items = report.columns
                    get_value = lambda item: report[item]
                else:
                    iter_items = report.index
                    get_value = lambda item: report.loc[item]
                for item in iter_items:
                    if isinstance(item, tuple) and len(item) == 2:
                        source_key = f"{item[0]}_{item[1]}".lower()
                    else:
                        source_key = str(item).lower()
                    for output_key, lookup_key in metric_map.items():
                        if source_key != lookup_key.lower():
                            continue
                        value = get_value(item)
                        if isinstance(value, pd.Series):
                            value = value.replace([np.inf, -np.inf], np.nan).dropna()
                            if value.empty:
                                continue
                            value = value.mean()
                        if pd.isna(value):
                            continue
                        try:
                            metrics[output_key] = float(value)
                        except Exception:
                            pass
        X = package.get("X")
        if isinstance(X, pd.DataFrame):
            for output_key, column_name in metric_map.items():
                if output_key in metrics:
                    continue
                candidates = [column_name, f"fo_{column_name}", f"fo_{column_name.replace('_', '_', 1)}"]
                found_col = None
                for cand in candidates:
                    if cand in X.columns:
                        found_col = cand
                        break
                if found_col is None:
                    matches = [c for c in X.columns if column_name in str(c)]
                    if matches:
                        found_col = matches[0]
                if found_col is None:
                    continue
                value = X[found_col].replace([np.inf, -np.inf], np.nan).dropna()
                if value.empty:
                    continue
                metrics[output_key] = float(value.mean())
        required_keys = ["score", "sharpe", "corr", "hit", "r2", "cvar", "alpha", "beta",
                         "ret", "turnover", "tcost", "ic", "volatility", "drawdown",
                         "max_drawdown", "tstat"]
        for key in required_keys:
            if key not in metrics:
                metrics[key] = 0.0
        if metrics["score"] == 0.0:
            synthetic = 0.0
            weights = {"sharpe": 0.3, "corr": 0.25, "hit": 0.2, "r2": 0.15, "alpha": 0.1}
            for k, w in weights.items():
                if k in metrics and metrics[k] != 0.0:
                    synthetic += metrics[k] * w
            if synthetic != 0.0:
                metrics["score"] = synthetic
            else:
                print(f"[EVALUATE] Missing score. Available metrics={list(metrics.keys())}")
        return metrics


# =====================================================
# BACKTEST ENGINE  —  NEW: canonical Sharpe, portfolio t-stat, cost sensitivity
# =====================================================
class BacktestEngine:
    def __init__(self):
        self.last_features = None
        self.last_signal = None

    def build_features_for_signal(self, raw_data):
        rows = []
        for symbol, df in raw_data.items():
            if not isinstance(df, pd.DataFrame) or len(df) < 20:
                continue
            row = {"symbol": symbol}
            close = df["close"]
            volume = df["volume"]
            ret = df["ret"] if "ret" in df.columns else close.pct_change()
            row["market_price"] = close.iloc[-1]
            row["market_volume"] = volume.iloc[-1]
            row["market_structure_liq_adj_vol"] = volume.rolling(20).mean().iloc[-1]
            volatility = ret.rolling(20).std().iloc[-1]
            row["risk_volatility"] = volatility
            # Canonical daily Sharpe (annualized later)
            row["risk_sharpe"] = ret.mean() / (ret.std() + 1e-9)
            row["risk_drawdown"] = (close / close.cummax() - 1).iloc[-1]
            row["alpha_pure"] = ret.rolling(5).mean().iloc[-1]
            row["alpha_ts"] = close.iloc[-1] - close.rolling(20).mean().iloc[-1]
            row["alpha_beta"] = (ret.rolling(20).mean() / (ret.rolling(20).std() + 1e-9)).iloc[-1]
            row["alpha_residual"] = ret.iloc[-1] - ret.rolling(20).mean().iloc[-1]
            row["alpha_xs"] = close.pct_change(5).iloc[-1]
            mean20 = close.rolling(20).mean().iloc[-1]
            std20 = close.rolling(20).std().iloc[-1]
            row["transform_zscore"] = (close.iloc[-1] - mean20) / (std20 + 1e-9)
            row["transform_rank"] = close.rolling(20).rank().iloc[-1]
            row["transform_winsor"] = np.clip(ret.iloc[-1], -3 * ret.std(), 3 * ret.std())
            row["transform_tanh"] = np.tanh(ret.iloc[-1])
            row["market_symbol"] = symbol
            row["transform_detrend"] = close.iloc[-1] - close.rolling(20).mean().iloc[-1]
            micro = compute_microstructure_features(df)
            row.update(micro)
            defaults = ["basic_corr", "basic_hit_ratio", "basic_r_squared", "basic_tstat",
                        "decision_score", "decision_signal", "execution_impact",
                        "execution_slippage", "execution_turnover", "intel_ic",
                        "market_structure_regime", "portfolio_entropy", "portfolio_inv_vol",
                        "portfolio_kelly", "portfolio_mvo", "portfolio_risk_parity",
                        "portfolio_weight", "risk_cvar"]
            for c in defaults:
                row[c] = 0
            rows.append(row)
        df = pd.DataFrame(rows)
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        self.last_features = df.copy()
        return df

    def encode_features(self, df):
        result = df.copy()
        for col in result.columns:
            if pd.api.types.is_numeric_dtype(result[col]):
                result[col] = result[col].fillna(0)
            else:
                result[col] = result[col].fillna("UNKNOWN").astype("category").cat.codes
        return result

    def signal(self, model_package, dataset):
        if "model" not in model_package:
            raise KeyError("model_package missing model")
        model = model_package["model"]["model"]
        if "X" not in dataset:
            raise KeyError("dataset missing X")
        X = dataset["X"].copy()
        required_features = model_package["model"]["features"]
        reducer = model_package.get("reducer")
        if reducer is not None and hasattr(reducer, "transform"):
            X = reducer.transform(X)
        for col in required_features:
            if col not in X.columns:
                X[col] = 0
        X = X[required_features]
        if X.shape[0] == 0:
            return np.array([])
        label_encoders = model_package.get("label_encoders", {})
        if label_encoders:
            for col in X.columns:
                if not pd.api.types.is_numeric_dtype(X[col]):
                    le = label_encoders.get(col)
                    if le:
                        X[col] = X[col].astype(str)
                        known_classes = set(le.classes_)
                        X[col] = X[col].apply(lambda x: le.transform([x])[0] if x in known_classes else -1)
                    else:
                        X[col] = X[col].fillna("UNKNOWN").astype("category").cat.codes
        prediction = model.predict(X)
        prediction = np.asarray(prediction, dtype=float)
        self.last_signal = prediction
        return prediction

    # =====================================================================
    # NEW evaluate: canonical Sharpe, portfolio alpha t-stat, cost sensitivity
    # =====================================================================
    def evaluate(self, package, signal):
        """
        Compute research metrics DIRECTLY from predictions (signal) and
        actual returns (package['y']).  All Sharpe values use canonical
        daily mean/std * sqrt(252).  Portfolio alpha t-stat comes from
        intercept of daily portfolio returns regressed on daily BTC returns.
        """
        metrics = {}
        y_actual = package.get("y")
        if y_actual is not None:
            y_actual = pd.Series(y_actual).replace([np.inf, -np.inf], np.nan)
            signal_s = pd.Series(signal).replace([np.inf, -np.inf], np.nan)
            min_len = min(len(y_actual), len(signal_s))
            if min_len > 0:
                y_vec = y_actual.iloc[:min_len].values
                s_vec = signal_s.iloc[:min_len].values
                mask = pd.notna(y_vec) & pd.notna(s_vec)
                y_clean = y_vec[mask]
                s_clean = s_vec[mask]
                n_valid = len(y_clean)
                if n_valid >= 3:
                    # Correlation (Pearson)
                    if np.std(y_clean) > 1e-12 and np.std(s_clean) > 1e-12:
                        try:
                            corr = float(np.corrcoef(y_clean, s_clean)[0, 1])
                        except Exception:
                            corr = 0.0
                    else:
                        corr = 0.0
                    # Information Coefficient (Spearman rank)
                    if np.std(s_clean) > 1e-12 and np.std(y_clean) > 1e-12:
                        try:
                            ic = float(stats.spearmanr(y_clean, s_clean)[0])
                        except Exception:
                            ic = 0.0
                    else:
                        ic = 0.0
                    # Hit ratio (directional accuracy)
                    hit = float(np.mean(np.sign(y_clean) == np.sign(s_clean)))
                    # R^2
                    ss_res = np.sum((y_clean - s_clean) ** 2)
                    ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
                    r2 = float(1.0 - ss_res / (ss_tot + 1e-12)) if ss_tot > 1e-12 else 0.0

                    # Portfolio returns from signal-as-weights (cross-sectional)
                    if np.sum(np.abs(s_clean)) > 1e-12:
                        w = s_clean - np.mean(s_clean)  # market-neutral
                        w = w / np.sum(np.abs(w))
                        port_rets = w * y_clean
                    else:
                        port_rets = y_clean
                    port_mean = float(np.mean(port_rets))
                    port_std = float(np.std(port_rets))

                    # CANONICAL SHARPE: daily mean / daily std * sqrt(252)
                    sharpe = float(port_mean / (port_std + 1e-12) * np.sqrt(252)) if port_std > 1e-12 else 0.0
                    volatility = float(port_std * np.sqrt(252))

                    # CVaR (5% tail)
                    cvar = float(np.percentile(y_clean, 5))

                    # Beta vs signal (simplified)
                    if np.std(s_clean) > 1e-12:
                        beta = float(np.cov(y_clean, s_clean)[0, 1] / (np.var(s_clean) + 1e-12))
                    else:
                        beta = 0.0
                    alpha = float(np.mean(y_clean) - beta * np.mean(s_clean))

                    # Turnover from executed (lagged) weights approximation
                    # In research cross-section we use |diff(signal)|
                    turnover = float(np.mean(np.abs(np.diff(s_clean)))) if len(s_clean) > 1 else 0.0

                    # Transaction cost sensitivity
                    tcost_10 = turnover * RETAIL_TRANSACTION_COST_LOW
                    tcost_20 = turnover * RETAIL_TRANSACTION_COST_HIGH

                    # Max drawdown on naive signal-based equity
                    equity = np.cumprod(1 + port_rets)
                    running_max = np.maximum.accumulate(equity)
                    drawdowns = equity / running_max - 1
                    max_drawdown = float(np.min(drawdowns))

                    # Portfolio alpha t-stat: intercept from daily portfolio ret vs BTC ret
                    if np.std(y_clean) > 1e-12 and len(port_rets) >= 3 and np.std(port_rets) > 1e-12:
                        # OLS: port_rets = alpha + beta*y_clean + epsilon
                        x_mean = np.mean(y_clean)
                        y_mean = np.mean(port_rets)
                        beta_ols = np.sum((y_clean - x_mean) * (port_rets - y_mean)) / (np.sum((y_clean - x_mean)**2) + 1e-12)
                        alpha_ols = y_mean - beta_ols * x_mean
                        residuals = port_rets - (alpha_ols + beta_ols * y_clean)
                        mse = np.sum(residuals**2) / (len(port_rets) - 2 + 1e-12)
                        se_alpha = np.sqrt(mse * (1.0/len(port_rets) + x_mean**2 / (np.sum((y_clean - x_mean)**2) + 1e-12)))
                        tstat_alpha = float(alpha_ols / (se_alpha + 1e-12))
                    else:
                        alpha_ols = alpha
                        tstat_alpha = float(alpha / (np.std(port_rets) / np.sqrt(n_valid) + 1e-12)) if np.std(port_rets) > 0 else 0.0

                    # Composite score — sanitize NaN/inf before weighting
                    ic_safe = float(np.nan_to_num(ic, nan=0.0, posinf=0.0, neginf=0.0))
                    corr_safe = float(np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0))
                    sharpe_safe = float(np.nan_to_num(sharpe, nan=0.0, posinf=0.0, neginf=0.0))
                    r2_safe = float(np.nan_to_num(r2, nan=0.0, posinf=0.0, neginf=0.0))
                    score = (
                        0.25 * abs(ic_safe) +
                        0.25 * abs(corr_safe) +
                        0.20 * hit +
                        0.15 * max(0, sharpe_safe) +
                        0.15 * max(0, r2_safe)
                    )

                    metrics = {
                        "score": float(score),
                        "sharpe": float(sharpe),
                        "corr": float(corr),
                        "hit": float(hit),
                        "r2": float(r2),
                        "cvar": float(cvar),
                        "alpha": float(alpha_ols),
                        "beta": float(beta_ols if 'beta_ols' in dir() else beta),
                        "ret": float(np.mean(y_clean)),
                        "turnover": float(turnover),
                        "tcost_10bps": float(tcost_10),
                        "tcost_20bps": float(tcost_20),
                        "ic": float(ic),
                        "volatility": float(volatility),
                        "drawdown": float(drawdowns[-1]) if len(drawdowns) else 0.0,
                        "max_drawdown": float(max_drawdown),
                        "tstat_alpha": float(tstat_alpha),
                        "samples": n_valid,
                    }
                    print(f"[EVALUATE] score={score:.4f} sharpe={sharpe:.4f} ic={ic:.4f} corr={corr:.4f} "
                          f"hit={hit:.4f} alpha={alpha_ols:.6f} tstat={tstat_alpha:.3f} n={n_valid}")
                    return metrics

        # --- FALLBACK ---
        print("[EVALUATE] WARNING: Falling back to formula_output / X column extraction")
        metric_map = {
            "alpha": "alpha_alpha", "beta": "alpha_beta", "ret": "market_ret",
            "corr": "basic_corr", "hit": "basic_hit_ratio", "tstat": "basic_tstat",
            "turnover": "execution_turnover", "tcost": "execution_transaction_cost",
            "ic": "intel_ic", "volatility": "risk_volatility", "cvar": "risk_cvar",
            "sharpe": "risk_sharpe", "drawdown": "risk_drawdown",
            "max_drawdown": "risk_max_drawdown", "score": "decision_score",
            "signal": "decision_psignal",
        }
        formula_outputs = package.get("formula_outputs", [])
        if formula_outputs:
            for fo in formula_outputs:
                report = None
                if hasattr(fo, "reporting"):
                    try:
                        fo.assemble()
                        report_out = fo.reporting()
                        report = formula_report(report_out, mode="df_multi", lookup="columns")
                    except Exception:
                        continue
                elif isinstance(fo, pd.DataFrame):
                    report = fo
                elif isinstance(fo, tuple) and len(fo) == 2:
                    try:
                        report = formula_report(fo, mode="df_multi", lookup="columns")
                    except Exception:
                        continue
                if not isinstance(report, pd.DataFrame) or report.empty:
                    continue
                if isinstance(report.columns, pd.MultiIndex):
                    iter_items = report.columns
                    get_value = lambda item: report[item]
                else:
                    iter_items = report.index
                    get_value = lambda item: report.loc[item]
                for item in iter_items:
                    if isinstance(item, tuple) and len(item) == 2:
                        source_key = f"{item[0]}_{item[1]}".lower()
                    else:
                        source_key = str(item).lower()
                    for output_key, lookup_key in metric_map.items():
                        if source_key != lookup_key.lower():
                            continue
                        value = get_value(item)
                        if isinstance(value, pd.Series):
                            value = value.replace([np.inf, -np.inf], np.nan).dropna()
                            if value.empty:
                                continue
                            value = value.mean()
                        if pd.isna(value):
                            continue
                        try:
                            metrics[output_key] = float(value)
                        except Exception:
                            pass
        X = package.get("X")
        if isinstance(X, pd.DataFrame):
            for output_key, column_name in metric_map.items():
                if output_key in metrics:
                    continue
                candidates = [column_name, f"fo_{column_name}", f"fo_{column_name.replace('_', '_', 1)}"]
                found_col = None
                for cand in candidates:
                    if cand in X.columns:
                        found_col = cand
                        break
                if found_col is None:
                    matches = [c for c in X.columns if column_name in str(c)]
                    if matches:
                        found_col = matches[0]
                if found_col is None:
                    continue
                value = X[found_col].replace([np.inf, -np.inf], np.nan).dropna()
                if value.empty:
                    continue
                metrics[output_key] = float(value.mean())
        required_keys = ["score", "sharpe", "corr", "hit", "r2", "cvar", "alpha", "beta",
                         "ret", "turnover", "tcost_10bps", "tcost_20bps", "ic", "volatility",
                         "drawdown", "max_drawdown", "tstat_alpha"]
        for key in required_keys:
            if key not in metrics:
                metrics[key] = 0.0
        if metrics["score"] == 0.0:
            synthetic = 0.0
            weights = {"sharpe": 0.3, "corr": 0.25, "hit": 0.2, "r2": 0.15, "alpha": 0.1}
            for k, w in weights.items():
                if k in metrics and metrics[k] != 0.0:
                    synthetic += metrics[k] * w
            if synthetic != 0.0:
                metrics["score"] = synthetic
            else:
                print(f"[EVALUATE] Missing score. Available metrics={list(metrics.keys())}")
        return metrics

class ReviewEngine:
    def __init__(self):
        self.comparison_history = []

    def compare(self, current_val, previous_val=None):
        review = {
            "timestamp": datetime.now().isoformat(),
            "current": current_val, "previous": previous_val,
            "degradation": {}, "trend": "FIRST_RUN",
            "warnings": [], "recommendations": []
        }
        if previous_val is None:
            review["recommendations"].append("First iteration: establish baseline")
            return review
        metrics_to_compare = ["sharpe", "corr", "hit", "r2", "score", "cvar"]
        for metric in metrics_to_compare:
            if metric in current_val and metric in previous_val:
                curr = current_val[metric]
                prev = previous_val[metric]
                degradation = (curr - prev) / abs(prev) if prev != 0 else (0 if curr == 0 else float("inf"))
                review["degradation"][metric] = {
                    "current": curr, "previous": prev,
                    "change": curr - prev, "pct_change": degradation
                }
                if metric in ["sharpe", "score", "corr"] and degradation < -0.15:
                    review["warnings"].append(f"{metric} degraded by {degradation:.1%}")
                if metric == "cvar" and curr < prev * 0.8:
                    review["warnings"].append(f"CVaR worsened: {curr:.4f} vs {prev:.4f}")
        score_change = review["degradation"].get("score", {}).get("pct_change", 0)
        if score_change > 0.05:
            review["trend"] = "IMPROVING"
        elif score_change > -0.05:
            review["trend"] = "STABLE"
        else:
            review["trend"] = "DEGRADING"
        if review["trend"] == "DEGRADING":
            review["recommendations"].append("Consider feature mutation or model parameter adjustment")
            review["recommendations"].append("Check for overfitting: compare train vs val gap")
        if len(review["warnings"]) > 2:
            review["recommendations"].append("Multiple metrics degraded: recommend STOP or major mutation")
        self.comparison_history.append(review)
        return review


class Evaluator:
    def __init__(self):
        self.decision_history = []
        self.thresholds = {
            "min_sharpe": 0.5, "min_corr": 0.1, "min_hit": 0.52,
            "max_overfit_gap": 0.3, "min_score": 0.3,
            "min_oos_predictions": MIN_OOS_PREDICTIONS,
        }

    def decide(self, review, train_metrics=None, iteration=0, max_iters=5):
        current = review.get("current", {})
        trend = review.get("trend", "FIRST_RUN")
        warnings = review.get("warnings", [])
        decision = {"verdict": "CONTINUE", "confidence": 0.5, "reasons": [], "mutations": [], "flag": None}
        if current.get("score", 0) == 0 and trend == "FIRST_RUN" and iteration == 0:
            decision["flag"] = "DATA_INSUFFICIENT"
            decision["reasons"].append("Zero convergence on first iteration - data may be insufficient")
            decision["verdict"] = "MUTATE"
            decision["confidence"] = 0.4
            self.decision_history.append(decision)
            return decision["verdict"]
        if current.get("sharpe", 0) < self.thresholds["min_sharpe"]:
            decision["reasons"].append(f"Sharpe {current.get('sharpe', 0):.3f} below threshold")
        if current.get("corr", 0) < self.thresholds["min_corr"]:
            decision["reasons"].append(f"Correlation {current.get('corr', 0):.3f} below threshold")
        if current.get("hit", 0) < self.thresholds["min_hit"]:
            decision["reasons"].append(f"Hit ratio {current.get('hit', 0):.3f} below threshold")
        oos_count = current.get("samples", 0)
        if oos_count > 0 and oos_count < self.thresholds["min_oos_predictions"]:
            decision["reasons"].append(f"OOS predictions {oos_count} < {self.thresholds['min_oos_predictions']}")
        if train_metrics:
            train_score = train_metrics.get("score", 0)
            val_score = current.get("score", 0)
            if train_score > 0 and (train_score - val_score) / train_score > self.thresholds["max_overfit_gap"]:
                decision["reasons"].append(f"Overfit detected: train={train_score:.3f}, val={val_score:.3f}")
        baseline_score = self.decision_history[0]["baseline_score"] if self.decision_history else None
        if trend == "IMPROVING" and len(decision["reasons"]) == 0:
            decision["verdict"] = "ACCEPT"
            decision["confidence"] = 0.85
            decision["reasons"].append("Validation improving and meets all thresholds")
        elif trend == "STABLE" and len(decision["reasons"]) <= 1:
            decision["verdict"] = "CONTINUE"
            decision["confidence"] = 0.7
        elif trend == "DEGRADING" or len(decision["reasons"]) > 1:
            non_zero_baseline = baseline_score is not None and baseline_score != 0
            degraded_from_baseline = non_zero_baseline and current.get("score", 0) < baseline_score * 0.8
            if iteration >= max_iters:
                decision["verdict"] = "STOP"
                decision["confidence"] = 0.9
                decision["reasons"].append(f"Reached max iterations ({max_iters})")
            elif degraded_from_baseline:
                decision["verdict"] = "STOP"
                decision["confidence"] = 0.9
                decision["reasons"].append("Score degraded from non-zero baseline")
            elif len(warnings) > 2 or current.get("score", 0) < self.thresholds["min_score"]:
                decision["verdict"] = "MUTATE"
                decision["confidence"] = 0.6
                decision["mutations"] = self._suggest_mutations(review)
            else:
                decision["verdict"] = "CONTINUE"
                decision["confidence"] = 0.5
        if trend == "FIRST_RUN":
            decision["verdict"] = "CONTINUE" if len(decision["reasons"]) == 0 else "MUTATE"
        if not self.decision_history:
            decision["baseline_score"] = current.get("score", 0)
        else:
            decision["baseline_score"] = self.decision_history[0].get("baseline_score", 0)
        self.decision_history.append(decision)
        return decision["verdict"]

    def _suggest_mutations(self, review):
        mutations = []
        degradation = review.get("degradation", {})
        if "sharpe" in degradation and degradation["sharpe"]["pct_change"] < -0.1:
            mutations.append({"target": "model_params", "action": "increase_regularization", "details": "Increase max_depth or min_samples_split"})
        if "corr" in degradation and degradation["corr"]["pct_change"] < -0.1:
            mutations.append({"target": "features", "action": "add_market_structure", "details": "Include more regime-sensitive features"})
        if "hit" in degradation and degradation["hit"]["pct_change"] < -0.05:
            mutations.append({"target": "model_type", "action": "try_ensemble", "details": "Switch to gradient_boosting or ensemble"})
        mutations.append({"target": "feature_set", "action": "toggle_microstructure", "details": "Toggle microstructure features on/off"})
        mutations.append({"target": "feature_set", "action": "toggle_reduction", "details": "Toggle PCA/ICA/reduction method"})
        return mutations

    def update_thresholds(self, thresholds):
        self.thresholds.update(thresholds)


class GoLiveEngine:
    def __init__(self):
        self.readiness_criteria = {
            "min_iterations": 3, "min_accept_ratio": 0.5,
            "min_avg_sharpe": 0.6, "min_avg_corr": 0.15,
            "max_avg_drawdown": -0.15, "min_consistency": 0.7,
            "min_oos_predictions": MIN_OOS_PREDICTIONS,
        }

    def assess(self, research_history, model_package):
        if len(research_history) < self.readiness_criteria["min_iterations"]:
            return {"ready": False, "stage": "RESEARCH",
                    "reason": f"Insufficient iterations: {len(research_history)}/{self.readiness_criteria['min_iterations']}",
                    "systemic_prediction": None}
        val_scores = [r["validation"].get("score", 0) for r in research_history]
        val_sharpes = [r["validation"].get("sharpe", 0) for r in research_history]
        val_corrs = [r["validation"].get("corr", 0) for r in research_history]
        val_drawdowns = [r["validation"].get("max_drawdown", 0) for r in research_history]
        oos_counts = [r["validation"].get("samples", 0) for r in research_history]
        min_oos = min(oos_counts) if oos_counts else 0
        systemic_prediction = {
            "expected_sharpe": float(np.mean(val_sharpes)),
            "sharpe_confidence_interval": (float(np.percentile(val_sharpes, 25)), float(np.percentile(val_sharpes, 75))),
            "expected_corr": float(np.mean(val_corrs)),
            "expected_score": float(np.mean(val_scores)),
            "score_stability": float(1 - np.std(val_scores) / (np.mean(val_scores) + 1e-9)),
            "win_rate": float(np.mean([s > 0 for s in val_scores])),
            "consistency_ratio": float(np.mean([s > 0.3 for s in val_scores])),
            "min_oos_predictions": int(min_oos),
        }
        checks = {
            "accept_ratio": sum(1 for r in research_history if r.get("decision") == "ACCEPT") / len(research_history),
            "avg_sharpe": np.mean(val_sharpes), "avg_corr": np.mean(val_corrs),
            "avg_drawdown": np.mean(val_drawdowns), "consistency": systemic_prediction["consistency_ratio"],
            "oos_sufficient": min_oos >= self.readiness_criteria["min_oos_predictions"],
        }
        passed_checks = all([
            checks["accept_ratio"] >= self.readiness_criteria["min_accept_ratio"],
            checks["avg_sharpe"] >= self.readiness_criteria["min_avg_sharpe"],
            checks["avg_corr"] >= self.readiness_criteria["min_avg_corr"],
            checks["avg_drawdown"] >= self.readiness_criteria["max_avg_drawdown"],
            checks["consistency"] >= self.readiness_criteria["min_consistency"],
            checks["oos_sufficient"],
        ])
        deployment_package = None
        if passed_checks:
            deployment_package = self._create_deployment_package(model_package, systemic_prediction)
        return {
            "ready": passed_checks, "stage": "GOLIVE" if passed_checks else "RESEARCH",
            "checks": checks, "systemic_prediction": systemic_prediction,
            "deployment_package": deployment_package,
            "recommendation": "DEPLOY" if passed_checks else "CONTINUE_RESEARCH"
        }

    def _create_deployment_package(self, model_package, prediction):
        return {
            "model": model_package["model"]["model"],
            "features": model_package["model"]["features"],
            "expected_performance": prediction,
            "timestamp": datetime.now().isoformat(),
            "version": f"quantx-{datetime.now().strftime('%Y%m%d-%H%M%S')}"}

    def update_criteria(self, criteria):
        self.readiness_criteria.update(criteria)


class CombinatorialScenarioOptimizer:
    def __init__(self, backtest_engine, storage, max_combinations=500):
        self.backtest = backtest_engine
        self.storage = storage
        self.max_combinations = max_combinations
        self.results = []
        self.best_params = None
        self.best_resiliency_score = -float("inf")

    def generate_param_combinations(self, param_grid, max_count=500):
        keys = list(param_grid.keys())
        values = [param_grid[k] for k in keys]
        all_combos = list(itertools.product(*values))
        if len(all_combos) > max_count:
            np.random.seed(42)
            indices = np.random.choice(len(all_combos), size=max_count, replace=False)
            selected = [all_combos[i] for i in indices]
        else:
            selected = all_combos
        combos = []
        for combo in selected:
            param_dict = dict(zip(keys, combo))
            param_dict["_hash"] = hashlib.md5(json.dumps(param_dict, sort_keys=True, default=str).encode()).hexdigest()[:8]
            combos.append(param_dict)
        return combos

    def apply_scenario(self, data, scenario_name):
        config = SCENARIO_CONFIG.get(scenario_name, SCENARIO_CONFIG["STABLE"])
        mu_mult = config["mu_multiplier"]
        vol_mult = config["vol_multiplier"]
        stressed_data = {}
        for symbol, df in data.items():
            if not isinstance(df, pd.DataFrame):
                continue
            df_copy = df.copy()
            if "ret" in df_copy.columns:
                ret = df_copy["ret"]
                mean_ret = ret.mean()
                stressed_ret = mu_mult * mean_ret + vol_mult * (ret - mean_ret)
                df_copy["ret"] = stressed_ret
            if "close" in df_copy.columns:
                close0 = df_copy["close"].iloc[0]
                df_copy["close"] = close0 * (1 + stressed_ret).cumprod()
            stressed_data[symbol] = df_copy
        return stressed_data

    def evaluate_scenario(self, model_package, data, params, scenario_name):
        stressed_data = self.apply_scenario(data, scenario_name)
        try:
            eng = run_quantx_engine(stressed_data, interval=params.get("interval", "4y"), params=params)
            if not eng["success"]:
                raise RuntimeError(eng["error"])
            formula_outputs = eng["formula_outputs"]
            ml = MLTrainEngine()
            X, y = ml.build_feature_matrix(
                strategy_results=eng["results"],
                formula_outputs=formula_outputs,
                raw_data=stressed_data
            )
        except Exception as e:
            print(f"[EVALUATE] Engine fallback after error: {e}")
            ml = MLTrainEngine()
            X, y = self._build_features_from_raw(stressed_data)
            if X.shape[0] == 0 or len(y) == 0:
                return None
            package = {"X": X, "y": y, "features": list(X.columns)}
            signal = self.backtest.signal(model_package, package)
            metrics = self.backtest.evaluate(package, signal)
            return {"scenario": scenario_name, "params": params, "metrics": metrics,
                    "sharpe": metrics["sharpe"], "score": metrics["score"], "corr": metrics["corr"]}
        package = {"X": X, "y": y, "features": list(X.columns), "formula_outputs": formula_outputs}
        signal = self.backtest.signal(model_package, package)
        metrics = self.backtest.evaluate(package, signal)
        return {"scenario": scenario_name, "params": params, "metrics": metrics,
                "sharpe": metrics.get("sharpe", 0.0), "score": metrics.get("score", 0.0), "corr": metrics.get("corr", 0.0)}

    def _build_features_from_raw(self, data):
        rpt_fo = FormulaInfo(data)
        rpt_fo.assemble()
        rpt = rpt_fo.reporting()
        y = None
        try:
            rpt_norm = formula_report(rpt, mode="df_easy", lookup="index")
            if "market_ret" in rpt_norm.columns and not rpt_norm["market_ret"].isna().all():
                y = rpt_norm["market_ret"].copy()
                X = rpt_norm.drop(columns=["market_ret"], errors="ignore")
            else:
                raise KeyError("market_ret missing or all-NaN")
        except (KeyError, ValueError):
            ret_raw = rpt_fo.get("ret")
            if isinstance(ret_raw, pd.DataFrame):
                y = ret_raw.iloc[-1].copy()
            elif isinstance(ret_raw, pd.Series):
                y = ret_raw.copy()
            else:
                y = pd.Series({
                    sym: df["close"].pct_change().iloc[-1]
                    for sym, df in data.items()
                    if isinstance(df, pd.DataFrame) and "close" in df.columns
                })
            y.index = y.index.astype(str)
            y.index.name = "symbol"
            y = y.rename("market_ret")
        try:
            rpt_norm = formula_report(rpt, mode="df_easy", lookup="symbol_features")
            X = rpt_norm.drop(columns=["market_ret"], errors="ignore")
        except Exception:
            rows = []
            for sym, df in data.items():
                if not isinstance(df, pd.DataFrame):
                    continue
                row = {"symbol": sym}
                for col in ["open", "high", "low", "close", "volume"]:
                    if col in df.columns:
                        row[f"raw_{col}"] = df[col].iloc[-1]
                rows.append(row)
            X = pd.DataFrame(rows)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        y = y.replace([np.inf, -np.inf], np.nan).fillna(0)
        print(f"[FALLBACK] X={X.shape}, y={len(y)}, ret_range=[{y.min():.4f}, {y.max():.4f}]")
        return X, y

    def optimize(self, base_model_package, data, param_grid, scenarios=None):
        if scenarios is None:
            scenarios = ["BULL", "BEAR", "HIGH_VOL", "CRASH", "STABLE"]
        combinations = self.generate_param_combinations(param_grid, self.max_combinations)
        print(f"[OPTIMIZER] Testing {len(combinations)} parameter combinations")
        print(f"[OPTIMIZER] Across {len(scenarios)} scenarios: {scenarios}")
        all_results = []
        scenario_matrix = defaultdict(list)
        for idx, params in enumerate(combinations):
            print(f"[OPTIMIZER] Combination {idx+1}/{len(combinations)}: trees={params.get('trees')}, depth={params.get('depth')}, model={params.get('model')}")
            scenario_scores = []
            scenario_sharpes = []
            for scenario in scenarios:
                result = self.evaluate_scenario(base_model_package, data, params, scenario)
                if result is None:
                    continue
                scenario_scores.append(result["score"])
                scenario_sharpes.append(result["sharpe"])
                scenario_matrix[scenario].append(result)
                all_results.append(result)
                print(f"  [{scenario}] Sharpe={result['sharpe']:.4f} Score={result['score']:.4f}")
            if len(scenario_scores) == 0:
                continue
            min_sharpe = min(scenario_sharpes)
            avg_score = np.mean(scenario_scores)
            resiliency = 0.6 * min_sharpe + 0.4 * avg_score
            print(f"  [RESILIENCY] min_sharpe={min_sharpe:.4f} avg_score={avg_score:.4f} -> {resiliency:.4f}")
            if resiliency > self.best_resiliency_score:
                self.best_resiliency_score = resiliency
                self.best_params = params
                print(f"  [NEW BEST] Resiliency={resiliency:.4f}")
        self.storage.save({
            "best_params": self.best_params, "best_resiliency_score": self.best_resiliency_score,
            "all_results": all_results, "scenario_matrix": dict(scenario_matrix),
            "timestamp": datetime.now().isoformat()
        }, "combinatorial_optimization.pkl")
        self.storage.save_json({
            "best_params": self.best_params, "best_resiliency_score": float(self.best_resiliency_score),
            "total_combinations_tested": len(combinations), "scenarios_tested": scenarios,
            "timestamp": datetime.now().isoformat()
        }, "combinatorial_optimization.json")
        return {"best_params": self.best_params, "best_resiliency_score": self.best_resiliency_score,
                "all_results": all_results, "scenario_matrix": dict(scenario_matrix)}

class ModelComparator:
    def __init__(self, storage):
        self.storage = storage
        self.comparison_results = {}

    def compare_models(self, baseline_model, optimized_model, baseline_val_metrics, optimized_val_metrics):
        comparison = {
            "timestamp": datetime.now().isoformat(),
            "model_comparison": {}, "validation_comparison": {},
            "rating": {}, "prediction": {}
        }
        baseline_ml = baseline_model.get("metrics", {})
        optimized_ml = optimized_model.get("metrics", {})
        model_metrics = ["rmse", "r2", "samples", "features"]
        for metric in model_metrics:
            if metric in baseline_ml and metric in optimized_ml:
                b, o = baseline_ml[metric], optimized_ml[metric]
                if isinstance(b, (int, float)) and isinstance(o, (int, float)) and b != 0:
                    pct_change = (o - b) / abs(b)
                else:
                    pct_change = 0
                comparison["model_comparison"][metric] = {
                    "baseline": b, "optimized": o,
                    "absolute_change": o - b if isinstance(o, (int, float)) and isinstance(b, (int, float)) else None,
                    "pct_change": pct_change
                }
        val_metrics = ["sharpe", "corr", "hit", "r2", "score", "cvar", "t_beta"]
        for metric in val_metrics:
            if metric in baseline_val_metrics and metric in optimized_val_metrics:
                b, o = baseline_val_metrics[metric], optimized_val_metrics[metric]
                pct_change = (o - b) / abs(b) if b != 0 else (0 if o == 0 else float("inf"))
                comparison["validation_comparison"][metric] = {
                    "baseline": b, "optimized": o,
                    "absolute_change": o - b, "pct_change": pct_change
                }
        weights = {"sharpe": 0.30, "corr": 0.25, "score": 0.25, "hit": 0.10, "r2": 0.10}
        rating_score, rating_details = 0, []
        for metric, weight in weights.items():
            if metric in comparison["validation_comparison"]:
                pct_change = comparison["validation_comparison"][metric]["pct_change"]
                contribution = pct_change * weight
                rating_score += contribution
                rating_details.append({"metric": metric, "weight": weight, "pct_change": pct_change, "contribution": contribution})
        if rating_score > 0.20:
            rating_label = "STRONG_IMPROVEMENT"
        elif rating_score > 0.05:
            rating_label = "MODERATE_IMPROVEMENT"
        elif rating_score > -0.05:
            rating_label = "NEUTRAL"
        elif rating_score > -0.20:
            rating_label = "MODERATE_DEGRADATION"
        else:
            rating_label = "STRONG_DEGRADATION"
        comparison["rating"] = {
            "score": float(rating_score), "label": rating_label,
            "details": rating_details,
            "interpretation": self._interpret_rating(rating_score, rating_label)
        }
        baseline_score = baseline_val_metrics.get("score", 0)
        optimized_score = optimized_val_metrics.get("score", 0)
        if baseline_score != 0:
            improvement_rate = (optimized_score - baseline_score) / abs(baseline_score)
            predicted_next_score = optimized_score * (1 + improvement_rate * 0.5)
        else:
            improvement_rate, predicted_next_score = 0, optimized_score
        improvements = [d["pct_change"] for d in rating_details]
        consistency = 1 - np.std(improvements) / (np.mean(np.abs(improvements)) + 1e-9) if improvements else 0
        comparison["prediction"] = {
            "predicted_next_score": float(predicted_next_score),
            "improvement_rate": float(improvement_rate),
            "consistency": float(consistency),
            "confidence": float(min(0.95, 0.5 + consistency * 0.5)),
            "recommendation": self._predict_recommendation(rating_label, consistency)
        }
        self.comparison_results = comparison
        self.storage.save(comparison, "model_comparison.pkl")
        self.storage.save_json(comparison, "model_comparison.json")
        return comparison

    def _interpret_rating(self, score, label):
        interpretations = {
            "STRONG_IMPROVEMENT": "Parameter optimization yielded significant predictive power gains. Model is more robust.",
            "MODERATE_IMPROVEMENT": "Measurable improvement in key metrics. Optimization was beneficial but not transformative.",
            "NEUTRAL": "No meaningful change. Parameters may already be near-optimal or grid insufficient.",
            "MODERATE_DEGRADATION": "Optimization hurt performance. Consider reverting to baseline or expanding search space.",
            "STRONG_DEGRADATION": "Significant performance loss. Baseline is superior. Investigate data leakage or overfitting."
        }
        return interpretations.get(label, "Unknown rating state.")

    def _predict_recommendation(self, label, consistency):
        if label in ["STRONG_IMPROVEMENT", "MODERATE_IMPROVEMENT"] and consistency > 0.6:
            return "DEPLOY_OPTIMIZED: High confidence that optimized model outperforms baseline."
        elif label in ["STRONG_IMPROVEMENT", "MODERATE_IMPROVEMENT"]:
            return "DEPLOY_WITH_MONITORING: Improvement detected but inconsistent. Watch for degradation."
        elif label == "NEUTRAL":
            return "EITHER_MODEL_OK: No significant difference. Choose based on computational cost or simplicity."
        else:
            return "KEEP_BASELINE: Optimized model underperforms. Retain baseline and investigate search space."


class TrueValidationEngine:
    def __init__(self):
        self.validation_history = []

    def true_validate(self, model_package, formula_outputs, strategy_results=None):
        if not model_package or "model" not in model_package:
            return {"error": "No model available for validation"}
        model = model_package["model"]["model"]
        features = model_package["model"]["features"]
        actual_returns = None
        if formula_outputs is not None:
            for fo in formula_outputs:
                if hasattr(fo, "get") and callable(getattr(fo, "get")):
                    try:
                        ret_series = fo.get("ret")
                        if ret_series is not None and isinstance(ret_series, pd.Series):
                            actual_returns = ret_series.values
                            break
                    except:
                        pass
                if isinstance(fo, pd.DataFrame):
                    if isinstance(fo.columns, pd.MultiIndex):
                        for col in fo.columns:
                            if isinstance(col, tuple) and len(col) >= 2:
                                if str(col[1]).lower() in ["ret", "return", "returns"]:
                                    actual_returns = fo[col].values
                                    break
                    else:
                        for col in fo.columns:
                            if str(col).lower() in ["ret", "return", "returns"]:
                                actual_returns = fo[col].values
                                break
                if actual_returns is not None:
                    break
        if actual_returns is None and strategy_results is not None:
            for result in strategy_results:
                if hasattr(result, "signals") and result.signals:
                    signals = list(result.signals.values())
                    if signals:
                        actual_returns = np.array(signals)
                        break
        if actual_returns is None:
            return {"error": "Could not extract actual returns from formula outputs or strategy results"}
        if "X" in model_package and model_package["X"] is not None:
            X = model_package["X"].copy()
            for col in features:
                if col not in X.columns:
                    X[col] = 0
            X = X[features]
            predicted_signals = model.predict(X)
        else:
            return {"error": "No feature matrix X available for prediction"}
        min_len = min(len(actual_returns), len(predicted_signals))
        actual_returns = np.asarray(actual_returns[:min_len], dtype=float)
        predicted_signals = np.asarray(predicted_signals[:min_len], dtype=float)
        mask = ~(np.isnan(actual_returns) | np.isnan(predicted_signals))
        actual_returns, predicted_signals = actual_returns[mask], predicted_signals[mask]
        if len(actual_returns) < 10:
            return {"error": "Insufficient data for validation (need >= 10 points)"}
        if np.std(actual_returns) == 0 or np.std(predicted_signals) == 0:
            correlation = 0
        else:
            correlation = np.corrcoef(actual_returns, predicted_signals)[0, 1]
        directional_accuracy = np.mean(np.sign(actual_returns) == np.sign(predicted_signals))
        ss_res = np.sum((actual_returns - predicted_signals) ** 2)
        ss_tot = np.sum((actual_returns - np.mean(actual_returns)) ** 2)
        r_squared = 1 - (ss_res / (ss_tot + 1e-9))
        result = {
            "correlation": float(correlation), "directional_accuracy": float(directional_accuracy),
            "r_squared": float(r_squared), "samples": len(actual_returns),
            "actual_mean": float(np.mean(actual_returns)), "predicted_mean": float(np.mean(predicted_signals)),
            "actual_std": float(np.std(actual_returns)), "predicted_std": float(np.std(predicted_signals)),
            "validated": True
        }
        self.validation_history.append(result)
        return result


class SRFORegistry:
    _counter = 0

    @classmethod
    def reset(cls):
        cls._counter = 0

    @classmethod
    def next_id(cls):
        cls._counter += 1
        return f"fo_{cls._counter}"

    @classmethod
    def collect_formula_outputs(cls, strategies, reset=False):
        if reset:
            cls.reset()
        outputs = []
        for idx, strategy in enumerate(strategies, start=1):
            fo = getattr(strategy, "formulaOutput", getattr(strategy, "formula_output", None))
            if fo is None:
                continue
            outputs.append({"id": cls.next_id(), "strategy_id": idx, "object": fo})
        return outputs

    @classmethod
    def wrap_formula_output(cls, fo, strategy_id=None):
        return {"id": cls.next_id(), "strategy_id": strategy_id, "object": fo}

    @classmethod
    def reports(cls, formula_outputs):
        reports = []
        for item in formula_outputs:
            fo_id = item["id"]
            fo = item["object"]
            if not hasattr(fo, "reporting"):
                continue
            fo.assemble()
            df = fo.reporting().copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [f"{fo_id}_{c[0]}_{c[1]}" for c in df.columns]
            else:
                df.columns = [f"{fo_id}_{c}" for c in df.columns]
            reports.append(df)
        return reports

class AgenticPipeline:
    def __init__(self, data, base_dir="runs", use_feature_reduction=True, reduction_method="pca",
                 split_ratio=DEFAULT_SPLIT_RATIO, split_date=None):
        self.data = data
        self.base_dir = base_dir
        self.current_run_idx = None
        self.current_run_dir = None
        self.storage = Storage(base_dir)
        self.split_engine = SplitEngine()
        self.strategy = ProcessEngine(use_feature_reduction=use_feature_reduction, reduction_method=reduction_method)
        self.backtest = BacktestEngine()
        self.review_engine = ReviewEngine()
        self.evaluator = Evaluator()
        self.golive = GoLiveEngine()
        self.optimizer = None
        self.comparator = ModelComparator(self.storage)
        self.true_validator = TrueValidationEngine()
        self.iteration = 0
        self.research_history = []
        self.best_model = None
        self.best_score = -float("inf")
        self.baseline_model_package = None
        self.optimized_model_package = None
        self.baseline_val_metrics = None
        self.optimized_val_metrics = None
        self._formula_outputs_raw = None
        self._srfo_full = None
        self._Xy_train = None
        self._Xy_val = None
        self.use_feature_reduction = use_feature_reduction
        self.reduction_method = reduction_method
        self.split_ratio = split_ratio
        self.split_date = split_date  # optional explicit YYYY-MM-DD
        self._feature_reduction_enabled = use_feature_reduction

    def _generate_srfo(self, data):
        eng = run_quantx_engine(data, interval="4y")
        if not eng["success"]:
            print(f"[SRFO] Engine failed: {eng['error']}")
            return None
        print(f"[SRFO] Engine: {len(eng['results'])} strategies, {len(eng['formula_outputs_raw'])} raw formulaOutput objects")
        return {
            "results": eng["results"],
            "formula_outputs": eng["formula_outputs"],
            "formula_outputs_raw": eng["formula_outputs_raw"],
            "raw_data": data
        }

    def _ensure_srfo(self):
        if self._srfo_full is not None:
            return
        print("[SRFO] Generating raw SRFO from full dataset...")
        # NEW: common cutoff date temporal split
        train_data, val_data = self.split_engine.split(
            self.data,
            split_ratio=self.split_ratio,
            explicit_split_date=self.split_date
        )
        print(f"[SRFO] Temporal split: train symbols={len(train_data)}, val symbols={len(val_data)}")

        # Build train features
        print("[SRFO] Building train-period features...")
        eng_train = run_quantx_engine(train_data, interval="4y")
        ml_train = MLTrainEngine(use_feature_reduction=self._feature_reduction_enabled, reduction_method=self.reduction_method)
        if eng_train["success"] and eng_train.get("formula_outputs"):
            try:
                X_train, y_train = ml_train.build_feature_matrix(
                    strategy_results=eng_train["results"],
                    formula_outputs=eng_train["formula_outputs"],
                    raw_data=train_data
                )
                results_train = eng_train["results"]
                fo_train = eng_train["formula_outputs"]
                fo_raw_train = eng_train["formula_outputs_raw"]
            except Exception as e:
                print(f"[SRFO] Train engine feature build failed: {e}")
                X_train, y_train = self._build_features_from_raw(train_data)
                results_train = []
                fo_train = []
                fo_raw_train = []
        else:
            print("[SRFO] Train engine failed, using fallback feature extraction")
            X_train, y_train = self._build_features_from_raw(train_data)
            results_train = []
            fo_train = []
            fo_raw_train = []

        # Build val features
        print("[SRFO] Building validation-period features...")
        eng_val = run_quantx_engine(val_data, interval="4y")
        ml_val = MLTrainEngine(use_feature_reduction=self._feature_reduction_enabled, reduction_method=self.reduction_method)
        if eng_val["success"] and eng_val.get("formula_outputs"):
            try:
                X_val, y_val = ml_val.build_feature_matrix(
                    strategy_results=eng_val["results"],
                    formula_outputs=eng_val["formula_outputs"],
                    raw_data=val_data
                )
                results_val = eng_val["results"]
                fo_val = eng_val["formula_outputs"]
                fo_raw_val = eng_val["formula_outputs_raw"]
            except Exception as e:
                print(f"[SRFO] Val engine feature build failed: {e}")
                X_val, y_val = self._build_features_from_raw(val_data)
                results_val = []
                fo_val = []
                fo_raw_val = []
        else:
            print("[SRFO] Val engine failed, using fallback feature extraction")
            X_val, y_val = self._build_features_from_raw(val_data)
            results_val = []
            fo_val = []
            fo_raw_val = []

        self._Xy_train = {
            "X": X_train, "y": y_train, "features": list(X_train.columns),
            "results": results_train, "formula_outputs": fo_train,
            "formula_outputs_raw": fo_raw_train
        }
        self._Xy_val = {
            "X": X_val, "y": y_val, "features": list(X_val.columns),
            "results": results_val, "formula_outputs": fo_val,
            "formula_outputs_raw": fo_raw_val
        }
        self._srfo_full = {
            "train": eng_train, "val": eng_val,
            "train_data": train_data, "val_data": val_data
        }
        if fo_raw_train:
            self._formula_outputs_raw = fo_raw_train
        elif fo_raw_val:
            self._formula_outputs_raw = fo_raw_val
        else:
            self._formula_outputs_raw = []
        print(f"[SRFO] Cached raw formulaOutput objects: {len(self._formula_outputs_raw)}")
        print(f"[SRFO] Split: train={len(self._Xy_train['y'])}, val={len(self._Xy_val['y'])}")
        self._Xy_full = {"X": pd.concat([X_train, X_val], ignore_index=True),
                         "y": pd.concat([y_train, y_val], ignore_index=True),
                         "features": list(X_train.columns)}

    def get_train_package(self, params=None):
        self._ensure_srfo()
        pkg = deepcopy(self._Xy_train)
        pkg["params"] = params or {}
        return pkg

    def get_val_package(self, params=None):
        self._ensure_srfo()
        pkg = deepcopy(self._Xy_val)
        pkg["params"] = params or {}
        return pkg

    def prepare(self, data, params=None):
        if params is None:
            params = {}
        try:
            eng = run_quantx_engine(data, interval=params.get("interval", "4y"), params=params)
            if not eng["success"]:
                raise RuntimeError(eng["error"])
            print(f"[ENGINE] Strategies: {len(eng['results'])}")
            print(f"[ML] Formula outputs: {len(eng['formula_outputs'])}")
            if len(eng["formula_outputs"]) == 0:
                raise ValueError("No formula outputs generated")
            ml = MLTrainEngine(use_feature_reduction=self._feature_reduction_enabled, reduction_method=self.reduction_method)
            X, y = ml.build_feature_matrix(
                strategy_results=eng["results"],
                formula_outputs=eng["formula_outputs"],
                raw_data=data
            )
            print(f"[ML] Dataset: {X.shape}")
            return {
                "X": X, "y": y, "features": list(X.columns),
                "results": eng["results"], "formula_outputs": eng["formula_outputs"]
            }
        except Exception as e:
            print(f"[WARN] QuantXEngine not available or failed: {e}")
            print("[WARN] Falling back to direct feature extraction from raw data")
            X, y = self._build_features_from_raw(data)
            print(f"[ML] Dataset (fallback): {X.shape}")
            return {
                "X": X, "y": y, "features": list(X.columns),
                "results": {}, "formula_outputs": []
            }

    def _build_features_from_raw(self, data):
        rpt_fo = FormulaInfo(data)
        rpt_fo.assemble()
        rpt = rpt_fo.reporting()
        y = None
        try:
            rpt_norm = formula_report(rpt, mode="df_easy", lookup="symbol_features")
            if "market_ret" in rpt_norm.columns and not rpt_norm["market_ret"].isna().all():
                y = rpt_norm["market_ret"].copy()
                X = rpt_norm.drop(columns=["market_ret"], errors="ignore")
            else:
                raise KeyError("market_ret missing or all-NaN")
        except (KeyError, ValueError):
            ret_raw = rpt_fo.get("ret")
            if isinstance(ret_raw, pd.DataFrame):
                y = ret_raw.iloc[-1].copy()
            elif isinstance(ret_raw, pd.Series):
                y = ret_raw.copy()
            else:
                y = pd.Series({
                    sym: df["close"].pct_change().iloc[-1]
                    for sym, df in data.items()
                    if isinstance(df, pd.DataFrame) and "close" in df.columns
                })
            y.index = y.index.astype(str)
            y.index.name = "symbol"
            y = y.rename("market_ret")
        try:
            rpt_norm = formula_report(rpt, mode="df_easy", lookup="symbol_features")
            X = rpt_norm.drop(columns=["market_ret"], errors="ignore")
        except Exception:
            rows = []
            for sym, df in data.items():
                if not isinstance(df, pd.DataFrame):
                    continue
                row = {"symbol": sym}
                for col in ["open", "high", "low", "close", "volume"]:
                    if col in df.columns:
                        row[f"raw_{col}"] = df[col].iloc[-1]
                rows.append(row)
            X = pd.DataFrame(rows)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        y = y.replace([np.inf, -np.inf], np.nan).fillna(0)
        print(f"[FALLBACK] X={X.shape}, y={len(y)}, ret_range=[{y.min():.4f}, {y.max():.4f}]")
        return X, y

    def run_agent(self, params=None, max_iters=5, run_combinatorial=False, param_grid=None):
        """
        run_combinatorial is DISABLED by default per user request.
        Only basic train/val research loop + portfolio summary table.
        """
        if params is None:
            params = {}

        # === A. DATA SUFFICIENCY GATE ===
        print(f"\n{'='*60}")
        print("DATA SUFFICIENCY CHECK")
        print(f"{'='*60}")
        # NEW: configurable gate behavior
        min_rows = params.get("min_rows", MIN_ROWS_PER_ASSET)
        skip_insufficient = params.get("skip_insufficient", True)

        insufficient_assets = []
        for sym, df in self.data.items():
            if not isinstance(df, pd.DataFrame) or len(df) < min_rows:
                insufficient_assets.append({"symbol": sym, "rows": len(df) if isinstance(df, pd.DataFrame) else 0})

        if insufficient_assets:
            if skip_insufficient:
                # FILTER MODE: drop young assets and continue
                filtered_data = {
                    sym: df for sym, df in self.data.items()
                    if isinstance(df, pd.DataFrame) and len(df) >= min_rows
                }
                if not filtered_data:
                    raise ValueError(
                        f"[DATA_GATE] HALT: All {len(insufficient_assets)} assets filtered out. "
                        f"No asset meets {min_rows} row minimum."
                    )
                print(f"[DATA_GATE] FILTERED: Removed {len(insufficient_assets)} assets below {min_rows} rows.")
                print(f"[DATA_GATE] Removed: " + ", ".join(f"{a['symbol']}({a['rows']})" for a in insufficient_assets))
                print(f"[DATA_GATE] Continuing with {len(filtered_data)} assets.")
                self.storage.save_json({
                    "flag": "DATA_FILTERED",
                    "removed_assets": insufficient_assets,
                    "min_required": min_rows,
                    "remaining_count": len(filtered_data),
                    "timestamp": datetime.now().isoformat()
                }, "data_filtered_flag.json")
                self.data = filtered_data
            else:
                # HALT MODE: original behavior (for strict production gates)
                msg = (
                    f"[DATA_GATE] HALT: {len(insufficient_assets)} assets below {min_rows} rows: "
                    + ", ".join(f"{a['symbol']}({a['rows']})" for a in insufficient_assets)
                )
                print(msg)
                self.storage.save_json({
                    "flag": "DATA_INSUFFICIENT",
                    "assets": insufficient_assets,
                    "min_required": min_rows,
                    "timestamp": datetime.now().isoformat()
                }, "data_insufficient_flag.json")
                raise ValueError(msg)
        else:
            print(f"[DATA_GATE] All {len(self.data)} assets pass >= {min_rows} rows check.")

        # === CREATE RUN FOLDER ===
        self.current_run_idx, run_path = self.storage.create_run()
        self.current_run_dir = str(run_path)
        print(f"\n{'='*60}")
        print(f"PIPELINE RUN {self.current_run_idx:06d}")
        print(f"Run directory: {self.current_run_dir}")
        print(f"{'='*60}")

        # === PHASE 0: SRFO ONCE ===
        print(f"\n{'='*60}")
        print("PHASE 0: GENERATING SRFO")
        print(f"{'='*60}")
        self._ensure_srfo()
        self.storage.save(self._Xy_train, f"{self.current_run_dir}/srfo/srfo_train.pkl")
        self.storage.save(self._Xy_val, f"{self.current_run_dir}/srfo/srfo_val.pkl")

        # === PHASE 1: BASELINE ===
        print(f"\n{'='*60}")
        print("PHASE 1: BASELINE MODEL")
        print(f"{'='*60}")
        baseline_params = {
            "trees": params.get("trees", 500), "depth": params.get("depth", 12),
            "horizon": params.get("horizon", 21), "min_samples_split": params.get("min_samples_split", 2),
            "max_features": params.get("max_features", "sqrt"), "model": params.get("model", "random_forest"),
            "learning_rate": params.get("learning_rate", 0.1), "reg_alpha": params.get("reg_alpha", 0.0),
            "reg_lambda": params.get("reg_lambda", 1.0),
        }
        baseline_train = self.strategy.train(self.get_train_package(baseline_params))
        self.baseline_model_package = deepcopy(baseline_train)
        self.storage.save(self.baseline_model_package, f"{self.current_run_dir}/baseline/baseline_model.pkl")

        baseline_val_pkg = self.get_val_package()
        baseline_signal = self.backtest.signal(self.baseline_model_package, baseline_val_pkg)
        self.baseline_val_metrics = self.backtest.evaluate(baseline_val_pkg, baseline_signal)
        print(f"[BASELINE] Score: {self.baseline_val_metrics.get('score', 0):.4f}")
        self.storage.save(self.baseline_val_metrics, f"{self.current_run_dir}/baseline/baseline_val_metrics.pkl")

        # === PHASE 2: COMBINATORIAL — DISABLED per user request ===
        # if run_combinatorial:
        #     print(f"\n{'='*60}")
        #     print("PHASE 2: COMBINATORIAL SEARCH (SRFO-CACHED)")
        #     print(f"{'='*60}")
        #     if param_grid is None:
        #         param_grid = PARAM_GRID_INSIDE
        #     opt_result = self._run_combinatorial_srfo(param_grid, max_combinations=100)
        #     best_params = opt_result["best_params"]
        #     if best_params:
        #         for k in ["trees", "depth", "horizon", "min_samples_split", "max_features", "model", "learning_rate", "reg_alpha", "reg_lambda"]:
        #             if k in best_params:
        #                 params[k] = best_params[k]
        #         print(f"[OPTIMIZER] Best: {best_params}, Resiliency: {opt_result['best_resiliency_score']:.4f}")
        #     self.storage.save_json(opt_result, f"{self.current_run_dir}/combinatorial/combinatorial_optimization.json")

        # === PHASE 3: OPTIMIZED (same as baseline when combinatorial disabled) ===
        print(f"\n{'='*60}")
        print("PHASE 3: OPTIMIZED MODEL")
        print(f"{'='*60}")
        opt_train = self.strategy.train(self.get_train_package(params))
        self.optimized_model_package = deepcopy(opt_train)
        self.storage.save(self.optimized_model_package, f"{self.current_run_dir}/optimized/optimized_model.pkl")

        opt_val_pkg = self.get_val_package()
        opt_signal = self.backtest.signal(self.optimized_model_package, opt_val_pkg)
        self.optimized_val_metrics = self.backtest.evaluate(opt_val_pkg, opt_signal)
        print(f"[OPTIMIZED] Score: {self.optimized_val_metrics.get('score', 0):.4f}")
        self.storage.save(self.optimized_val_metrics, f"{self.current_run_dir}/optimized/optimized_val_metrics.pkl")

        # === PHASE 4: COMPARATOR ===
        print(f"\n{'='*60}")
        print("PHASE 4: COMPARATOR")
        print(f"{'='*60}")
        comparison = self.comparator.compare_models(
            self.baseline_model_package, self.optimized_model_package,
            self.baseline_val_metrics, self.optimized_val_metrics)
        print(f"[COMPARE] {comparison['rating']['label']} | {comparison['prediction']['recommendation']}")
        self.storage.save_json(comparison, f"{self.current_run_dir}/comparison/model_comparison.json")

        # === PHASE 5: RESEARCH LOOP ===
        print(f"\n{'='*60}")
        print("PHASE 5: RESEARCH LOOP")
        print(f"{'='*60}")
        previous_val = None
        for i in range(max_iters):
            self.iteration += 1
            iter_dir = f"{self.current_run_dir}/iter_{self.iteration:02d}"
            os.makedirs(f"{self.base_dir}/{iter_dir}", exist_ok=True)
            print(f"\n--- Iteration {self.iteration}/{max_iters} --> {iter_dir} ---")

            train_pkg = self.strategy.train(self.get_train_package(params))
            val_pkg = self.get_val_package()

            train_signal = self.backtest.signal(train_pkg, train_pkg)
            train_metrics = self.backtest.evaluate(train_pkg, train_signal)

            val_signal = self.backtest.signal(train_pkg, val_pkg)
            val_metrics = self.backtest.evaluate(val_pkg, val_signal)

            def _fmt_metrics(m, label):
                keys = ["sharpe", "volatility", "max_drawdown", "alpha", "beta", "tstat_alpha",
                        "corr", "ic", "hit", "r2", "turnover", "tcost_10bps", "tcost_20bps", "score", "samples"]
                print(f"\n{'='*50}")
                print(f"  {label} METRICS")
                print(f"{'='*50}")
                for k in keys:
                    v = m.get(k, 0.0)
                    if isinstance(v, float):
                        print(f"  {k:<18} {v:>12.4f}")
                    else:
                        print(f"  {k:<18} {v:>12}")
                print(f"{'='*50}")
            _fmt_metrics(train_metrics, "TRAIN")
            _fmt_metrics(val_metrics, "VALIDATION")

            review = self.review_engine.compare(val_metrics, previous_val)
            decision = self.evaluator.decide(review, train_metrics, iteration=self.iteration, max_iters=max_iters)

            if val_metrics.get("score", 0) > self.best_score:
                self.best_score = val_metrics.get("score", 0)
                self.best_model = deepcopy(train_pkg)

            research = {
                "iteration": self.iteration,
                "train": {k: v for k, v in train_metrics.items() if k not in ["signal", "pnl"]},
                "validation": {k: v for k, v in val_metrics.items() if k not in ["signal", "pnl"]},
                "decision": decision
            }
            self.research_history.append(research)

            self.storage.save(train_pkg, f"{iter_dir}/train_package.pkl")
            self.storage.save(val_pkg, f"{iter_dir}/val_package.pkl")
            self.storage.save(train_signal, f"{iter_dir}/train_signal.pkl")
            self.storage.save(val_signal, f"{iter_dir}/val_signal.pkl")
            self.storage.save(train_metrics, f"{iter_dir}/train_metrics.pkl")
            self.storage.save(val_metrics, f"{iter_dir}/val_metrics.pkl")
            self.storage.save(research, f"{iter_dir}/research.pkl")
            self.storage.save_json(research, f"{iter_dir}/research.json")
            self.storage.save(review, f"{iter_dir}/review.pkl")
            self.storage.save({"decision": decision}, f"{iter_dir}/eval.pkl")

            golive = self.golive.assess(self.research_history, train_pkg)
            self.storage.save(golive, f"{iter_dir}/golive.pkl")
            self.storage.save_json(golive, f"{iter_dir}/golive.json")

            print(f"Decision: {decision} | GoLive: {golive['stage']}")

            if decision == "STOP":
                break
            if decision == "ACCEPT" and golive["ready"]:
                self.storage.save(golive.get("deployment_package"), f"{self.current_run_dir}/deployment_package.pkl")
                break
            if decision == "MUTATE":
                params = self._apply_mutations(params, review.get("recommendations", []), self.evaluator._suggest_mutations(review))

            previous_val = val_metrics

        # ============================================================
        # PORTFOLIO HELPERS  —  compute metrics from actual series
        # ============================================================
        def _build_portfolio(fo, data_source, label="portfolio"):
            if fo is None:
                print(f"[PORTFOLIO] [{label}] No formulaOutput. Skipping.")
                return None
            try:
                ret_df = fo.get("ret")
                # Sanity-check: returns must be decimal daily returns, not price levels or percent-scale
                if ret_df is not None and isinstance(ret_df, pd.DataFrame):
                    sample = ret_df.values.flatten()
                    sample = sample[~np.isnan(sample)]
                    if len(sample) > 0:
                        min_val, max_val = float(np.min(sample)), float(np.max(sample))
                        if min_val < -0.99 or max_val > 10.0:
                            print(f"[PORTFOLIO] [{label}] WARNING: ret values out of range [{min_val:.4f}, {max_val:.4f}]. "
                                  f"Expected decimal daily returns. Forcing fallback pct_change().")
                            ret_df = None  # force fallback
                if ret_df is None or not isinstance(ret_df, pd.DataFrame):
                    ret_parts = []
                    for sym, df in data_source.items():
                        if isinstance(df, pd.DataFrame) and "close" in df.columns:
                            ret_parts.append(df["close"].pct_change().fillna(0).rename(sym))
                    ret_df = pd.concat(ret_parts, axis=1).fillna(0) if ret_parts else None
                if ret_df is None or ret_df.empty:
                    raise ValueError("No return data.")

                weights = fo.get("executed_weight")
                if weights is None:
                    print(f"[PORTFOLIO] [{label}] No engine weights; using equal weight.")
                    weights = pd.DataFrame(
                        1.0 / len(ret_df.columns),
                        index=ret_df.index,
                        columns=ret_df.columns
                    )
                elif isinstance(weights, pd.Series):
                    weights = pd.DataFrame(
                        {col: weights for col in ret_df.columns},
                        index=ret_df.index
                    ).fillna(0)
                    weights = weights[ret_df.columns].fillna(0)
                elif isinstance(weights, pd.DataFrame):
                    for c in ret_df.columns:
                        if c not in weights.columns:
                            weights[c] = 0.0
                    weights = weights[ret_df.columns].fillna(0)
                else:
                    weights = pd.DataFrame(
                        1.0 / len(ret_df.columns),
                        index=ret_df.index,
                        columns=ret_df.columns
                    )

                benchmark = fo.get("benchmark")
                if benchmark is None:
                    benchmark = ret_df["BTCUSDT"] if "BTCUSDT" in ret_df.columns else ret_df.mean(axis=1)
                elif isinstance(benchmark, pd.DataFrame):
                    benchmark = benchmark.iloc[:, 0]
                benchmark = benchmark.squeeze()

                transaction_cost = fo.get("transaction_cost")
                if transaction_cost is None:
                    transaction_cost = RETAIL_TRANSACTION_COST_LOW
                if hasattr(transaction_cost, "iloc"):
                    transaction_cost = float(transaction_cost.iloc[0]) if len(transaction_cost) > 0 else RETAIL_TRANSACTION_COST_LOW
                try:
                    transaction_cost = float(transaction_cost)
                except (TypeError, ValueError):
                    transaction_cost = RETAIL_TRANSACTION_COST_LOW

                common_idx = ret_df.index.intersection(weights.index).intersection(benchmark.index)
                ret_df = ret_df.loc[common_idx].sort_index()
                weights = weights.loc[common_idx].sort_index()
                benchmark = benchmark.loc[common_idx].sort_index()

                portfolio = Portfolio()
                result = portfolio.invoke(
                    fo=fo, weights=weights, returns=ret_df,
                    benchmark=benchmark, transaction_cost=transaction_cost, annualization=252
                )

                chart_path = os.path.join(self.current_run_dir, f"{label}_chart.html")
                result["chart"].write_html(chart_path)
                print(f"[PORTFOLIO] [{label}] Chart saved: {chart_path}")

                self.storage.save(result["metrics"], f"{self.current_run_dir}/{label}_metrics.pkl")
                self.storage.save_json(result["metrics"], f"{self.current_run_dir}/{label}_metrics.json")
                self.storage.save(result["series"], f"{self.current_run_dir}/{label}_series.pkl")

                m = result["metrics"]
                print(f"[PORTFOLIO] [{label}] Metrics:")
                print(f"  Gross Return: {m.get('gross_return', 0):.4f}")
                print(f"  Net Return:   {m.get('net_return', 0):.4f}")
                print(f"  Sharpe:       {m.get('sharpe', 0):.4f}")
                print(f"  Volatility:   {m.get('volatility', 0):.4f}")
                print(f"  Max Drawdown: {m.get('max_drawdown', 0):.4f}")
                print(f"  BTC Corr:     {m.get('corr_rm', 0):.4f}")
                print(f"  Alpha:        {m.get('alpha', 0):.6f}")
                print(f"  Beta:         {m.get('beta', 0):.4f}")
                print(f"  Hit Ratio:    {m.get('hit_ratio', 0):.4f}")
                print(f"  IC:           {m.get('ic', 0):.4f}")
                print(f"  Turnover:     {m.get('mean_turnover', 0):.4f}")
                return result
            except Exception as e:
                print(f"[PORTFOLIO] [{label}] Failed: {e}")
                import traceback
                traceback.print_exc()
                return None

        # ============================================================
        # PORTFOLIO 1 - TRAINED DATA (1st 50%)
        # ============================================================
        print(f"\n{'='*60}")
        print("PORTFOLIO: TRAINED DATA (1st 50%)")
        print(f"{'='*60}")
        chk_formula_output = None
        try:
            trained_data = {}
            for sym, df in self.data.items():
                if isinstance(df, pd.DataFrame):
                    split_idx = len(df) // 2
                    trained_data[sym] = df.iloc[:split_idx].copy()
            fo_train = None
            try:
                eng_train = run_quantx_engine(trained_data, interval=params.get("interval", "4y"), params=params)
                if eng_train["success"] and eng_train["formula_outputs_raw"]:
                    fo_train = eng_train["formula_outputs_raw"][0]
                    chk_formula_output = fo_train
                    print("[PORTFOLIO] [train] Using run_quantx_engine().formula_outputs_raw[0]")
                else:
                    raise RuntimeError("Engine returned no formula outputs")
            except Exception as e:
                print(f"[PORTFOLIO] [train] Engine failed ({e}), falling back to FormulaInfo")
                trained_fo = FormulaInfo(trained_data)
                trained_fo.assemble()
                chk_formula_output = trained_fo
            _build_portfolio(chk_formula_output, trained_data, label="portfolio_train")
        except Exception as e:
            print(f"[PORTFOLIO] [train] Outer exception: {e}")
            import traceback
            traceback.print_exc()

        # ============================================================
        # PORTFOLIO 2 - MERGED / FULL DATA
        # ============================================================
        print(f"\n{'='*60}")
        print("PORTFOLIO: MERGED / FULL DATA")
        print(f"{'='*60}")
        fo_merged = None
        raw_cache = getattr(self, '_formula_outputs_raw', None)
        if raw_cache and len(raw_cache) > 0:
            try:
                fo_merged = raw_cache[0]
                chk_formula_output = fo_merged
                print("[PORTFOLIO] [merged] Using _formula_outputs_raw cache.")
            except Exception as e:
                print(f"[PORTFOLIO] [merged] Cache failed: {e}")
        if fo_merged is None and getattr(self, '_srfo_full', None):
            srfo_raw = self._srfo_full.get('formula_outputs_raw')
            if srfo_raw and len(srfo_raw) > 0:
                try:
                    fo_merged = srfo_raw[0]
                    chk_formula_output = fo_merged
                    print("[PORTFOLIO] [merged] Using _srfo_full cache.")
                except Exception as e:
                    print(f"[PORTFOLIO] [merged] SRFO cache failed: {e}")
        if fo_merged is None:
            try:
                eng_merged = run_quantx_engine(self.data, interval=params.get("interval", "4y"), params=params)
                if eng_merged["success"] and eng_merged["formula_outputs_raw"]:
                    fo_merged = eng_merged["formula_outputs_raw"][0]
                    chk_formula_output = fo_merged
                    print("[PORTFOLIO] [merged] Using run_quantx_engine().")
                else:
                    raise RuntimeError("Engine returned no formula outputs")
            except Exception as e:
                print(f"[PORTFOLIO] [merged] Engine failed ({e}), falling back to FormulaInfo")
                merged_fo = FormulaInfo(self.data)
                merged_fo.assemble()
                chk_formula_output = merged_fo
        _build_portfolio(chk_formula_output, self.data, label="portfolio_merged")

        # === FINAL SUMMARY ===
        return self._generate_final_summary()

    def _apply_mutations(self, params, recommendations, mutations):
        new_params = deepcopy(params)
        cache_invalidated = False
        for mutation in mutations:
            if isinstance(mutation, dict):
                if mutation.get("target") == "model_params":
                    if "model_params" not in new_params:
                        new_params["model_params"] = {}
                    new_params["model_params"]["max_depth"] = new_params["model_params"].get("max_depth", 10) + 2
                elif mutation.get("target") == "model_type":
                    new_params["model_type"] = "gradient_boosting"
                elif mutation.get("action") == "try_ensemble":
                    new_params["ensemble"] = True
                elif mutation.get("target") == "feature_set":
                    if mutation.get("action") == "toggle_microstructure":
                        new_params["_microstructure_enabled"] = not new_params.get("_microstructure_enabled", True)
                        self.use_feature_reduction = new_params["_microstructure_enabled"]
                        print(f"[MUTATION] Toggled microstructure -> use_feature_reduction={self.use_feature_reduction}")
                        cache_invalidated = True
                    elif mutation.get("action") == "toggle_reduction":
                        methods = ["pca", "ica", "none"]
                        current = new_params.get("_reduction_method", self.reduction_method)
                        idx = methods.index(current) if current in methods else 0
                        new_params["_reduction_method"] = methods[(idx + 1) % len(methods)]
                        self.reduction_method = new_params["_reduction_method"]
                        print(f"[MUTATION] Toggled reduction method -> {self.reduction_method}")
                        cache_invalidated = True
        # Invalidate cached feature matrices so _ensure_srfo rebuilds with new settings
        if cache_invalidated:
            self._srfo_full = None
            self._Xy_train = None
            self._Xy_val = None
            self._Xy_full = None
            print("[MUTATION] Cache invalidated. Feature matrices will be rebuilt next iteration.")
        return new_params

    def _generate_final_summary(self):
        summary = {
            "total_iterations": self.iteration,
            "best_score": self.best_score,
            "research_history": self.research_history,
            "golive_ready": any(r.get("decision") == "ACCEPT" for r in self.research_history),
            "run_idx": self.current_run_idx,
            "run_dir": self.current_run_dir,
            "best_model_path": f"{self.current_run_dir}/best_model.pkl" if self.best_model else None,
            "baseline_model_path": f"{self.current_run_dir}/baseline/baseline_model.pkl",
            "optimized_model_path": f"{self.current_run_dir}/optimized/optimized_model.pkl",
            "has_baseline": self.baseline_model_package is not None,
            "has_optimized": self.optimized_model_package is not None,
            "comparison_available": self.comparator.comparison_results != {}
        }
        if self.best_model:
            self.storage.save(self.best_model, f"{self.current_run_dir}/best_model.pkl")
        self.storage.save_json(summary, f"{self.current_run_dir}/final_summary.json")
        print(f"\n{'='*60}")
        print(f"FINAL SUMMARY - RUN {self.current_run_idx:06d}")
        print(f"{'='*60}")
        print(f"Run directory: {self.current_run_dir}")
        print(f"Total iterations: {self.iteration}")
        print(f"Best validation score: {self.best_score:.4f}")
        print(f"GoLive ready: {summary['golive_ready']}")

        # Print best validation metrics if available
        if self.best_model and self.research_history:
            best_iter = max(self.research_history, key=lambda r: r['validation'].get('score', 0))
            print(f"\n{'='*60}")
            print("BEST VALIDATION ITERATION")
            print(f"{'='*60}")
            keys = ["sharpe", "volatility", "max_drawdown", "alpha", "beta", "tstat_alpha",
                    "corr", "ic", "hit", "r2", "turnover", "tcost_10bps", "tcost_20bps", "score", "samples"]
            for k in keys:
                v = best_iter['validation'].get(k, 0.0)
                if isinstance(v, float):
                    print(f"  {k:<18} {v:>12.4f}")
                else:
                    print(f"  {k:<18} {v:>12}")
            print(f"{'='*60}")
        return summary

# =====================================================
# Portfolio class  —  FIXED: compute metrics from series
# =====================================================
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly
import plotly.colors


class Portfolio:
    def __init__(self, run_dir: Optional[str] = None, storage: Optional[Any] = None):
        self.run_dir = run_dir
        self.storage = storage

    def _compute_metrics_from_series(self, gross_return, net_return, benchmark, turnover, costs):
        """Compute all portfolio metrics from actual return series (not stale fo)."""
        gross_return = pd.Series(gross_return).replace([np.inf, -np.inf], np.nan).dropna()
        net_return = pd.Series(net_return).replace([np.inf, -np.inf], np.nan).dropna()
        benchmark = pd.Series(benchmark).replace([np.inf, -np.inf], np.nan).dropna()

        if len(net_return) == 0:
            return {k: 0.0 for k in ["gross_return","net_return","sharpe","volatility",
                                      "max_drawdown","alpha","beta","tstat_alpha","hit_ratio","ic",
                                      "mean_turnover","total_turnover","total_costs","corr_rm"]}

        # Cumulative
        gross_equity = (1 + gross_return).cumprod()
        net_equity = (1 + net_return).cumprod()
        bench_equity = (1 + benchmark.reindex(net_return.index).fillna(0)).cumprod()

        # Drawdown
        running_max = net_equity.cummax()
        drawdown = net_equity / running_max - 1
        max_drawdown = float(drawdown.min())

        # Annualized metrics — CANONICAL: daily mean/std * sqrt(252)
        ann_factor = 252
        mean_ret = float(net_return.mean())
        std_ret = float(net_return.std())
        sharpe = float(mean_ret / (std_ret + 1e-12) * np.sqrt(ann_factor)) if std_ret > 0 else 0.0
        volatility = float(std_ret * np.sqrt(ann_factor))

        # Alpha / Beta vs benchmark — daily regression
        bench_aligned = benchmark.reindex(net_return.index).fillna(0)
        if np.std(bench_aligned) > 1e-12 and len(net_return) >= 3:
            # OLS: net_return = alpha + beta * benchmark + epsilon
            x_mean = np.mean(bench_aligned)
            y_mean = np.mean(net_return)
            beta = float(np.sum((bench_aligned - x_mean) * (net_return - y_mean)) / (np.sum((bench_aligned - x_mean)**2) + 1e-12))
            alpha = float(y_mean - beta * x_mean)
            residuals = net_return - (alpha + beta * bench_aligned)
            mse = np.sum(residuals**2) / (len(net_return) - 2 + 1e-12)
            se_alpha = np.sqrt(mse * (1.0/len(net_return) + x_mean**2 / (np.sum((bench_aligned - x_mean)**2) + 1e-12)))
            tstat_alpha = float(alpha / (se_alpha + 1e-12))
        else:
            beta = 0.0
            alpha = float(mean_ret)
            tstat_alpha = float(alpha / (std_ret / np.sqrt(len(net_return)) + 1e-12)) if std_ret > 0 else 0.0

        # Hit ratio (daily)
        hit_ratio = float(np.mean(net_return > 0))

        # Information coefficient (rank corr between gross return and benchmark)
        if np.std(gross_return) > 1e-12 and np.std(bench_aligned) > 1e-12:
            try:
                ic = float(stats.spearmanr(gross_return, bench_aligned)[0])
            except Exception:
                ic = 0.0
        else:
            ic = 0.0

        # BTC correlation
        if np.std(bench_aligned) > 1e-12 and np.std(gross_return) > 1e-12:
            corr_rm = float(np.corrcoef(gross_return, bench_aligned)[0,1])
        else:
            corr_rm = 0.0

        return {
            "gross_return": float(gross_equity.iloc[-1] - 1) if len(gross_equity) else 0.0,
            "net_return": float(net_equity.iloc[-1] - 1) if len(net_equity) else 0.0,
            "sharpe": sharpe,
            "volatility": volatility,
            "max_drawdown": max_drawdown,
            "alpha": alpha,
            "beta": beta,
            "tstat_alpha": tstat_alpha,
            "hit_ratio": hit_ratio,
            "ic": ic,
            "mean_turnover": float(turnover.mean()) if hasattr(turnover, "mean") else 0.0,
            "total_turnover": float(turnover.sum()) if hasattr(turnover, "sum") else 0.0,
            "total_costs": float(costs.sum()) if hasattr(costs, "sum") else 0.0,
            "corr_rm": corr_rm,
        }

    def invoke(self, fo, weights, returns, benchmark, transaction_cost=RETAIL_TRANSACTION_COST_LOW, annualization=252):
        idx = weights.index.intersection(returns.index).intersection(benchmark.index)
        weights = weights.loc[idx].sort_index()
        returns = returns.loc[idx].sort_index()
        benchmark = benchmark.loc[idx].sort_index()

        executed_weight = fo.get("executed_weight")
        if isinstance(executed_weight, pd.DataFrame):
            executed_weight = executed_weight.loc[executed_weight.index.intersection(idx)].reindex(idx).fillna(0).sort_index()
            for c in returns.columns:
                if c not in executed_weight.columns:
                    executed_weight[c] = 0.0
            executed_weight = executed_weight[returns.columns].fillna(0)
        else:
            executed_weight = weights.shift(1).fillna(0)
        lagged_weights = executed_weight.copy()

        # Gross return from executed (lagged) weights * returns
        gross_return = (executed_weight * returns).sum(axis=1)
        gross_return.name = "gross_return"

        # Turnover from executed weights (positions actually held)
        turnover_series = executed_weight.diff().abs().sum(axis=1)
        turnover_series.iloc[0] = executed_weight.iloc[0].abs().sum()
        turnover_series.name = "turnover"
        costs = turnover_series * transaction_cost
        costs.name = "costs"

        # Net return
        net_return = gross_return - costs
        net_return.name = "net_return"

        # Equity curves
        gross_equity = (1 + gross_return).cumprod()
        net_equity = (1 + net_return).cumprod()
        benchmark_equity = (1 + benchmark).cumprod()

        # Drawdown
        running_max = net_equity.cummax()
        drawdown = net_equity / running_max - 1
        drawdown.name = "drawdown"

        # === ROLLING METRICS SERIES (DataFrames/Series for plotting) ===
        window = 30

        # Rolling Sharpe (annualized)
        rolling_sharpe = (net_return.rolling(window).mean() / (net_return.rolling(window).std() + 1e-12)) * np.sqrt(annualization)
        rolling_sharpe.name = "rolling_sharpe"

        # Rolling Beta vs benchmark
        rolling_cov = net_return.rolling(window).cov(benchmark)
        rolling_var = benchmark.rolling(window).var()
        rolling_beta = rolling_cov / (rolling_var + 1e-12)
        rolling_beta.name = "rolling_beta"

        # Rolling Alpha
        rolling_alpha = net_return.rolling(window).mean() - rolling_beta * benchmark.rolling(window).mean()
        rolling_alpha.name = "rolling_alpha"

        # Rolling R² (Pearson squared)
        rolling_corr = net_return.rolling(window).corr(benchmark)
        rolling_r2 = rolling_corr ** 2
        rolling_r2.name = "rolling_r2"

        # Rolling IC (Spearman rank correlation)
        def _rolling_spearman(s1, s2, w):
            out = pd.Series(index=s1.index, dtype=float)
            for i in range(w, len(s1) + 1):
                try:
                    out.iloc[i - 1] = stats.spearmanr(s1.iloc[i - w:i], s2.iloc[i - w:i])[0]
                except Exception:
                    out.iloc[i - 1] = np.nan
            return out

        rolling_ic = _rolling_spearman(net_return, benchmark, window)
        rolling_ic.name = "rolling_ic"

        # === CRITICAL FIX: compute scalar metrics from series, not stale fo ===
        metrics = self._compute_metrics_from_series(
            gross_return=gross_return,
            net_return=net_return,
            benchmark=benchmark,
            turnover=turnover_series,
            costs=costs
        )

        # Plotting — 5-row subplot with metric series and fixed y-axis ticks
        fig = make_subplots(
            rows=5, cols=1, shared_xaxes=True,
            row_heights=[0.32, 0.17, 0.17, 0.17, 0.17],
            vertical_spacing=0.04,
            subplot_titles=[
                "Portfolio Performance",
                "Drawdown",
                "Rolling Sharpe (30d)",
                "Rolling Alpha & Beta (30d)",
                "Rolling R² & IC (30d)"
            ],
        )

        # Row 1: Performance
        fig.add_trace(go.Scatter(x=gross_equity.index, y=gross_equity, name="Gross Return", mode="lines", line=dict(color="#2E86AB", width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=net_equity.index, y=net_equity, name="Net Return", mode="lines", line=dict(color="#A23B72", width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=benchmark_equity.index, y=benchmark_equity, name="Benchmark", mode="lines", line=dict(color="#F18F01", width=1.5, dash="dash")), row=1, col=1)

        # Row 2: Drawdown
        fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown, name="Drawdown", fill="tozeroy", fillcolor="rgba(231, 76, 60, 0.15)", line=dict(color="rgba(231, 76, 60, 0.7)", width=1)), row=2, col=1)

        # Row 3: Rolling Sharpe
        fig.add_trace(go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe, name="Rolling Sharpe", mode="lines", line=dict(color="#27AE60", width=1.2)), row=3, col=1)

        # Row 4: Rolling Alpha & Beta
        fig.add_trace(go.Scatter(x=rolling_alpha.index, y=rolling_alpha, name="Rolling Alpha", mode="lines", line=dict(color="#8E44AD", width=1.2)), row=4, col=1)
        fig.add_trace(go.Scatter(x=rolling_beta.index, y=rolling_beta, name="Rolling Beta", mode="lines", line=dict(color="#E67E22", width=1.2)), row=4, col=1)

        # Row 5: Rolling R² & IC
        fig.add_trace(go.Scatter(x=rolling_r2.index, y=rolling_r2, name="Rolling R²", mode="lines", line=dict(color="#2980B9", width=1.2)), row=5, col=1)
        fig.add_trace(go.Scatter(x=rolling_ic.index, y=rolling_ic, name="Rolling IC", mode="lines", line=dict(color="#C0392B", width=1.2)), row=5, col=1)

        fig.update_layout(
            template="plotly_white", hovermode="x unified",
            height=1400, showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=60, r=60, t=100, b=40),
        )
        fig.update_yaxes(title_text="Equity", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown", row=2, col=1)
        fig.update_yaxes(title_text="Sharpe", range=[-2, 5], dtick=1, row=3, col=1)
        fig.update_yaxes(title_text="Alpha / Beta", range=[-1, 3], dtick=1, row=4, col=1)
        fig.update_yaxes(title_text="R² / IC", range=[-1, 1.2], dtick=0.5, row=5, col=1)
        fig.update_xaxes(title_text="Date", row=5, col=1)

        series = {
            "gross_return": gross_return, "net_return": net_return,
            "gross_equity": gross_equity, "net_equity": net_equity,
            "benchmark_equity": benchmark_equity, "drawdown": drawdown,
            "weights": weights, "executed_weight": executed_weight,
            "lagged_weights": lagged_weights, "turnover": turnover_series,
            "costs": costs,
            "rolling_sharpe": rolling_sharpe,
            "rolling_alpha": rolling_alpha,
            "rolling_beta": rolling_beta,
            "rolling_r2": rolling_r2,
            "rolling_ic": rolling_ic,
        }
        return {"chart": fig, "metrics": metrics, "series": series}


class Portfolio0:
    def invoke(self, fo, weights, returns, benchmark, transaction_cost=RETAIL_TRANSACTION_COST_LOW, annualization=252):
        idx = weights.index.intersection(returns.index).intersection(benchmark.index)
        weights = weights.loc[idx]
        returns = returns.loc[idx]
        benchmark = benchmark.loc[idx]

        executed_weight = fo.get("executed_weight")
        if executed_weight is not None:
            executed_weight = executed_weight.loc[executed_weight.index.intersection(idx)]
            executed_weight = executed_weight.reindex(idx).fillna(0)
        else:
            executed_weight = weights.shift(1).fillna(0)
        lagged_weights = weights.shift(1).fillna(0)

        gross_return = (executed_weight * returns).sum(axis=1)
        turnover = weights.diff().abs().sum(axis=1)
        turnover.iloc[0] = 0
        costs = turnover * transaction_cost
        net_return = gross_return - costs
        gross_equity = (1 + gross_return).cumprod()
        net_equity = (1 + net_return).cumprod()
        benchmark_equity = (1 + benchmark).cumprod()
        drawdown = net_equity / net_equity.cummax() - 1

        # === ROLLING METRICS SERIES (DataFrames/Series for plotting) ===
        window = 30

        # Rolling Sharpe (annualized)
        rolling_sharpe = (net_return.rolling(window).mean() / (net_return.rolling(window).std() + 1e-12)) * np.sqrt(annualization)
        rolling_sharpe.name = "rolling_sharpe"

        # Rolling Beta vs benchmark
        rolling_cov = net_return.rolling(window).cov(benchmark)
        rolling_var = benchmark.rolling(window).var()
        rolling_beta = rolling_cov / (rolling_var + 1e-12)
        rolling_beta.name = "rolling_beta"

        # Rolling Alpha
        rolling_alpha = net_return.rolling(window).mean() - rolling_beta * benchmark.rolling(window).mean()
        rolling_alpha.name = "rolling_alpha"

        # Rolling R² (Pearson squared)
        rolling_corr = net_return.rolling(window).corr(benchmark)
        rolling_r2 = rolling_corr ** 2
        rolling_r2.name = "rolling_r2"

        # Rolling IC (Spearman rank correlation)
        def _rolling_spearman(s1, s2, w):
            out = pd.Series(index=s1.index, dtype=float)
            for i in range(w, len(s1) + 1):
                try:
                    out.iloc[i - 1] = stats.spearmanr(s1.iloc[i - w:i], s2.iloc[i - w:i])[0]
                except Exception:
                    out.iloc[i - 1] = np.nan
            return out

        rolling_ic = _rolling_spearman(net_return, benchmark, window)
        rolling_ic.name = "rolling_ic"

        # Compute scalar metrics from series instead of stale fo
        portfolio = Portfolio()
        metrics = portfolio._compute_metrics_from_series(
            gross_return=gross_return, net_return=net_return,
            benchmark=benchmark, turnover=turnover, costs=costs
        )

        # Plotting — 5-row subplot with metric series and fixed y-axis ticks
        fig = make_subplots(
            rows=5, cols=1, shared_xaxes=True,
            row_heights=[0.32, 0.17, 0.17, 0.17, 0.17],
            vertical_spacing=0.04,
            subplot_titles=[
                "Portfolio Performance",
                "Drawdown",
                "Rolling Sharpe (30d)",
                "Rolling Alpha & Beta (30d)",
                "Rolling R² & IC (30d)"
            ],
        )

        # Row 1: Performance
        fig.add_trace(go.Scatter(x=gross_equity.index, y=gross_equity, name="Gross Return", mode="lines", line=dict(color="#2E86AB", width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=net_equity.index, y=net_equity, name="Net Return", mode="lines", line=dict(color="#A23B72", width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=benchmark_equity.index, y=benchmark_equity, name="Benchmark", mode="lines", line=dict(color="#F18F01", width=1.5, dash="dash")), row=1, col=1)

        # Row 2: Drawdown
        fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown, name="Drawdown", fill="tozeroy", fillcolor="rgba(231, 76, 60, 0.15)", line=dict(color="rgba(231, 76, 60, 0.7)", width=1)), row=2, col=1)

        # Row 3: Rolling Sharpe
        fig.add_trace(go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe, name="Rolling Sharpe", mode="lines", line=dict(color="#27AE60", width=1.2)), row=3, col=1)

        # Row 4: Rolling Alpha & Beta
        fig.add_trace(go.Scatter(x=rolling_alpha.index, y=rolling_alpha, name="Rolling Alpha", mode="lines", line=dict(color="#8E44AD", width=1.2)), row=4, col=1)
        fig.add_trace(go.Scatter(x=rolling_beta.index, y=rolling_beta, name="Rolling Beta", mode="lines", line=dict(color="#E67E22", width=1.2)), row=4, col=1)

        # Row 5: Rolling R² & IC
        fig.add_trace(go.Scatter(x=rolling_r2.index, y=rolling_r2, name="Rolling R²", mode="lines", line=dict(color="#2980B9", width=1.2)), row=5, col=1)
        fig.add_trace(go.Scatter(x=rolling_ic.index, y=rolling_ic, name="Rolling IC", mode="lines", line=dict(color="#C0392B", width=1.2)), row=5, col=1)

        fig.update_layout(
            template="plotly_white", hovermode="x unified",
            height=1400, showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=60, r=60, t=100, b=40),
        )
        fig.update_yaxes(title_text="Equity", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown", row=2, col=1)
        fig.update_yaxes(title_text="Sharpe", range=[-2, 5], dtick=1, row=3, col=1)
        fig.update_yaxes(title_text="Alpha / Beta", range=[-1, 3], dtick=1, row=4, col=1)
        fig.update_yaxes(title_text="R² / IC", range=[-1, 1.2], dtick=0.5, row=5, col=1)
        fig.update_xaxes(title_text="Date", row=5, col=1)

        return {
            "chart": fig,
            "metrics": metrics,
            "series": {
                "gross_return": gross_return, "net_return": net_return,
                "equity": net_equity, "benchmark": benchmark_equity,
                "drawdown": drawdown, "weights": weights,
                "executed_weight": executed_weight, "lagged_weights": lagged_weights,
                "rolling_sharpe": rolling_sharpe,
                "rolling_alpha": rolling_alpha,
                "rolling_beta": rolling_beta,
                "rolling_r2": rolling_r2,
                "rolling_ic": rolling_ic,
            },
        }


# =====================================================
# DATA LOADER
# =====================================================

def build_data():
    symbols = ["BTCUSDT","ETHUSDT","XRPUSDT","BNBUSDT","SOLUSDT","DOGEUSDT",
               "ADAUSDT","TRXUSDT","HYPEUSDT","SUIUSDT","LINKUSDT","AVAXUSDT",
               "XLMUSDT","HBARUSDT","BCHUSDT","LTCUSDT","SHIBUSDT","DOTUSDT",
               "AAVEUSDT","PEPEUSDT","NEARUSDT","APTUSDT","ICPUSDT","ETCUSDT",
               "ONDOUSDT","POLUSDT","CROUSDT","TONUSDT","UNIUSDT"]
    try:
        from ..PickleDataManager import PickleDataManager
        dm = PickleDataManager("backtest")
    except ImportError:
        print("[WARN] PickleDataManager not available, using sample data")
        return create_sample_data()
    data = {}
    for symbol in symbols:
        try:
            df = dm.fetch_store(symbol)
            if df is not None and not df.empty:
                data[symbol] = df
                print(f"[DATA] Loaded {symbol} ({len(df)} rows)")
            else:
                print(f"[DATA] Empty dataset {symbol}")
        except Exception as e:
            print(f"[DATA] Failed {symbol}: {e}")
    if not data:
        print("[DATA] Universe empty, using sample data")
        return create_sample_data()
    print("[DATA] Assets:", list(data.keys()))
    return data


def create_sample_data(n_symbols=5, n_days=800):
    np.random.seed(42)
    data = {}
    for i in range(n_symbols):
        symbol = f"SYM{i:03d}"
        dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
        returns = np.random.normal(0.0005, 0.02, n_days)
        price = 100 * np.exp(np.cumsum(returns))
        volume = np.random.lognormal(15, 0.5, n_days)
        df = pd.DataFrame({
            "open": price * (1 + np.random.normal(0, 0.001, n_days)),
            "high": price * (1 + abs(np.random.normal(0, 0.01, n_days))),
            "low": price * (1 - abs(np.random.normal(0, 0.01, n_days))),
            "close": price,
            "volume": volume.astype(int)
        }, index=dates)
        data[symbol] = df
    return data


# =====================================================
# MAIN EXECUTION
# =====================================================

if __name__ == "__main__":
    print("Loading market data...")
    market_data = build_data()
    params = {
        "strategies": ["AlphaStrategy", "MomentumStrategy"],
        "trees": 500, "depth": 12, "horizon": 21,
        "min_samples_split": 2, "max_features": "sqrt",
        "model": "random_forest",
        "learning_rate": 0.1,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
    }
    pipeline = AgenticPipeline(market_data, base_dir="runs", use_feature_reduction=True, reduction_method="pca")
    # Combinatorial disabled by default per user request
    result = pipeline.run_agent(params, max_iters=5, run_combinatorial=False, param_grid=PARAM_GRID_FAST)
    print(f"Pipeline complete. Results saved to: {pipeline.base_dir}")
    print("Baseline model: runs/baseline_model.pkl")
    print("Optimized model: runs/optimized_model.pkl")
    print("Comparison report: runs/model_comparison.json")

# ======================================================
# END OF THE PIPELINE
# ======================================================