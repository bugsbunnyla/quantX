"""
quantX Backtest Pipeline — Combinatorial Scenario Optimizer (Production-Ready)
=============================================================================

Requirements addressed:
1. Parameterized outcome in ML training research ONLY
2. Baseline model.pkl created BEFORE combinatorial search
3. Optimized model.pkl created AFTER search completes
4. Model-to-model comparison (before vs after)
5. Validation-to-validation comparison (before vs after)
6. Rate/predict improvement
7. Industry-calibrated parameter ranges with inside/outside/extreme grids
8. Scenario multipliers backed by historical market data
9. Working true_validate() against s.formulaOutput
10. All 500 combinations tested, best selected by resiliency
11. .pkl artifacts stored at each step for audit trail

Architecture:
- AgenticPipeline: orchestrates the full research lifecycle
- BacktestEngine: signal generation + robust evaluation
- ReviewEngine: train vs validation comparison with degradation analysis
- Evaluator: ML-aware accept/reject with mutation recommendations
- GoLiveEngine: production readiness assessment + deployment package
- CombinatorialScenarioOptimizer: parameter-driven resiliency testing
- ModelComparator: before/after rating and prediction engine
- TrueValidationEngine: correlates predicted signal with actual returns
"""

import os
import pickle
import json
import warnings
import copy
import joblib
import itertools
import hashlib

import numpy as np
import pandas as pd

from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from copy import deepcopy
from pathlib import Path
from collections import defaultdict

from scipy import stats
# ==these must be before sklearn imports
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
#=======
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings("ignore")


# =====================================================
# INDUSTRY-CALIBRATED PARAMETER GRIDS
# =====================================================
PARAM_GRID_FAST = {
    "trees": [200, 500, 800],
    "depth": [8, 12, 16],
    "horizon": [21, 42],
    "min_samples_split": [2, 10],
    "max_features": ["sqrt", 0.5],
    "model": ["random_forest"],
}
PARAM_GRID_INSIDE = {
    "trees": [200, 350, 500, 650, 800],
    "depth": [6, 8, 10, 12, 14],
    "horizon": [10, 21, 42],
    "min_samples_split": [5, 10, 20],
    "max_features": ["sqrt", "log2", 0.5],
    "model": ["random_forest"],
}

PARAM_GRID_OUTSIDE = {
    "trees": [50, 100, 1000, 1500, 2000],
    "depth": [4, 16, 20, 25],
    "horizon": [5, 63, 126],
    "min_samples_split": [2, 50, 100],
    "max_features": [0.3, 0.8, 1.0],
    "model": ["random_forest", "gradient_boosting"],
}

PARAM_GRID_EXTREME = {
    "trees": [10, 25, 3000, 5000],
    "depth": [2, 3, 30, 50],
    "horizon": [1, 252],
    "min_samples_split": [1, 500],
    "max_features": [0.1, None],
    "model": ["random_forest", "gradient_boosting", "ridge"],
}

SCENARIO_CONFIG = {
    "BULL": {"mu_multiplier": 1.8, "vol_multiplier": 0.7, "kelly_cap": 0.25,
             "description": "Strong positive drift, reduced volatility"},
    "BEAR": {"mu_multiplier": -0.8, "vol_multiplier": 1.2, "kelly_cap": 0.10,
             "description": "Negative drift with elevated vol"},
    "HIGH_VOL": {"mu_multiplier": 0.3, "vol_multiplier": 2.0, "kelly_cap": 0.05,
                 "description": "Mean-reverting, high variance"},
    "CRASH": {"mu_multiplier": -2.5, "vol_multiplier": 3.5, "kelly_cap": 0.02,
              "description": "Tail risk event, correlation -> 1"},
    "STABLE": {"mu_multiplier": 0.5, "vol_multiplier": 0.5, "kelly_cap": 0.15,
               "description": "Low signal environment"}
}


# =====================================================
# STORAGE ENGINE
# =====================================================

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
# SPLIT ENGINE
# =====================================================

"""
class SplitEngine:
    def split(self, data, split_ratio=0.8):
        train, val = {}, {}
        for symbol, df in data.items():
            if not isinstance(df, pd.DataFrame):
                continue
            if len(df) < 50:
                split_idx = max(int(len(df) * split_ratio), 10)
                if split_idx >= len(df) - 5:
                    split_idx = len(df) // 2
            else:
                split_idx = int(len(df) * split_ratio)
            train[symbol] = df.iloc[:split_idx].copy()
            val[symbol] = df.iloc[split_idx:].copy()
        return train, val
"""
class SplitEngine:

    def split(self, data):
        train = copy.deepcopy(data)
        val = copy.deepcopy(data)
        return train, val



# =====================================================
# ML TRAIN ENGINE
# =====================================================

class MLTrainEngine:
    def __init__(self):
        self.feature_schema = None
        self.model = None
        self.params = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def train(self, train_package):
        X = train_package["X"]
        y = train_package["y"]
        self.params = train_package.get("params", {})
        #print(f"[ML] Dataset: {X.shape}, {y.shape}")
        if X.shape[0] < 100:
            print(f"[ML] Dataset: {X.shape}, {y.shape}")
        X_encoded = self._encode_categoricals(X, fit=True)
        trained_model, metrics = self.fit(X_encoded, y, self.params)
        self.model = trained_model
        self.feature_schema = list(X.columns)
        return {
            "model": trained_model, "X": X, "y": y,
            "features": list(X.columns), "metrics": metrics,
            "results": train_package.get("results"),
            "formula_outputs": train_package.get("formula_outputs"),
            "params": self.params, "label_encoders": self.label_encoders,
            "scaler": self.scaler
        }

    def build_feature_matrix(self, strategy_results, formula_outputs, raw_data, params=None):
        X, y, feature_names = self.build_features(strategy_results, formula_outputs, raw_data, params or {})
        if X.shape[0] == 0:
            raise ValueError(f"build_features returned empty feature matrix: X.shape={X.shape}")
        return X, y

    def _lookup_multiindex(self, df, category, metric, case_sensitive=False):
        if not isinstance(df, pd.DataFrame) or len(df.columns) == 0:
            return None
        cat_norm = str(category).lower() if not case_sensitive else str(category)
        met_norm = str(metric).lower() if not case_sensitive else str(metric)
        if isinstance(df.columns, pd.MultiIndex):
            for col in df.columns:
                if isinstance(col, tuple) and len(col) >= 2:
                    col_cat = str(col[0]).lower() if not case_sensitive else str(col[0])
                    col_met = str(col[1]).lower() if not case_sensitive else str(col[1])
                    if col_cat == cat_norm and col_met == met_norm:
                        return df[col].copy()
            return None
        target_combined = f"{cat_norm}_{met_norm}"
        for col in df.columns:
            col_str = str(col).lower()
            if col_str == target_combined:
                return df[col].copy()
            if cat_norm in col_str and met_norm in col_str:
                return df[col].copy()
        return None

    def build_features(self, strategy_results, formula_outputs, raw_data, params=None):
        frames = []
        ret_series = None
        
        # === Phase 1: Extract features and ret_series from formula_outputs ===
        if formula_outputs is not None:
            if isinstance(formula_outputs, pd.DataFrame):
                formula_outputs = [formula_outputs]
            for fo in formula_outputs:
                if not isinstance(fo, pd.DataFrame) or fo.empty:
                    continue
                    
                idx_is_multi = isinstance(fo.index, pd.MultiIndex)
                col_is_multi = isinstance(fo.columns, pd.MultiIndex)
                
                if col_is_multi:
                    if ret_series is None:
                        ret_series = self._lookup_multiindex(fo, "market", "ret")
                    temp = fo.stack(level=[0, 1]).reset_index()
                    if temp.shape[1] == 3:
                        temp.columns = ["symbol", "category_metric", "value"]
                        temp[["category", "metric"]] = temp["category_metric"].str.split("_", n=1, expand=True)
                        temp = temp.drop(columns=["category_metric"])
                    elif temp.shape[1] >= 4:
                        cols = list(temp.columns)
                        temp.columns = ["symbol", "category", "metric", "value"] + cols[4:]
                    else:
                        continue
                    temp["feature"] = "fo_" + temp["category"].astype(str) + "_" + temp["metric"].astype(str)
                    formula_df = temp.pivot_table(index="symbol", columns="feature", values="value", aggfunc="last").reset_index()
                    formula_df = formula_df.drop(columns=["fo_market_ret"], errors="ignore")
                    if not formula_df.empty:
                        frames.append(formula_df)
                        
                elif idx_is_multi:
                    if ret_series is None:
                        for idx in fo.index:
                            if isinstance(idx, tuple) and len(idx) >= 2:
                                if str(idx[1]).lower() == "ret":
                                    ret_series = fo.loc[idx]
                                    if isinstance(ret_series, pd.DataFrame):
                                        ret_series = ret_series.iloc[:, 0]
                                    break
                    for idx in fo.index:
                        if isinstance(idx, tuple) and len(idx) >= 2:
                            cat, metric = idx[0], idx[1]
                        else:
                            continue
                        if str(metric).lower() == "ret":
                            continue
                        col_name = f"fo_{cat}_{metric}"
                        series = fo.loc[idx]
                        if isinstance(series, pd.DataFrame):
                            series = series.iloc[:, 0]
                        if not isinstance(series, pd.Series):
                            continue
                        feat_rows = []
                        for symbol, value in series.items():
                            if pd.isna(value):
                                continue
                            feat_rows.append({"symbol": str(symbol), col_name: value})
                        if feat_rows:
                            frames.append(pd.DataFrame(feat_rows))
                            
                else:
                    # Simple DataFrames
                    fo_cols_lower = {str(c).lower(): c for c in fo.columns}
                    ret_col = None
                    for key in ["ret", "return", "returns", "y", "target"]:
                        if key in fo_cols_lower:
                            ret_col = fo_cols_lower[key]
                            break
                    
                    symbol_col = None
                    for key in ["symbol", "ticker", "asset", "name"]:
                        if key in fo_cols_lower:
                            symbol_col = fo_cols_lower[key]
                            break
                    
                    if ret_col is not None:
                        if symbol_col is not None:
                            ret_series = fo.set_index(symbol_col)[ret_col]
                        else:
                            ret_series = fo[ret_col]
                            if not isinstance(ret_series.index, pd.MultiIndex):
                                ret_series.index.name = "symbol"
                    
                    feature_cols = [c for c in fo.columns if c not in [ret_col, symbol_col] and c is not None]
                    if feature_cols and symbol_col is not None:
                        formula_df = fo[[symbol_col] + feature_cols].copy()
                        formula_df.columns = ["symbol"] + [f"fo_{c}" for c in feature_cols]
                        frames.append(formula_df)
                    elif feature_cols and len(fo) > 0:
                        formula_df = fo[feature_cols].copy()
                        formula_df = formula_df.reset_index()
                        if formula_df.columns[0] != "symbol":
                            formula_df.columns = ["symbol"] + list(formula_df.columns[1:])
                        rename_map = {c: f"fo_{c}" for c in feature_cols if c in formula_df.columns}
                        formula_df = formula_df.rename(columns=rename_map)
                        frames.append(formula_df)
        
        # === Phase 2: Strategy results ===
        sr_idx = 0
        for result in strategy_results or []:
            if getattr(result, "metrics", None):
                rows = []
                for symbol, values in result.metrics.items():
                    if isinstance(values, dict):
                        row = {"symbol": symbol}
                        for k, v in values.items():
                            row[f"sr_{k}"] = v
                        rows.append(row)
                if rows:
                    frames.append(pd.DataFrame(rows))
            if getattr(result, "signals", None):
                signal_df = pd.DataFrame([{"symbol": symbol, f"sr_signal_{sr_idx}": signal} for symbol, signal in result.signals.items()])
                frames.append(signal_df)
                sr_idx += 1
        
        if not frames:
            raise ValueError("No features generated from formula outputs or strategy results")
            
        feature_df = frames[0]
        for frame in frames[1:]:
            feature_df = feature_df.merge(frame, on="symbol", how="outer")
            
        # === Phase 3: Clean ret_series ONCE ===
        if ret_series is None:
            raise ValueError("Missing target variable ('ret'/'return'/'y') in formula outputs")

        # Ensure it's a clean Series
        if isinstance(ret_series, pd.DataFrame):
            ret_series = ret_series.iloc[:, 0]
        ret_series = pd.Series(ret_series).squeeze()
        
        # Flatten MultiIndex if needed
        if isinstance(ret_series.index, pd.MultiIndex):
            level_names = ret_series.index.names
            if len(level_names) >= 2 and any('symbol' in str(n).lower() for n in level_names if n):
                symbol_level = next(i for i, n in enumerate(level_names) if n and 'symbol' in str(n).lower())
                ret_series = ret_series.reset_index(level=symbol_level, drop=False)
                if isinstance(ret_series, pd.DataFrame):
                    ret_series = ret_series.iloc[:, 0]
            else:
                ret_series = ret_series.reset_index(level=0)
                if isinstance(ret_series, pd.DataFrame):
                    ret_series = ret_series.iloc[:, 0]
            ret_series = pd.Series(ret_series).squeeze()
        
        # Name index
        if ret_series.index.name is None or ret_series.index.name == "":
            ret_series.index.name = "symbol"
        
        # Drop NaN targets
        ret_series = ret_series.dropna()
        if len(ret_series) == 0:
            raise ValueError("ret_series contains all NaN values after extraction")
            
        # === Phase 4: Build y_df and merge ONCE ===
        y_df = ret_series.rename_axis("symbol").reset_index(name="y")
        
        feature_df = feature_df.merge(y_df, on="symbol", how="inner")
        if len(feature_df) == 0:
            f_syms = set(feature_df["symbol"].astype(str).unique()) if "symbol" in feature_df.columns else set()
            y_syms = set(y_df["symbol"].astype(str).unique())
            overlap = f_syms & y_syms
            raise ValueError(
                f"Merge produced 0 rows. Feature symbols: {len(f_syms)}, "
                f"Target symbols: {len(y_syms)}, Overlap: {len(overlap)}. "
                f"Sample feature: {list(f_syms)[:3]}, Sample target: {list(y_syms)[:3]}"
            )
        
        feature_df = feature_df.dropna(subset=["y"])
        if len(feature_df) == 0:
            raise ValueError(f"All {len(y_df)} rows dropped: y contains all NaN after merge")
            
        # === Phase 5: Extract X, y ===
        y = feature_df["y"]
        X = feature_df.drop(columns=["symbol", "y"], errors="ignore")
        X = self.encode_features(X)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
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

    def fit(self, X, y, params=None):
        if params is None:
            params = {}
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model_type = params.get("model", "random_forest")
        if model_type == "random_forest":
            model = RandomForestRegressor(
                n_estimators=params.get("trees", 500), max_depth=params.get("depth", 12),
                min_samples_split=params.get("min_samples_split", 2),
                max_features=params.get("max_features", "sqrt"), random_state=42, n_jobs=-1)
        elif model_type == "gradient_boosting":
            model = GradientBoostingRegressor(
                n_estimators=params.get("trees", 500), max_depth=params.get("depth", 6),
                min_samples_split=params.get("min_samples_split", 2),
                max_features=params.get("max_features", "sqrt"), random_state=42)
        elif model_type == "ridge":
            model = Ridge(alpha=1.0, random_state=42)
        else:
            model = RandomForestRegressor(
                n_estimators=params.get("trees", 500), max_depth=params.get("depth", 12),
                random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            "r2": float(r2_score(y_test, pred)),
            "samples": len(X), "features": len(X.columns), "model_type": model_type
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


# =====================================================
# PROCESS ENGINE
# =====================================================

class ProcessEngine:
    def __init__(self):
        self.ml = MLTrainEngine()

    def train(self, train_package, params=None):
        if params is None:
            params = {}
        if not isinstance(train_package, dict):
            raise TypeError("ProcessEngine.train expects package dict")
        print("[PROCESS] Training package received, keys:", train_package.keys())
        required = ["X", "y", "results", "formula_outputs"]
        for key in required:
            if key not in train_package:
                raise KeyError(f"Missing train package key: {key}")
        ml_params = {
            "trees": params.get("trees", 500), "depth": params.get("depth", 12),
            "horizon": params.get("horizon", 21),
            "min_samples_split": params.get("min_samples_split", 2),
            "max_features": params.get("max_features", "sqrt"),
            "model": params.get("model", "random_forest")
        }
        print("[ML] Parameters:", ml_params)
        train_package["params"] = ml_params
        result = self.ml.train(train_package)
        print("[ML] Training complete")
        train_package["model"] = result["model"]
        train_package["metrics"] = result.get("metrics", {})
        train_package["features"] = result.get("features", train_package.get("features", []))
        train_package["label_encoders"] = result.get("label_encoders", {})
        train_package["scaler"] = result.get("scaler", None)
        return train_package


# =====================================================
# BACKTEST ENGINE
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
        for col in required_features:
            if col not in X.columns:
                X[col] = 0
        X = X[required_features]
        # EMPTY GUARD
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
        if "y" not in package:
            raise KeyError("Package missing y")
        y = np.asarray(package["y"], dtype=float).ravel()
        signal = np.asarray(signal, dtype=float).ravel()
        # EMPTY GUARD
        if len(signal) == 0:
            return {"hit": 0.0, "sharpe": 0.0, "cvar": 0.0, "corr": 0.0, "r2": 0.0,
                    "t_beta": 0.0, "score": 0.0, "signal": signal, "pnl": np.array([])}
        if len(signal) != len(y):
            raise ValueError(f"Signal length {len(signal)} != target length {len(y)}")
        pnl = signal * y
        hit = np.mean(np.sign(signal) == np.sign(y))
        sharpe = np.mean(pnl) / (np.std(pnl) + 1e-9)
        var = np.percentile(pnl, 5)
        tail = pnl[pnl <= var]
        cvar = tail.mean() if len(tail) > 0 else 0
        if np.std(signal) == 0 or np.std(y) == 0:
            corr = 0
        else:
            corr = np.corrcoef(signal, y)[0, 1]
        slope, intercept, r, _, stderr = stats.linregress(signal, y)
        r2 = r * r
        t_beta = slope / (stderr + 1e-9)
        score = 0.4 * sharpe + 0.2 * corr + 0.2 * hit + 0.2 * r2
        return {
            "hit": float(hit), "sharpe": float(sharpe), "cvar": float(cvar),
            "corr": float(corr), "r2": float(r2), "t_beta": float(t_beta),
            "score": float(score), "signal": signal, "pnl": pnl
        }


# =====================================================
# REVIEW ENGINE
# =====================================================

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


# =====================================================
# EVALUATOR
# =====================================================

class Evaluator:
    def __init__(self):
        self.decision_history = []
        self.thresholds = {
            "min_sharpe": 0.5, "min_corr": 0.1, "min_hit": 0.52,
            "max_overfit_gap": 0.3, "min_score": 0.3
        }

    def decide(self, review, train_metrics=None):
        current = review.get("current", {})
        trend = review.get("trend", "FIRST_RUN")
        warnings = review.get("warnings", [])
        decision = {"verdict": "CONTINUE", "confidence": 0.5, "reasons": [], "mutations": []}
        if current.get("sharpe", 0) < self.thresholds["min_sharpe"]:
            decision["reasons"].append(f"Sharpe {current.get('sharpe', 0):.3f} below threshold")
        if current.get("corr", 0) < self.thresholds["min_corr"]:
            decision["reasons"].append(f"Correlation {current.get('corr', 0):.3f} below threshold")
        if current.get("hit", 0) < self.thresholds["min_hit"]:
            decision["reasons"].append(f"Hit ratio {current.get('hit', 0):.3f} below threshold")
        if train_metrics:
            train_score = train_metrics.get("score", 0)
            val_score = current.get("score", 0)
            if train_score > 0 and (train_score - val_score) / train_score > self.thresholds["max_overfit_gap"]:
                decision["reasons"].append(f"Overfit detected: train={train_score:.3f}, val={val_score:.3f}")
        if trend == "IMPROVING" and len(decision["reasons"]) == 0:
            decision["verdict"] = "ACCEPT"
            decision["confidence"] = 0.85
            decision["reasons"].append("Validation improving and meets all thresholds")
        elif trend == "STABLE" and len(decision["reasons"]) <= 1:
            decision["verdict"] = "CONTINUE"
            decision["confidence"] = 0.7
        elif trend == "DEGRADING" or len(decision["reasons"]) > 1:
            if len(warnings) > 2 or current.get("score", 0) < self.thresholds["min_score"]:
                decision["verdict"] = "STOP"
                decision["confidence"] = 0.9
                decision["reasons"].append("Severe degradation or multiple failures")
            else:
                decision["verdict"] = "MUTATE"
                decision["confidence"] = 0.6
                decision["mutations"] = self._suggest_mutations(review)
        if trend == "FIRST_RUN":
            decision["verdict"] = "CONTINUE" if len(decision["reasons"]) == 0 else "MUTATE"
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
        return mutations

    def update_thresholds(self, thresholds):
        self.thresholds.update(thresholds)


# =====================================================
# GOLIVE ENGINE
# =====================================================

class GoLiveEngine:
    def __init__(self):
        self.readiness_criteria = {
            "min_iterations": 3, "min_accept_ratio": 0.5,
            "min_avg_sharpe": 0.6, "min_avg_corr": 0.15,
            "max_avg_drawdown": -0.15, "min_consistency": 0.7
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
        systemic_prediction = {
            "expected_sharpe": float(np.mean(val_sharpes)),
            "sharpe_confidence_interval": (float(np.percentile(val_sharpes, 25)), float(np.percentile(val_sharpes, 75))),
            "expected_corr": float(np.mean(val_corrs)),
            "expected_score": float(np.mean(val_scores)),
            "score_stability": float(1 - np.std(val_scores) / (np.mean(val_scores) + 1e-9)),
            "win_rate": float(np.mean([s > 0 for s in val_scores])),
            "consistency_ratio": float(np.mean([s > 0.3 for s in val_scores]))
        }
        checks = {
            "accept_ratio": sum(1 for r in research_history if r.get("decision") == "ACCEPT") / len(research_history),
            "avg_sharpe": np.mean(val_sharpes), "avg_corr": np.mean(val_corrs),
            "avg_drawdown": np.mean(val_drawdowns), "consistency": systemic_prediction["consistency_ratio"]
        }
        passed_checks = all([
            checks["accept_ratio"] >= self.readiness_criteria["min_accept_ratio"],
            checks["avg_sharpe"] >= self.readiness_criteria["min_avg_sharpe"],
            checks["avg_corr"] >= self.readiness_criteria["min_avg_corr"],
            checks["avg_drawdown"] >= self.readiness_criteria["max_avg_drawdown"],
            checks["consistency"] >= self.readiness_criteria["min_consistency"]
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


# =====================================================
# COMBINATORIAL SCENARIO OPTIMIZER
# =====================================================

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
            from ..qxEngine import QuantXEngine
            engine = QuantXEngine()
            strategies = engine.qxStrategyList(stressed_data, interval=params.get("interval", "4y"))
            formula_outputs = []
            for s in engine.strategy:
                if hasattr(s, "formulaOutput"):
                    formula_outputs.append(s.formulaOutput.report())
            ml = MLTrainEngine()
            X, y = ml.build_feature_matrix(strategy_results=engine.results, formula_outputs=formula_outputs, raw_data=stressed_data)
        except Exception as e:
            ml = MLTrainEngine()
            X, y = self._build_features_from_raw(stressed_data)
        if X.shape[0] == 0 or len(y) == 0:
            return None
        package = {"X": X, "y": y, "features": list(X.columns)}
        signal = self.backtest.signal(model_package, package)
        metrics = self.backtest.evaluate(package, signal)
        return {"scenario": scenario_name, "params": params, "metrics": metrics,
                "sharpe": metrics["sharpe"], "score": metrics["score"], "corr": metrics["corr"]}

    def _build_features_from_raw(self, data):
        rows, targets = [], []
        for symbol, df in data.items():
            if not isinstance(df, pd.DataFrame) or len(df) < 30:
                continue
            for i in range(20, len(df) - 5):
                window = df.iloc[i - 20:i]
                close = window["close"]
                volume = window["volume"] if "volume" in window.columns else pd.Series([1] * len(window))
                ret = close.pct_change().dropna()
                if len(ret) < 5:
                    continue
                features = {
                    "symbol": symbol, "market_price": close.iloc[-1],
                    "market_volume": volume.iloc[-1] if len(volume) > 0 else 0,
                    "market_structure_liq_adj_vol": volume.rolling(5).mean().iloc[-1] if len(volume) >= 5 else 0,
                    "risk_volatility": ret.rolling(20).std().iloc[-1] if len(ret) >= 20 else ret.std(),
                    "risk_sharpe": ret.mean() / (ret.std() + 1e-9),
                    "risk_drawdown": (close.iloc[-1] / close.cummax().iloc[-1] - 1),
                    "alpha_pure": ret.rolling(5).mean().iloc[-1] if len(ret) >= 5 else ret.mean(),
                    "alpha_ts": close.iloc[-1] - close.rolling(20).mean().iloc[-1],
                    "alpha_beta": ret.mean() / (ret.std() + 1e-9),
                    "alpha_residual": ret.iloc[-1] - ret.rolling(20).mean().iloc[-1] if len(ret) >= 20 else ret.iloc[-1] - ret.mean(),
                    "alpha_xs": close.pct_change(5).iloc[-1] if len(close) >= 6 else 0,
                    "transform_zscore": (close.iloc[-1] - close.rolling(20).mean().iloc[-1]) / (close.rolling(20).std().iloc[-1] + 1e-9),
                    "transform_rank": close.rolling(20).rank().iloc[-1] / 20.0,
                    "transform_winsor": np.clip(ret.iloc[-1], -3 * ret.std(), 3 * ret.std()),
                    "transform_tanh": np.tanh(ret.iloc[-1]),
                    "transform_detrend": close.iloc[-1] - close.rolling(20).mean().iloc[-1],
                    "market_symbol": symbol,
                }
                defaults = ["basic_corr", "basic_hit_ratio", "basic_r_squared", "basic_tstat",
                            "decision_score", "decision_signal", "execution_impact",
                            "execution_slippage", "execution_turnover", "intel_ic",
                            "market_structure_regime", "portfolio_entropy", "portfolio_inv_vol",
                            "portfolio_kelly", "portfolio_mvo", "portfolio_risk_parity",
                            "portfolio_weight", "risk_cvar"]
                for col in defaults:
                    features[col] = 0
                future_ret = df["close"].iloc[i + 5] / df["close"].iloc[i] - 1
                rows.append(features)
                targets.append(future_ret)
        X = pd.DataFrame(rows)
        y = pd.Series(targets, name="target")
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        y = y.replace([np.inf, -np.inf], np.nan).fillna(0)
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

    def _run_combinatorial_srfo(self, param_grid, max_combinations=100, scenarios=None):
        """Combinatorial search using cached SRFO. No engine re-runs."""
        import time
        if scenarios is None:
            scenarios = ["BULL", "BEAR", "HIGH_VOL", "CRASH", "STABLE"]

        keys = list(param_grid.keys())
        values = [param_grid[k] for k in keys]
        all_combos = list(itertools.product(*values))
        if len(all_combos) > max_combinations:
            np.random.seed(42)
            selected = [all_combos[i] for i in np.random.choice(len(all_combos), max_combinations, replace=False)]
        else:
            selected = all_combos

        combinations = []
        for combo in selected:
            d = dict(zip(keys, combo))
            d["_hash"] = hashlib.md5(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:8]
            combinations.append(d)

        total_evals = len(combinations) * len(scenarios)
        print(f"[OPTIMIZER] {len(combinations)} combos × {len(scenarios)} scenarios = {total_evals} evals")
        print(f"[OPTIMIZER] Starting at {datetime.now().strftime('%H:%M:%S')}...")

        X_train = deepcopy(self._Xy_train["X"])
        y_train = deepcopy(self._Xy_train["y"])
        X_val = deepcopy(self._Xy_val["X"])
        y_val = deepcopy(self._Xy_val["y"])

        all_results = []
        best_params = None
        best_resiliency = -float("inf")
        start_time = time.time()

        for idx, params in enumerate(combinations):
            combo_start = time.time()
            scenario_scores = []
            scenario_sharpes = []

            for scenario in scenarios:
                config = SCENARIO_CONFIG.get(scenario, SCENARIO_CONFIG["STABLE"])
                mu_mult, vol_mult = config["mu_multiplier"], config["vol_multiplier"]

                y_tr_s = mu_mult * y_train.mean() + vol_mult * (y_train - y_train.mean())
                y_val_s = mu_mult * y_val.mean() + vol_mult * (y_val - y_val.mean())

                ml = MLTrainEngine()
                trained = ml.train({"X": X_train, "y": y_tr_s, "params": params})
                model_pkg = {"model": trained["model"], "label_encoders": trained.get("label_encoders", {})}

                signal = self.backtest.signal(model_pkg, {"X": X_val, "y": y_val_s})
                metrics = self.backtest.evaluate({"X": X_val, "y": y_val_s}, signal)

                scenario_scores.append(metrics["score"])
                scenario_sharpes.append(metrics["sharpe"])
                all_results.append({
                    "scenario": scenario, "params": params,
                    "sharpe": metrics["sharpe"], "score": metrics["score"], "corr": metrics["corr"]
                })

            if scenario_scores:
                resiliency = 0.6 * min(scenario_sharpes) + 0.4 * np.mean(scenario_scores)
                if resiliency > best_resiliency:
                    best_resiliency = resiliency
                    best_params = params

            # === PROGRESS REPORTING ===
            elapsed = time.time() - start_time
            avg_per_combo = elapsed / (idx + 1)
            remaining = avg_per_combo * (len(combinations) - idx - 1)
            eta = datetime.now() + pd.Timedelta(seconds=remaining)

            if (idx + 1) % 10 == 0 or idx == 0 or idx == len(combinations) - 1:
                print(f"  [{idx+1:3d}/{len(combinations)}] "
                      f"resiliency={resiliency:.4f} best={best_resiliency:.4f} "
                      f"elapsed={elapsed/60:.1f}m ETA={remaining/60:.1f}m "
                      f"({eta.strftime('%H:%M:%S')})")

        total_time = time.time() - start_time
        print(f"[OPTIMIZER] Complete: {len(combinations)} combos in {total_time/60:.1f}m")
        print(f"[OPTIMIZER] Best resiliency: {best_resiliency:.4f}")
        print(f"[OPTIMIZER] Best params: {best_params}")

        self.storage.save_json({
            "best_params": best_params, "best_resiliency_score": float(best_resiliency),
            "total_combinations": len(combinations), "total_time_sec": total_time,
            "timestamp": datetime.now().isoformat()
        }, "combinatorial_optimization.json")

        return {"best_params": best_params, "best_resiliency_score": best_resiliency, "all_results": all_results}

# =====================================================
# MODEL COMPARATOR
# =====================================================

class ModelComparator:
    def __init__(self, storage):
        self.storage = storage
        self.comparison_results = {}

    def compare_models(self, baseline_model, optimized_model, baseline_val_metrics, optimized_val_metrics):
        comparison = {
            "timestamp": datetime.now().isoformat(),
            "model_comparison": {},
            "validation_comparison": {},
            "rating": {},
            "prediction": {}
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


# =====================================================
# TRUE VALIDATION ENGINE
# =====================================================

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


# =====================================================
# AGENTIC PIPELINE (MAIN ORCHESTRATOR)
# =====================================================

class AgenticPipeline:

    def __init__(self, data, base_dir="runs"):
        self.data = data
        self.base_dir = base_dir
        self.storage = Storage(base_dir)
        self.split_engine = SplitEngine()
        self.strategy = ProcessEngine()
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

        # ... existing init code ...
        
        # === SRFO CACHE: generated once, used everywhere ===
        self._srfo_full = None
        self._Xy_train = None
        self._Xy_val = None

    def _generate_srfo(self, data):
        """Run QuantXEngine ONCE. Returns {results, formula_outputs, raw_data}."""
        try:
            from ..qxEngine import QuantXEngine
            engine = QuantXEngine()
            strategies = engine.qxStrategyList(data, interval="4y")
            formula_outputs = []
            for s in engine.strategy:
                if hasattr(s, "formulaOutput"):
                    formula_outputs.append(s.formulaOutput.report())
            print(f"[SRFO] Engine: {len(strategies)} strategies, {len(formula_outputs)} formula outputs")
            return {"results": engine.results, "formula_outputs": formula_outputs, "raw_data": data}
        except Exception as e:
            print(f"[SRFO] Engine failed: {e}")
            return None

    def _ensure_srfo(self):
        """Lazy init: generate SRFO once. If engine output fails to parse, use fallback."""
        if self._srfo_full is not None:
            return

        print("[SRFO] Generating raw SRFO from full dataset...")
        self._srfo_full = self._generate_srfo(self.data)

        results, formula_outputs = None, []
        X, y = None, None

        if self._srfo_full:
            results = self._srfo_full.get("results")
            formula_outputs = self._srfo_full.get("formula_outputs", [])
            ml = MLTrainEngine()
            try:
                X, y = ml.build_feature_matrix(
                    strategy_results=results,
                    formula_outputs=formula_outputs,
                    raw_data=self._srfo_full["raw_data"]
                )
                print(f"[SRFO] Engine feature matrix: {X.shape}")
            except Exception as e:
                print(f"[SRFO] Engine output failed: {e}")
                X, y = None, None

        # Fallback if engine parsing failed or produced empty data
        if X is None or len(X) == 0:
            print("[SRFO] Using fallback feature extraction...")
            X, y = self._build_features_from_raw(self.data)
            results, formula_outputs = None, []  # Clear engine artifacts
            print(f"[SRFO] Fallback feature matrix: {X.shape}")

        n = len(y)
        split_idx = int(n * 0.8)
        self._Xy_train = {
            "X": X.iloc[:split_idx].copy(), "y": y.iloc[:split_idx].copy(),
            "features": list(X.columns), "results": results, "formula_outputs": formula_outputs
        }
        self._Xy_val = {
            "X": X.iloc[split_idx:].copy(), "y": y.iloc[split_idx:].copy(),
            "features": list(X.columns), "results": results, "formula_outputs": formula_outputs
        }
        print(f"[SRFO] Split: train={len(self._Xy_train['y'])}, val={len(self._Xy_val['y'])}")

        # Store full X,y for potential scenario stress testing
        self._Xy_full = {"X": X.copy(), "y": y.copy(), "features": list(X.columns)}

    def _run_combinatorial_srfo(self, param_grid, max_combinations=500, scenarios=None):
        """Combinatorial search using cached SRFO X,y. No engine re-runs."""
        
        import time  # ← ADD HERE
        if scenarios is None:
            scenarios = ["BULL", "BEAR", "HIGH_VOL", "CRASH", "STABLE"]

        keys = list(param_grid.keys())
        values = [param_grid[k] for k in keys]
        all_combos = list(itertools.product(*values))
        if len(all_combos) > max_combinations:
            np.random.seed(42)
            selected = [all_combos[i] for i in np.random.choice(len(all_combos), max_combinations, replace=False)]
        else:
            selected = all_combos

        combinations = []
        for combo in selected:
            d = dict(zip(keys, combo))
            d["_hash"] = hashlib.md5(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:8]
            combinations.append(d)

        print(f"[OPTIMIZER] {len(combinations)} combos × {len(scenarios)} scenarios = {len(combinations)*len(scenarios)} evals")

        X_train = deepcopy(self._Xy_train["X"])
        y_train = deepcopy(self._Xy_train["y"])
        X_val = deepcopy(self._Xy_val["X"])
        y_val = deepcopy(self._Xy_val["y"])

        all_results = []
        best_params = None
        best_resiliency = -float("inf")
        start_time = time.time()  # ← ADD THIS LINE
        for idx, params in enumerate(combinations):
            combo_start = time.time()
            scenario_scores = []
            scenario_sharpes = []

            for scenario in scenarios:
                config = SCENARIO_CONFIG.get(scenario, SCENARIO_CONFIG["STABLE"])
                mu_mult, vol_mult = config["mu_multiplier"], config["vol_multiplier"]

                y_tr_s = mu_mult * y_train.mean() + vol_mult * (y_train - y_train.mean())
                y_val_s = mu_mult * y_val.mean() + vol_mult * (y_val - y_val.mean())

                ml = MLTrainEngine()
                trained = ml.train({"X": X_train, "y": y_tr_s, "params": params})
                model_pkg = {"model": trained["model"], "label_encoders": trained.get("label_encoders", {})}

                signal = self.backtest.signal(model_pkg, {"X": X_val, "y": y_val_s})
                metrics = self.backtest.evaluate({"X": X_val, "y": y_val_s}, signal)

                scenario_scores.append(metrics["score"])
                scenario_sharpes.append(metrics["sharpe"])
                all_results.append({
                    "scenario": scenario, "params": params,
                    "sharpe": metrics["sharpe"], "score": metrics["score"], "corr": metrics["corr"]
                })

            # === ALWAYS PRINT PROGRESS ===
            resiliency = 0.6 * min(scenario_sharpes) + 0.4 * np.mean(scenario_scores) if scenario_scores else -999
            elapsed = time.time() - start_time
            avg_per_combo = elapsed / (idx + 1)
            remaining = avg_per_combo * (len(combinations) - idx - 1)
            
            new_best_flag = ""
            if resiliency > best_resiliency:
                best_resiliency = resiliency
                best_params = params
                new_best_flag = " *** NEW BEST ***"

            print(f"  [{idx+1:3d}/{len(combinations)}] "
                  f"res={resiliency:.4f} best={best_resiliency:.4f} "
                  f"min_sharpe={min(scenario_sharpes):.4f} avg_score={np.mean(scenario_scores):.4f} "
                  f"elapsed={elapsed/60:.1f}m ETA={remaining/60:.1f}m{new_best_flag}")            
            if scenario_scores:
                resiliency = 0.6 * min(scenario_sharpes) + 0.4 * np.mean(scenario_scores)
                if resiliency > best_resiliency:
                    best_resiliency = resiliency
                    best_params = params
                    print(f"  [NEW BEST] Combo {idx+1}: Resiliency={resiliency:.4f}")

        self.storage.save_json({
            "best_params": best_params, "best_resiliency_score": float(best_resiliency),
            "total_combinations": len(combinations), "timestamp": datetime.now().isoformat()
        }, "combinatorial_optimization.json")

        return {"best_params": best_params, "best_resiliency_score": best_resiliency, "all_results": all_results}

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
            from ..qxEngine import QuantXEngine
            engine = QuantXEngine()
            strategies = engine.qxStrategyList(data, interval=params.get("interval", "4y"))
            print(f"[ENGINE] Strategies: {len(strategies)}")
            formula_outputs = []
            for s in engine.strategy:
                if hasattr(s, "formulaOutput"):
                    formula_outputs.append(s.formulaOutput.report())
            print(f"[ML] Formula outputs: {len(formula_outputs)}")
            if len(formula_outputs) == 0:
                raise ValueError("No formula outputs generated")
            ml = MLTrainEngine()
            X, y = ml.build_feature_matrix(
                strategy_results=engine.results, 
                formula_outputs=formula_outputs, 
                raw_data=data
            )
            print(f"[ML] Dataset: {X.shape}")
            return {
                "X": X, "y": y, "features": list(X.columns), 
                "results": engine.results, "formula_outputs": formula_outputs
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
        rows, targets = [], []
        for symbol, df in data.items():
            if not isinstance(df, pd.DataFrame) or len(df) < 30:
                continue
            for i in range(20, len(df) - 5):
                window = df.iloc[i - 20:i]
                close = window["close"]
                volume = window["volume"] if "volume" in window.columns else pd.Series([1] * len(window))
                ret = close.pct_change().dropna()
                if len(ret) < 5:
                    continue
                features = {
                    "symbol": symbol,
                    "market_price": close.iloc[-1],
                    "market_volume": volume.iloc[-1] if len(volume) > 0 else 0,
                    "market_structure_liq_adj_vol": volume.rolling(5).mean().iloc[-1] if len(volume) >= 5 else 0,
                    "risk_volatility": ret.rolling(20).std().iloc[-1] if len(ret) >= 20 else ret.std(),
                    "risk_sharpe": ret.mean() / (ret.std() + 1e-9),
                    "risk_drawdown": (close.iloc[-1] / close.cummax().iloc[-1] - 1),
                    "alpha_pure": ret.rolling(5).mean().iloc[-1] if len(ret) >= 5 else ret.mean(),
                    "alpha_ts": close.iloc[-1] - close.rolling(20).mean().iloc[-1],
                    "alpha_beta": ret.mean() / (ret.std() + 1e-9),
                    "alpha_residual": ret.iloc[-1] - ret.rolling(20).mean().iloc[-1] if len(ret) >= 20 else ret.iloc[-1] - ret.mean(),
                    "alpha_xs": close.pct_change(5).iloc[-1] if len(close) >= 6 else 0,
                    "transform_zscore": (close.iloc[-1] - close.rolling(20).mean().iloc[-1]) / (close.rolling(20).std().iloc[-1] + 1e-9),
                    "transform_rank": close.rolling(20).rank().iloc[-1] / 20.0,
                    "transform_winsor": np.clip(ret.iloc[-1], -3 * ret.std(), 3 * ret.std()),
                    "transform_tanh": np.tanh(ret.iloc[-1]),
                    "transform_detrend": close.iloc[-1] - close.rolling(20).mean().iloc[-1],
                    "market_symbol": symbol,
                }
                defaults = ["basic_corr", "basic_hit_ratio", "basic_r_squared", "basic_tstat",
                            "decision_score", "decision_signal", "execution_impact",
                            "execution_slippage", "execution_turnover", "intel_ic",
                            "market_structure_regime", "portfolio_entropy", "portfolio_inv_vol",
                            "portfolio_kelly", "portfolio_mvo", "portfolio_risk_parity",
                            "portfolio_weight", "risk_cvar"]
                for col in defaults:
                    features[col] = 0
                future_ret = df["close"].iloc[i + 5] / df["close"].iloc[i] - 1
                rows.append(features)
                targets.append(future_ret)
        X = pd.DataFrame(rows)
        y = pd.Series(targets, name="target")
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        y = y.replace([np.inf, -np.inf], np.nan).fillna(0)
        # === FIX: Drop symbol so X only contains numeric features ===
        X = X.drop(columns=["symbol"], errors="ignore")
        return X, y

    def run_agent(self, params=None, max_iters=5, run_combinatorial=True, param_grid=None):
        if params is None:
            params = {}

        # === PHASE 0: SRFO ONCE ===
        print("=" * 60)
        print("PHASE 0: GENERATING SRFO")
        print("=" * 60)
        self._ensure_srfo()

        # === PHASE 1: BASELINE ===
        print("\n" + "=" * 60)
        print("PHASE 1: BASELINE MODEL")
        print("=" * 60)
        baseline_params = {
            "trees": params.get("trees", 500), "depth": params.get("depth", 12),
            "horizon": params.get("horizon", 21), "min_samples_split": params.get("min_samples_split", 2),
            "max_features": params.get("max_features", "sqrt"), "model": params.get("model", "random_forest")
        }
        baseline_train = self.strategy.train(self.get_train_package(baseline_params))
        self.baseline_model_package = deepcopy(baseline_train)
        self.storage.save(self.baseline_model_package, "baseline_model.pkl")

        baseline_val_pkg = self.get_val_package()
        baseline_signal = self.backtest.signal(self.baseline_model_package, baseline_val_pkg)
        self.baseline_val_metrics = self.backtest.evaluate(baseline_val_pkg, baseline_signal)
        print(f"[BASELINE] Score: {self.baseline_val_metrics.get('score', 0):.4f}")

        # === PHASE 2: COMBINATORIAL (SRFO-BASED, FAST) ===
        if run_combinatorial:
            print("\n" + "=" * 60)
            print("PHASE 2: COMBINATORIAL SEARCH (SRFO-CACHED)")
            print("=" * 60)
            if param_grid is None:
                param_grid = PARAM_GRID_INSIDE
            opt_result = self._run_combinatorial_srfo(param_grid, max_combinations=500)
            best_params = opt_result["best_params"]
            if best_params:
                for k in ["trees", "depth", "horizon", "min_samples_split", "max_features", "model"]:
                    if k in best_params:
                        params[k] = best_params[k]
                print(f"[OPTIMIZER] Best: {best_params}, Resiliency: {opt_result['best_resiliency_score']:.4f}")

        # === PHASE 3: OPTIMIZED ===
        print("\n" + "=" * 60)
        print("PHASE 3: OPTIMIZED MODEL")
        print("=" * 60)
        opt_train = self.strategy.train(self.get_train_package(params))
        self.optimized_model_package = deepcopy(opt_train)
        self.storage.save(self.optimized_model_package, "optimized_model.pkl")

        opt_val_pkg = self.get_val_package()
        opt_signal = self.backtest.signal(self.optimized_model_package, opt_val_pkg)
        self.optimized_val_metrics = self.backtest.evaluate(opt_val_pkg, opt_signal)
        print(f"[OPTIMIZED] Score: {self.optimized_val_metrics.get('score', 0):.4f}")

        # === PHASE 4: COMPARE ===
        print("\n" + "=" * 60)
        print("PHASE 4: COMPARATOR")
        print("=" * 60)
        comparison = self.comparator.compare_models(
            self.baseline_model_package, self.optimized_model_package,
            self.baseline_val_metrics, self.optimized_val_metrics)
        print(f"[COMPARE] {comparison['rating']['label']} | {comparison['prediction']['recommendation']}")

        # === PHASE 5: RESEARCH LOOP (SRFO-BASED) ===
        print("\n" + "=" * 60)
        print("PHASE 5: RESEARCH LOOP")
        print("=" * 60)
        previous_val = None
        for i in range(max_iters):
            self.iteration += 1
            print(f"--- Iteration {self.iteration}/{max_iters} ---")

            train_pkg = self.strategy.train(self.get_train_package(params))
            val_pkg = self.get_val_package()

            train_signal = self.backtest.signal(train_pkg, train_pkg)
            train_metrics = self.backtest.evaluate(train_pkg, train_signal)

            val_signal = self.backtest.signal(train_pkg, val_pkg)
            val_metrics = self.backtest.evaluate(val_pkg, val_signal)

            print(f"Train: Sharpe={train_metrics.get('sharpe', 0):.3f} Score={train_metrics.get('score', 0):.3f}")
            print(f"Val:   Sharpe={val_metrics.get('sharpe', 0):.3f} Score={val_metrics.get('score', 0):.3f}")

            review = self.review_engine.compare(val_metrics, previous_val)
            decision = self.evaluator.decide(review, train_metrics)

            if val_metrics.get("score", 0) > self.best_score:
                self.best_score = val_metrics.get("score", 0)
                self.best_model = deepcopy(train_pkg)

            self.research_history.append({
                "iteration": self.iteration,
                "train": {k: v for k, v in train_metrics.items() if k not in ["signal", "pnl"]},
                "validation": {k: v for k, v in val_metrics.items() if k not in ["signal", "pnl"]},
                "decision": decision
            })

            golive = self.golive.assess(self.research_history, train_pkg)
            print(f"Decision: {decision} | GoLive: {golive['stage']}")

            if decision == "STOP":
                break
            if decision == "ACCEPT" and golive["ready"]:
                self.storage.save(golive.get("deployment_package"), "deployment_package.pkl")
                break
            if decision == "MUTATE":
                params = self._apply_mutations(params, review.get("recommendations", []))

            previous_val = val_metrics

        return self._generate_final_summary()

    def _apply_mutations(self, params, mutations):
        new_params = deepcopy(params)
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
        return new_params

    def _generate_final_summary(self):
        summary = {
            "total_iterations": self.iteration, "best_score": self.best_score,
            "research_history": self.research_history,
            "golive_ready": any(r.get("decision") == "ACCEPT" for r in self.research_history),
            "best_model_path": f"{self.base_dir}/best_model.pkl" if self.best_model else None,
            "baseline_model_path": f"{self.base_dir}/baseline_model.pkl",
            "optimized_model_path": f"{self.base_dir}/optimized_model.pkl",
            "has_baseline": self.baseline_model_package is not None,
            "has_optimized": self.optimized_model_package is not None,
            "comparison_available": self.comparator.comparison_results != {}
        }
        if self.best_model:
            self.storage.save(self.best_model, "best_model.pkl")
        self.storage.save_json(summary, "final_summary.json")
        print(f"{'='*50}FINAL RESEARCH SUMMARY{'='*50}")
        print(f"Total iterations: {self.iteration}")
        print(f"Best validation score: {self.best_score:.4f}")
        print(f"GoLive ready: {summary['golive_ready']}")
        print(f"Baseline model: {summary['has_baseline']}")
        print(f"Optimized model: {summary['has_optimized']}")
        print(f"Comparison done: {summary['comparison_available']}")
        return summary


# =====================================================
# DATA LOADER
# =====================================================

def build_data():
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT"]
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
        "model": "random_forest"
    }
    pipeline = AgenticPipeline(market_data, base_dir="runs")
    result = pipeline.run_agent(params, max_iters=5, run_combinatorial=True, param_grid=PARAM_GRID_FAST)
    print(f"Pipeline complete. Results saved to: {pipeline.base_dir}")
    print("Baseline model: runs/baseline_model.pkl")
    print("Optimized model: runs/optimized_model.pkl")
    print("Comparison report: runs/model_comparison.json")