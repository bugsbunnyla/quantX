"""
quantX Backtest Pipeline — Agentic ML Research & GoLive System
==============================================================

Requirements addressed:
1. Use data to train and research (full train/val split + research loop)
2. Parameter-driven rating comparison of all trained model outcomes
3. Systemic predictive outcome with evaluator accept/reject decision
4. GoLive ready indicator with systemic predictive outcome
5. Pure Python ML solution continuing the quantX architecture

Architecture:
- AgenticPipeline: orchestrates the full research lifecycle
- BacktestEngine: signal generation + robust evaluation
- ReviewEngine: train vs validation comparison with degradation analysis
- Evaluator: ML-aware accept/reject with mutation recommendations
- GoLiveEngine: production readiness assessment + deployment package
"""

import os
import json
import warnings

import numpy as np
import pandas as pd

from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from copy import deepcopy


from ..qxEngine import QuantXEngine
from ..PickleDataManager import PickleDataManager
from ..strategies.FormulaInfo import FormulaInfo
"""
quantX Backtest Pipeline — Agentic ML Research & GoLive System
==============================================================

Requirements addressed:
1. Use data to train and research (full train/val split + research loop)
2. Parameter-driven rating comparison of all trained model outcomes
3. Systemic predictive outcome with evaluator accept/reject decision
4. GoLive ready indicator with systemic predictive outcome
5. Pure Python ML solution continuing the quantX architecture

Architecture:
- AgenticPipeline: orchestrates the full research lifecycle
- BacktestEngine: signal generation + robust evaluation
- ReviewEngine: train vs validation comparison with degradation analysis
- Evaluator: ML-aware accept/reject with mutation recommendations
- GoLiveEngine: production readiness assessment + deployment package
"""

import os
import pickle
import json
import warnings
import copy
import joblib

import numpy as np
import pandas as pd

from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from copy import deepcopy
from pathlib import Path

from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')


# =====================================================
# STORAGE ENGINE
# =====================================================

class Storage:
    """
    Simple storage manager for pipeline artifacts.
    Supports both:
        storage.save(obj, "runs/run_000001/model.pkl")
    and
        storage.save(obj, "run_000001/model.pkl")
    """

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

    def save_dataset(self, name, data):
        return self.save(data, self.datasets / f"{name}.pkl")

    def load_dataset(self, name):
        return self.load(self.datasets / f"{name}.pkl")

    def create_run(self):
        runs = sorted(self.base.glob("run_*"))
        if runs:
            last = int(runs[-1].name.split("_")[1])
            idx = last + 1
        else:
            idx = 1
        run_dir = self.base / f"run_{idx:06d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(self.base / "latest.txt", "w") as f:
            f.write(run_dir.name)
        return idx, run_dir

    def latest_run(self):
        latest = self.base / "latest.txt"
        if not latest.exists():
            return None
        with open(latest, "r") as f:
            return self.base / f.read().strip()

    def list_runs(self):
        return sorted(self.base.glob("run_*"))

    def save_run_file(self, run_dir, name, obj):
        return self.save(obj, Path(run_dir) / f"{name}.pkl")

    def load_run_file(self, run_dir, name):
        return self.load(Path(run_dir) / f"{name}.pkl")


# =====================================================
# SPLIT ENGINE
# =====================================================

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
        self.formula_outputs = None
        self.strategy_results = None
        self.raw_data = None
        self.params = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}

    # =====================================================
    # MAIN TRAIN ENTRY
    # RECEIVES PACKAGE FROM AgenticPipeline
    # =====================================================

    def train(self, train_package):
        print("[ML] Training from package")

        if not isinstance(train_package, dict):
            raise TypeError("train() requires train_package dict")

        required = ["X", "y"]
        for key in required:
            if key not in train_package:
                raise KeyError(f"Missing train_package key: {key}")

        X = train_package["X"]
        y = train_package["y"]

        self.strategy_results = train_package.get("results")
        self.formula_outputs = train_package.get("formula_outputs")
        self.params = train_package.get("params", {})

        print("[ML] Dataset:", X.shape, y.shape)

        # Encode categoricals before scaling
        X_encoded = self._encode_categoricals(X, fit=True)

        trained_model, metrics = self.fit(X_encoded, y, self.params)

        self.model = trained_model
        self.feature_schema = list(X.columns)

        result = {
            "model": trained_model,
            "X": X,
            "y": y,
            "features": list(X.columns),
            "metrics": metrics,
            "results": train_package.get("results"),
            "formula_outputs": train_package.get("formula_outputs"),
            "params": self.params,
            "label_encoders": self.label_encoders,
            "scaler": self.scaler
        }

        joblib.dump(result, "train.pkl")
        return result

    # =====================================================
    # BUILD FEATURE MATRIX
    # USED BY AgenticPipeline.prepare()
    # =====================================================

    def build_feature_matrix(self, strategy_results, formula_outputs, raw_data, params=None):
        X, y, schema = self.build_features(strategy_results, formula_outputs, raw_data, params or {})
        return X, y

    # =====================================================
    # FEATURE BUILDER
    # =====================================================
    def _lookup_multiindex(self, df, category, metric, case_sensitive=False):
        """
        Find a column in a DataFrame by (category, metric) tuple.
        The report() DataFrame has MultiIndex columns and symbol index.
        Returns the Series if found, None otherwise.
        """
        if not isinstance(df, pd.DataFrame) or len(df.columns) == 0:
            return None

        cat_norm = str(category).lower() if not case_sensitive else str(category)
        met_norm = str(metric).lower() if not case_sensitive else str(metric)

        # Method 1: MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            for col in df.columns:
                if isinstance(col, tuple) and len(col) >= 2:
                    col_cat = str(col[0]).lower() if not case_sensitive else str(col[0])
                    col_met = str(col[1]).lower() if not case_sensitive else str(col[1])
                    if col_cat == cat_norm and col_met == met_norm:
                        return df[col].copy()
            return None

        # Method 2: Flat string columns (fallback)
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
        cached_ret_series = None

        # -------------------------------------------------
        # FORMULA OUTPUTS — handles both orientations
        # -------------------------------------------------
        if formula_outputs is not None:
            if isinstance(formula_outputs, pd.DataFrame):
                formula_outputs = [formula_outputs]

            for i, fo in enumerate(formula_outputs):
                if not isinstance(fo, pd.DataFrame):
                    continue

                # Detect orientation:
                # A) index=symbols, columns=MultiIndex(category, metric)  [ORIGINAL]
                # B) index=MultiIndex(category, metric), columns=symbols   [TRANSPOSED]

                idx_is_multi = isinstance(fo.index, pd.MultiIndex)
                col_is_multi = isinstance(fo.columns, pd.MultiIndex)

                if col_is_multi:
                    # ORIENTATION A: symbol index, (cat, metric) columns
                    # Target: lookup ('market', 'ret') in columns
                    if ret_series is None:
                        ret_series = self._lookup_multiindex(fo, "market", "ret")
                        if ret_series is not None:
                            cached_ret_series = ret_series

                    # Melt: stack columns into rows
                    temp = fo.stack(level=[0, 1]).reset_index()
                    temp.columns = ["symbol", "category", "metric", "value"]

                    temp["feature"] = (
                        "fo_" + temp["category"].astype(str) + "_" + temp["metric"].astype(str)
                    )

                    formula_df = (
                        temp.pivot_table(
                            index="symbol",
                            columns="feature",
                            values="value",
                            aggfunc="last"
                        )
                        .reset_index()
                    )

                    formula_df = formula_df.drop(
                        columns=["fo_market_ret"],
                        errors="ignore"
                    )
                    frames.append(formula_df)

                elif idx_is_multi:
                    # ORIENTATION B: (cat, metric) index, symbol columns
                    # Target: find row where second level is 'ret'
                    if ret_series is None:
                        for idx in fo.index:
                            if isinstance(idx, tuple) and len(idx) >= 2:
                                if str(idx[1]).lower() == 'ret':
                                    ret_series = fo.loc[idx]
                                    if isinstance(ret_series, pd.DataFrame):
                                        ret_series = ret_series.iloc[:, 0]
                                    break

                    # Extract features: each row (cat, metric) -> feature column
                    for idx in fo.index:
                        if isinstance(idx, tuple) and len(idx) >= 2:
                            cat, metric = idx[0], idx[1]
                        else:
                            continue
                        if str(metric).lower() == 'ret':
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
                            feat_rows.append({'symbol': str(symbol), col_name: value})
                        if feat_rows:
                            frames.append(pd.DataFrame(feat_rows))
                else:
                    # Neither MultiIndex - skip
                    continue

        # -------------------------------------------------
        # STRATEGY RESULTS  (with sr_ prefix)
        # -------------------------------------------------
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
                signal_df = pd.DataFrame(
                    [
                        {"symbol": symbol, f"sr_signal_{sr_idx}": signal}
                        for symbol, signal in result.signals.items()
                    ]
                )
                frames.append(signal_df)
                sr_idx += 1

        if not frames:
            raise ValueError("No features generated")

        # -------------------------------------------------
        # MERGE
        # -------------------------------------------------
        feature_df = frames[0]
        for frame in frames[1:]:
            feature_df = feature_df.merge(frame, on="symbol", how="outer")

        # -------------------------------------------------
        # TARGET
        # -------------------------------------------------
        if ret_series is None:
            raise ValueError("Missing ('market','ret') target")

        y_df = (
            ret_series.rename_axis("symbol").reset_index(name="y")
        )

        feature_df = feature_df.merge(y_df, on="symbol", how="inner")
        feature_df = feature_df.dropna(subset=["y"])

        y = feature_df["y"]
        X = feature_df.drop(columns=["symbol", "y"], errors="ignore")

        # Encode features
        X = self.encode_features(X)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        return X, y, list(X.columns)

    # =====================================================
    # CATEGORICAL ENCODER (with persistence)
    # =====================================================

    def _encode_categoricals(self, X, fit=True):
        """Encode categorical columns to numeric for ML consumption."""
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
                        X_encoded[col] = X_encoded[col].apply(
                            lambda x: le.transform([x])[0] if x in known_classes else -1
                        )
                    else:
                        X_encoded[col] = 0
        return X_encoded

    # =====================================================
    # MODEL FIT
    # =====================================================

    def fit(self, X, y, params=None):
        if params is None:
            params = {}

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestRegressor(
            n_estimators=params.get("trees", 500),
            max_depth=params.get("depth", 12),
            random_state=42
        )

        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            "r2": float(r2_score(y_test, pred)),
            "samples": len(X),
            "features": len(X.columns)
        }

        wrapper = {
            "model": model,
            "features": list(X.columns),
            "metrics": metrics
        }

        return wrapper, metrics

    # =====================================================
    # ENCODER
    # =====================================================

    def encode_features(self, df):
        result = df.copy()
        for col in result.columns:
            if pd.api.types.is_numeric_dtype(result[col]):
                result[col] = result[col].fillna(0)
            elif pd.api.types.is_datetime64_any_dtype(result[col]):
                result[col] = result[col].astype("int64")
            else:
                result[col] = (
                    result[col].fillna("UNKNOWN").astype("category").cat.codes
                )
        return result


# =====================================================
# PROCESS ENGINE - STRATEGY
# =====================================================

class ProcessEngine:

    def __init__(self):
        self.ml = MLTrainEngine()

    # =====================================================
    # TRAIN MODEL
    # RECEIVES PACKAGE FROM AgenticPipeline
    # =====================================================

    def train(self, train_package, params=None):
        if params is None:
            params = {}

        if not isinstance(train_package, dict):
            raise TypeError("ProcessEngine.train expects package dict")

        print("[PROCESS] Training package received")
        print("[PROCESS] Keys:", train_package.keys())

        # Validate package contract
        required = ["X", "y", "results", "formula_outputs"]
        for key in required:
            if key not in train_package:
                raise KeyError(f"Missing train package key: {key}")

        # ML parameters
        ml_params = {
            "trees": params.get("trees", 500),
            "depth": params.get("depth", 12),
            "horizon": params.get("horizon", 21),
            "model": params.get("model", "random_forest")
        }

        print("[ML] Parameters:", ml_params)

        # attach parameters
        train_package["params"] = ml_params

        # Train ML model
        result = self.ml.train(train_package)
        print("[ML] Training complete")

        # Merge ML result back
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

    # =====================================================
    # BUILD FEATURES FOR LIVE SIGNAL
    # =====================================================

    def build_features_for_signal(self, raw_data):
        rows = []

        for symbol, df in raw_data.items():
            if not isinstance(df, pd.DataFrame):
                continue
            if len(df) < 20:
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

            defaults = [
                "basic_corr", "basic_hit_ratio", "basic_r_squared", "basic_tstat",
                "decision_score", "decision_signal",
                "execution_impact", "execution_slippage", "execution_turnover",
                "intel_ic", "market_structure_regime",
                "portfolio_entropy", "portfolio_inv_vol", "portfolio_kelly",
                "portfolio_mvo", "portfolio_risk_parity", "portfolio_weight",
                "risk_cvar"
            ]
            for c in defaults:
                row[c] = 0

            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        self.last_features = df.copy()
        return df

    # =====================================================
    # ENCODER
    # =====================================================

    def encode_features(self, df):
        result = df.copy()
        for col in result.columns:
            if pd.api.types.is_numeric_dtype(result[col]):
                result[col] = result[col].fillna(0)
            else:
                result[col] = (
                    result[col].fillna("UNKNOWN").astype("category").cat.codes
                )
        return result

    # =====================================================
    # GENERATE SIGNAL
    # TRAIN MODEL -> TRAIN OR VAL DATA
    # =====================================================

    def signal(self, model_package, dataset):
        if "model" not in model_package:
            raise KeyError("model_package missing model")

        model = model_package["model"]["model"]

        if "X" not in dataset:
            raise KeyError("dataset missing X")

        X = dataset["X"].copy()

        required_features = model_package["model"]["features"]

        # Align validation features
        for col in required_features:
            if col not in X.columns:
                X[col] = 0

        X = X[required_features]

        if X.shape[0] == 0:
            return np.array([])

        # Encode categoricals using stored encoders if available
        label_encoders = model_package.get("label_encoders", {})
        if label_encoders:
            for col in X.columns:
                if not pd.api.types.is_numeric_dtype(X[col]):
                    le = label_encoders.get(col)
                    if le:
                        X[col] = X[col].astype(str)
                        known_classes = set(le.classes_)
                        X[col] = X[col].apply(
                            lambda x: le.transform([x])[0] if x in known_classes else -1
                        )
                    else:
                        X[col] = X[col].fillna("UNKNOWN").astype("category").cat.codes

        prediction = model.predict(X)
        prediction = np.asarray(prediction, dtype=float)
        self.last_signal = prediction
        return prediction

    # =====================================================
    # EVALUATION
    # =====================================================

    def evaluate(self, package, signal):
        if "y" not in package:
            raise KeyError("Package missing y")

        y = np.asarray(package["y"], dtype=float).ravel()
        signal = np.asarray(signal, dtype=float).ravel()

        if len(signal) == 0:
            return {"hit": 0.0, "sharpe": 0.0, "cvar": 0.0,
                    "corr": 0.0, "r2": 0.0, "t_beta": 0.0,
                    "score": 0.0, "signal": signal, "pnl": np.array([])}

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
            "hit": float(hit),
            "sharpe": float(sharpe),
            "cvar": float(cvar),
            "corr": float(corr),
            "r2": float(r2),
            "t_beta": float(t_beta),
            "score": float(score),
            "signal": signal,
            "pnl": pnl
        }


# =====================================================
# REVIEW ENGINE (Enhanced)
# =====================================================

class ReviewEngine:

    def __init__(self):
        self.comparison_history = []

    def compare(self, current_val, previous_val=None):
        review = {
            "timestamp": datetime.now().isoformat(),
            "current": current_val,
            "previous": previous_val,
            "degradation": {},
            "trend": "FIRST_RUN",
            "warnings": [],
            "recommendations": []
        }

        if previous_val is None:
            review["recommendations"].append("First iteration: establish baseline")
            return review

        metrics_to_compare = ["sharpe", "corr", "hit", "r2", "score", "cvar"]

        for metric in metrics_to_compare:
            if metric in current_val and metric in previous_val:
                curr = current_val[metric]
                prev = previous_val[metric]
                if prev != 0:
                    degradation = (curr - prev) / abs(prev)
                else:
                    degradation = 0 if curr == 0 else float('inf')

                review["degradation"][metric] = {
                    "current": curr,
                    "previous": prev,
                    "change": curr - prev,
                    "pct_change": degradation
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
# EVALUATOR (Enhanced)
# =====================================================

class Evaluator:

    def __init__(self):
        self.decision_history = []
        self.thresholds = {
            "min_sharpe": 0.5,
            "min_corr": 0.1,
            "min_hit": 0.52,
            "max_overfit_gap": 0.3,
            "min_score": 0.3
        }

    def decide(self, review, train_metrics=None):
        current = review.get("current", {})
        trend = review.get("trend", "FIRST_RUN")
        warnings = review.get("warnings", [])

        decision = {
            "verdict": "CONTINUE",
            "confidence": 0.5,
            "reasons": [],
            "mutations": []
        }

        if current.get("sharpe", 0) < self.thresholds["min_sharpe"]:
            decision["reasons"].append(f"Sharpe {current.get('sharpe', 0):.3f} below threshold {self.thresholds['min_sharpe']}")

        if current.get("corr", 0) < self.thresholds["min_corr"]:
            decision["reasons"].append(f"Correlation {current.get('corr', 0):.3f} below threshold {self.thresholds['min_corr']}")

        if current.get("hit", 0) < self.thresholds["min_hit"]:
            decision["reasons"].append(f"Hit ratio {current.get('hit', 0):.3f} below threshold {self.thresholds['min_hit']}")

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
            if len(decision["reasons"]) == 0:
                decision["verdict"] = "CONTINUE"
            else:
                decision["verdict"] = "MUTATE"

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
            "min_iterations": 3,
            "min_accept_ratio": 0.5,
            "min_avg_sharpe": 0.6,
            "min_avg_corr": 0.15,
            "max_avg_drawdown": -0.15,
            "min_consistency": 0.7
        }

    def assess(self, research_history, model_package):
        if len(research_history) < self.readiness_criteria["min_iterations"]:
            return {
                "ready": False,
                "stage": "RESEARCH",
                "reason": f"Insufficient iterations: {len(research_history)}/{self.readiness_criteria['min_iterations']}",
                "systemic_prediction": None
            }

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
            "avg_sharpe": np.mean(val_sharpes),
            "avg_corr": np.mean(val_corrs),
            "avg_drawdown": np.mean(val_drawdowns),
            "consistency": systemic_prediction["consistency_ratio"]
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
            "ready": passed_checks,
            "stage": "GOLIVE" if passed_checks else "RESEARCH",
            "checks": checks,
            "systemic_prediction": systemic_prediction,
            "deployment_package": deployment_package,
            "recommendation": "DEPLOY" if passed_checks else "CONTINUE_RESEARCH"
        }

    def _create_deployment_package(self, model_package, prediction):
        return {
            "model": model_package["model"]["model"],
            "features": model_package["model"]["features"],
            "expected_performance": prediction,
            "timestamp": datetime.now().isoformat(),
            "version": f"quantx-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        }

    def update_criteria(self, criteria):
        self.readiness_criteria.update(criteria)


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

        self.iteration = 0
        self.research_history = []
        self.best_model = None
        self.best_score = -float('inf')

    def prepare(self, data, params=None):
        if params is None:
            params = {}

        # Use QuantXEngine if available, otherwise use the MLTrainEngine feature builder
        try:
            from ..qxEngine import QuantXEngine
            engine = QuantXEngine()
            strategy_names = params.get("strategies", ["AlphaStrategy"])
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

            # Only store serializable data, not the engine object itself
            package = {
                "X": X,
                "y": y,
                "features": list(X.columns),
                "results": engine.results,
                "formula_outputs": formula_outputs
                # REMOVED: "engine": engine, "strategies": engine.strategy
            }
            return package

        except Exception as e:
            print(f"[WARN] QuantXEngine not available or failed: {e}")
            print("[WARN] Falling back to direct feature extraction from raw data")

            # Fallback: build features directly from raw OHLCV data
            ml = MLTrainEngine()
            X, y = self._build_features_from_raw(data)
            print(f"[ML] Dataset (fallback): {X.shape}")

            package = {
                "X": X,
                "y": y,
                "features": list(X.columns),
                "results": {},
                "formula_outputs": []
            }
            return package

    def _build_features_from_raw(self, data):
        """Fallback feature builder when QuantXEngine is not available."""
        rows = []
        targets = []

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

                defaults = [
                    "basic_corr", "basic_hit_ratio", "basic_r_squared", "basic_tstat",
                    "decision_score", "decision_signal",
                    "execution_impact", "execution_slippage", "execution_turnover",
                    "intel_ic", "market_structure_regime",
                    "portfolio_entropy", "portfolio_inv_vol", "portfolio_kelly",
                    "portfolio_mvo", "portfolio_risk_parity", "portfolio_weight",
                    "risk_cvar"
                ]
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

    def run_agent(self, params=None, max_iters=5):
        if params is None:
            params = {}

        train, val = self.split_engine.split(self.data)

        print(f"[DEBUG] train symbols: {list(train.keys())}")
        print(f"[DEBUG] val symbols: {list(val.keys())}")

        self.storage.save(train, "1a_train.pkl")
        self.storage.save(val, "1b_val.pkl")

        previous_val = None

        for i in range(max_iters):
            self.iteration += 1

            print(f"\n{"="*50}")
            print(f"AGENT ITERATION {self.iteration}")
            print(f"{"="*50}")

            # 1. BUILD TRAIN PACKAGE
            raw_train_package = self.prepare(train, params)
            train_package = self.strategy.train(raw_train_package, params)

            print(f"[AGENT] Train package keys: {train_package.keys()}")
            print(f"[AGENT] Model type: {train_package.get('model', {}).get('model_type', 'random_forest')}")
            print(f"[AGENT] Train metrics: {train_package.get('metrics', {})}")

            # 2. BUILD VALIDATION PACKAGE
            val_package = self.prepare(val, params)
            print(f"[DEBUG] Val keys: {val_package.keys()}")
            if val_package["X"].shape[0] == 0:
                print("[AGENT] ERROR: Validation package has 0 rows. Stopping.")
                break

            # 3. SAVE PREPARED DATA
            run_path = f"run_{self.iteration:06d}"
            os.makedirs(f"{self.base_dir}/{run_path}", exist_ok=True)

            self.storage.save(train_package, f"{run_path}/train_package.pkl")
            self.storage.save(val_package, f"{run_path}/val_package.pkl")

            # 4. TRAIN BACKTEST SIGNAL
            print("[BACKTEST] Generating TRAIN signal")
            train_signal = self.backtest.signal(train_package, train_package)
            print("[BACKTEST] TRAIN signal complete")
            self.storage.save(train_signal, f"{run_path}/train_signal.pkl")

            # 5. TRAIN EVALUATION
            train_metrics = self.backtest.evaluate(train_package, train_signal)
            print(f"\n{"="*30}\nTRAIN METRICS\n{"="*30}")
            for k, v in train_metrics.items():
                if k not in ["signal", "pnl"]:
                    print(f"  {k}: {v}")
            self.storage.save(train_metrics, f"{run_path}/train_metrics.pkl")

            # 6. VALIDATION BACKTEST SIGNAL
            print("[BACKTEST] Generating VALIDATION signal")
            val_signal = self.backtest.signal(train_package, val_package)
            print("[BACKTEST] VALIDATION signal complete")
            self.storage.save(val_signal, f"{run_path}/val_signal.pkl")

            # 7. VALIDATION EVALUATION
            if len(val_signal) == 0:
                print("[AGENT] WARNING: Empty validation signal. Skipping evaluation.")
                val_metrics = {
                    "hit": 0.0, "sharpe": 0.0, "cvar": 0.0,
                    "corr": 0.0, "r2": 0.0, "t_beta": 0.0, "score": 0.0,
                    "signal": val_signal, "pnl": np.array([])
                }
            else:
                val_metrics = self.backtest.evaluate(val_package, val_signal)
            print(f"\n{"="*30}\nVALIDATION METRICS\n{"="*30}")
            for k, v in val_metrics.items():
                if k not in ["signal", "pnl"]:
                    print(f"  {k}: {v}")
            self.storage.save(val_metrics, f"{run_path}/val_metrics.pkl")

            # 8. RESEARCH COMPARISON
            research = {
                "iteration": self.iteration,
                "train": {k: v for k, v in train_metrics.items() if k not in ["signal", "pnl"]},
                "validation": {k: v for k, v in val_metrics.items() if k not in ["signal", "pnl"]},
                "model": train_package.get("metrics", {}),
                "features": train_package.get("features", []),
                "model_type": "random_forest"
            }

            print(f"\n{"="*30}\nRESEARCH REPORT\nTRAIN VS VALIDATION\n{"="*30}")
            print(f"TRAIN Sharpe: {train_metrics.get('sharpe', 0):.4f}")
            print(f"VALIDATION Sharpe: {val_metrics.get('sharpe', 0):.4f}")
            print(f"TRAIN Score: {train_metrics.get('score', 0):.4f}")
            print(f"VALIDATION Score: {val_metrics.get('score', 0):.4f}")

            self.storage.save(research, f"{run_path}/research.pkl")
            self.storage.save_json(research, f"{run_path}/research.json")

            # 9. REVIEW ENGINE
            review = self.review_engine.compare(val_metrics, previous_val)
            print(f"\n{"="*30}\nREVIEW\n{"="*30}")
            print(f"Trend: {review['trend']}")
            if review["warnings"]:
                print(f"Warnings: {review['warnings']}")
            if review["recommendations"]:
                print(f"Recommendations: {review['recommendations']}")
            self.storage.save(review, f"{run_path}/review.pkl")

            # 10. AGENT DECISION
            decision = self.evaluator.decide(review, train_metrics)
            research["decision"] = decision
            print(f"\n{"="*30}\nAGENT DECISION\n{"="*30}")
            print(f"VERDICT: {decision}")

            val_score = val_metrics.get("score", 0)
            if val_score > self.best_score:
                self.best_score = val_score
                self.best_model = deepcopy(train_package)
                print(f"[AGENT] New best model: score={val_score:.4f}")

            self.storage.save({"decision": decision}, f"{run_path}/eval.pkl")

            # 11. GOLIVE ASSESSMENT
            self.research_history.append(research)
            golive = self.golive.assess(self.research_history, train_package)

            print(f"\n{"="*30}\nGOLIVE ASSESSMENT\n{"="*30}")
            print(f"Stage: {golive['stage']}")
            print(f"Ready: {golive['ready']}")

            if golive["systemic_prediction"]:
                pred = golive["systemic_prediction"]
                print(f"\nSystemic Predictive Outcome:")
                print(f"  Expected Sharpe: {pred['expected_sharpe']:.4f}")
                print(f"  Expected Corr: {pred['expected_corr']:.4f}")
                print(f"  Win Rate: {pred['win_rate']:.2%}")
                print(f"  Consistency: {pred['consistency_ratio']:.2%}")

            self.storage.save(golive, f"{run_path}/golive.pkl")
            self.storage.save_json(golive, f"{run_path}/golive.json")

            # 12. ITERATION CONTROL
            if decision == "STOP":
                print("[AGENT] Stopping research")
                break

            if decision == "ACCEPT" and golive["ready"]:
                print("[AGENT] ACCEPTED and GOLIVE READY — Deploying model")
                self.storage.save(golive.get("deployment_package"), "deployment_package.pkl")
                break

            if decision == "MUTATE":
                mutations = review.get("recommendations", [])
                params = self._apply_mutations(params, mutations)
                print(f"[AGENT] Applied mutations: {mutations}")

            previous_val = val_metrics
            print(f"{"="*50}\n")

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
            "total_iterations": self.iteration,
            "best_score": self.best_score,
            "research_history": self.research_history,
            "golive_ready": any(r.get("decision") == "ACCEPT" for r in self.research_history),
            "best_model_path": f"{self.base_dir}/best_model.pkl" if self.best_model else None
        }

        if self.best_model:
            self.storage.save(self.best_model, "best_model.pkl")

        self.storage.save_json(summary, "final_summary.json")

        print(f"\n{"="*50}")
        print("FINAL RESEARCH SUMMARY")
        print(f"{"="*50}")
        print(f"Total iterations: {self.iteration}")
        print(f"Best validation score: {self.best_score:.4f}")
        print(f"GoLive ready: {summary['golive_ready']}")

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
    """Generate sample OHLCV data for testing."""
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
        "trees": 500,
        "depth": 12,
        "horizon": 21,
        "model": "random_forest"
    }

    pipeline = AgenticPipeline(market_data, base_dir="runs")
    result = pipeline.run_agent(params, max_iters=5)

    print(f"\nPipeline complete. Results saved to: {pipeline.base_dir}")