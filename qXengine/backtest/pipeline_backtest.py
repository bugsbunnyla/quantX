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
from ..strategies.FormulaInfo import FormulaInfo

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

def formula_report( report,  mode="df_multi",  lookup="index",):
    """
    Normalize formulaOutput.report() output.

    Parameters
    ----------
    report : tuple
        Expected:
            (df, df_easy)

    mode : str
        df_multi -> use MultiIndex dataframe
        df_easy  -> use flat index dataframe

    lookup : str
        index   -> metrics are rows
        columns -> metrics are columns

    Returns
    -------
    pd.DataFrame
    """
    import pandas as pd
    # -------------------------------------------------
    # Unpack report tuple
    # -------------------------------------------------
    if (  not isinstance(report, tuple) or len(report) != 2    ):
        raise ValueError( "report must be tuple(df, df_easy)" )
    df, df_easy = report

    # -------------------------------------------------
    # Select format
    # -------------------------------------------------
    if mode == "df_multi":
        result = df
    elif mode == "df_easy":
        result = df_easy
    else:
        raise ValueError( "mode must be 'df_multi' or 'df_easy'" )
    if not isinstance(result, pd.DataFrame):
        raise TypeError( "Selected report is not a DataFrame"  )

    # -------------------------------------------------
    # Select orientation
    # -------------------------------------------------
    if lookup == "index":
        return result
    elif lookup == "columns":
        return result.T
    elif lookup == "symbol_features":
        # normalize for ML:
        # rows    -> symbols
        # columns -> features

        if isinstance(result.columns, pd.MultiIndex):
            # df_multi format
            result = result.copy()
            result.columns = [
                f"{a}_{b}" for a, b in result.columns
            ]
            return result

        else:
            # df_easy format:
            # rows are metrics, columns are symbols
            return result.T

    else:
        raise ValueError( "lookup must be 'index' or 'columns'  or 'symbol_features'"  )

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
    def split(self, data, split_ratio=None):   # ignore ratio for interleaved
        train, val = {}, {}
        for symbol, df in data.items():
            if not isinstance(df, pd.DataFrame):
                continue
            mask_even = np.arange(len(df)) % 2 == 0
            train[symbol] = df.iloc[mask_even].copy()
            val[symbol]   = df.iloc[~mask_even].copy()
        return train, val

"""
class SplitEngine:
    def split(self, data, split_ratio=0.5):
        train, val = {}, {}
        for symbol, df in data.items():
            if not isinstance(df, pd.DataFrame):
                continue
            split_idx = len(df) // 2
            train[symbol] = df.iloc[:split_idx].copy()
            val[symbol]   = df.iloc[split_idx:].copy()
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
    
     X = train_package["X"].copy()
     y = train_package["y"].copy()

     print("[DEBUG] X type:", type(X))
     print("[DEBUG] y type:", type(y))

     print("[DEBUG] X shape:", X.shape)
     print("[DEBUG] y shape:", y.shape)

     self.params = train_package.get("params", {})


    # =====================================================
    # Force y into single target vector
    # =====================================================

     if isinstance(y, pd.DataFrame):
          print( "[ML WARNING] y received as DataFrame:", list(y.columns) )
          if "y" in y.columns:
            y = y["y"]
          else:
            # fallback: first target column
            y = y.iloc[:, 0]
     y = ( pd.Series(y).astype(float).replace([ np.inf,-np.inf ], np.nan ) )

    # =====================================================
    # Remove invalid targets
    # =====================================================
     valid_mask = y.notna()
     print("X.index =", repr(X.index))
     print("y.index =", repr(y.index))
     print("mask.index =", repr(valid_mask.index))

     print("X == y ?", X.index.equals(y.index))
     print("X == mask ?", X.index.equals(valid_mask.index))

     print("len(X):", len(X))
     print("len(y):", len(y))
     X = X.loc[valid_mask]
     y = y.loc[valid_mask]
     print(type(valid_mask))
     print(valid_mask.shape)

     print("X index:", X.index)
     print("mask index:", valid_mask.index)

     print(valid_mask.head())
     if len(y) == 0:
        raise ValueError("No valid training targets after NaN removal"    )
    # =====================================================
    # Clean features
    # =====================================================
     X = ( X.replace( [ np.inf, -np.inf  ], np.nan).fillna(0)  )
     print( f"[ML] Dataset AFTER CLEAN: X={X.shape}, y={y.shape}" )
    # =====================================================
    # Encode categoricals
    # =====================================================
     X_encoded = self._encode_categoricals( X,  fit=True  )
    # =====================================================
    # Train
    # =====================================================
     trained_model, metrics = self.fit(
        X_encoded,
        y,
        self.params
       )
     self.model = trained_model
     self.feature_schema = list(X.columns   )
     return {
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

    def extract_metric(df, category, metric):
      cols = [
        c for c in df.columns
        if len(c) >= 3
        and c[1] == category
        and c[2] == metric
      ]

      if not cols:
        return None

      out = df[cols].copy()

      # keep only symbol names
      out.columns = [c[0] for c in cols]
      return out

    def build_feature_matrix(self, strategy_results, formula_outputs, raw_data, params=None):
        X, y, feature_names = self.build_features(strategy_results, formula_outputs, raw_data, params or {})
        if X.shape[0] == 0:
            raise ValueError(f"build_features returned empty feature matrix: X.shape={X.shape}")
        return X, y

    def _lookup_multiindex(self,df, df_easy, category, metric=None, mode="df_multi", lookup="index", case_sensitive=False,):
      """
      Generic report lookup.
      Parameters
      ----------
      df : pd.DataFrame
        MultiIndex report dataframe.
      df_easy : pd.DataFrame
        Flat index dataframe.
      category : str
        Category name or full key depending on mode.
      metric : str, optional
        Metric name for MultiIndex lookup.
      mode : str
        "df_multi" -> use df
        "df_easy"  -> use df_easy
      lookup : str
        "index"   -> lookup rows
        "columns" -> lookup columns
      case_sensitive : bool
      Returns
      -------
      pd.Series or None
      """

      import pandas as pd

      # ----------------------------------------------------
      # Select dataframe
      # ----------------------------------------------------
      if mode == "df_multi":
        report = df
      elif mode == "df_easy":
        report = df_easy
      else:
        raise ValueError(
            "mode must be 'df_multi' or 'df_easy'"
        )
      if not isinstance(report, pd.DataFrame) or report.empty:
        return None

      # Normalize
      cat = str(category)
      met = None if metric is None else str(metric)
      if not case_sensitive:
        cat = cat.lower()
        if met is not None:
            met = met.lower()

      # ====================================================
      # INDEX LOOKUP
      # ====================================================
      if lookup == "index":

        # -----------------------------------------------
        # df_multi:
        #
        # df.index:
        #   ('market','price')
        # -----------------------------------------------
        if mode == "df_multi":
            if isinstance(report.index, pd.MultiIndex):
                for idx in report.index:
                    if len(idx) < 2:
                        continue
                    idx_cat = str(idx[0])
                    idx_met = str(idx[1])
                    if not case_sensitive:
                        idx_cat = idx_cat.lower()
                        idx_met = idx_met.lower()
                    if idx_cat == cat and idx_met == met:
                        return report.loc[idx].copy()
                return None

        # -----------------------------------------------
        # df_easy:
        #
        # df_easy.index:
        #   market_price
        # -----------------------------------------------
        elif mode == "df_easy":
            target = cat
            if met is not None:
                target = f"{cat}_{met}"
            for idx in report.index:
                idx_val = str(idx)
                if not case_sensitive:
                    idx_val = idx_val.lower()
                if idx_val == target:
                    return report.loc[idx].copy()
            return None

      # ====================================================
      # COLUMN LOOKUP (legacy support)
      # ====================================================
      elif lookup == "columns":
        if isinstance(report.columns, pd.MultiIndex):
            for col in report.columns:
                if len(col) < 2:
                    continue
                col_cat = str(col[0])
                col_met = str(col[1])
                if not case_sensitive:
                    col_cat = col_cat.lower()
                    col_met = col_met.lower()
                if col_cat == cat and col_met == met:
                    return report[col].copy()
            return None
        else:
            target = cat
            if met is not None:
                target = f"{cat}_{met}"
            for col in report.columns:
                col_val = str(col)
                if not case_sensitive:
                    col_val = col_val.lower()
                if col_val == target:
                    return report[col].copy()
            return None
      else:
        raise ValueError(
            "lookup must be 'index' or 'columns'"
        )
    def build_features(self, strategy_results, formula_outputs, raw_data, params=None):
  
      frames = []
      ret_series = None
  
      # =====================================================
      # PHASE 0 — Diagnostic
      # =====================================================
      print(
          f"[FEATURE BUILD] "
          f"strategy_results={len(strategy_results or [])}, "
          f"formula_outputs={len(formula_outputs or [])}, "
          f"raw_data={'YES' if raw_data is not None else 'NO'}"
      )
  
      if strategy_results and formula_outputs:
          print("[FEATURE BUILD] Mode: PURE SRFO (strategies + formula outputs)")
      elif formula_outputs:
          print("[FEATURE BUILD] Mode: FORMULA OUTPUT ONLY")
      elif raw_data is not None:
          print("[FEATURE BUILD] Mode: RAW/FALLBACK FEATURE BUILD")
      else:
          print("[FEATURE BUILD] Mode: UNKNOWN")
  
      # =====================================================
      # PHASE 1
      # Formula outputs from report()
      #
      # Structure:
      #   index   = symbol
      #   columns = MultiIndex(category, metric)
      # =====================================================
      if formula_outputs is not None:
  
          if isinstance(formula_outputs, pd.DataFrame):
              formula_outputs = [formula_outputs]
          elif isinstance(formula_outputs, tuple):
              formula_outputs = list(formula_outputs)
  
          for fo in formula_outputs:
              if not isinstance(fo, pd.DataFrame):
                  continue
              if fo.empty:
                  continue
  
              fo_df = formula_report(
                  fo,
                  mode="df_multi",
                  lookup="columns",
              )
  
              if not isinstance(fo_df, pd.DataFrame):
                  continue
              if fo_df.empty:
                  continue
  
              # -------------------------------------------------
              # REPORT FORMAT
              #   index   = symbol
              #   columns = MultiIndex(category, metric)
              # -------------------------------------------------
              if isinstance(fo_df.columns, pd.MultiIndex):
  
                  feature_df = fo_df.copy()
  
                  # extract target (only once)
                  if ret_series is None:
                      if ("market", "ret") in feature_df.columns:
                          ret_series = feature_df[("market", "ret")].copy()
  
                  # flatten columns
                  feature_df.columns = [
                      f"fo_{c[0]}_{c[1]}"
                      for c in feature_df.columns
                  ]
                  feature_df.index.name = "symbol"
                  feature_df = feature_df.reset_index()
  
                  # remove target
                  feature_df = feature_df.drop(
                      columns=["fo_market_ret"],
                      errors="ignore"
                  )
                  frames.append(feature_df)
  
      # =====================================================
      # PHASE 2
      # Strategy Results
      # =====================================================
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
              frames.append(
                  pd.DataFrame([
                      {"symbol": s, f"signal_{idx}": v}
                      for s, v in signals.items()
                  ])
              )
  
      if not frames:
          raise ValueError("No features generated")
  
      # =====================================================
      # PHASE 3
      # Merge features
      # =====================================================
      feature_df = frames[0]
  
      for df in frames[1:]:
          feature_df = feature_df.merge(
              df,
              on="symbol",
              how="outer"
          )
  
      # =====================================================
      # PHASE 4
      # Clean target
      # =====================================================
      if ret_series is None:
          raise ValueError("Missing market.ret target")
  
      if isinstance(ret_series, pd.DataFrame):
          ret_series = ret_series.iloc[:, 0]
  
      ret_series = (
          pd.Series(ret_series)
          .astype(float)
          .replace([np.inf, -np.inf], np.nan)
          .dropna()
      )
  
      if ret_series.empty:
          raise ValueError("market.ret contains only NaN")
  
      ret_series.index = ret_series.index.astype(str)
      ret_series.index.name = "symbol"
  
      # =====================================================
      # PHASE 5
      # Join X / y
      # =====================================================
      y_df = ret_series.rename("y").reset_index()
  
      feature_df["symbol"] = feature_df["symbol"].astype(str)
      y_df["symbol"] = y_df["symbol"].astype(str)
  
      feature_df = feature_df.merge(
          y_df,
          on="symbol",
          how="inner"
      )
  
      if feature_df.empty:
          raise ValueError("No symbol overlap between features and target")
  
      y = feature_df["y"]
  
      X = feature_df.drop(
          columns=["symbol", "y"],
          errors="ignore"
      )
  
      X = self.encode_features(X)
  
      X = (
          X
          .replace([np.inf, -np.inf], np.nan)
          .fillna(0)
      )
  
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

       # no internal split - caller controls train/validation
       X_train = X
       y_train = y

       model_type = params.get("model", "random_forest")
       n_jobs = params.get("n_jobs", -1)

       if model_type == "random_forest":
        model = RandomForestRegressor(
            n_estimators=params.get("trees", 500),
            max_depth=params.get("depth", 12),
            min_samples_split=params.get("min_samples_split", 2),
            max_features=params.get("max_features", "sqrt"),
            random_state=42,
            n_jobs=n_jobs
        )

       elif model_type == "gradient_boosting":
         model = GradientBoostingRegressor(
            n_estimators=params.get("trees", 500),
            max_depth=params.get("depth", 6),
            min_samples_split=params.get("min_samples_split", 2),
            max_features=params.get("max_features", "sqrt"),
            random_state=42
         )
       elif model_type == "ridge":
          model = Ridge(alpha=1.0)
       else:
         model = RandomForestRegressor(
            n_estimators=params.get("trees", 500),
            max_depth=params.get("depth", 12),
            random_state=42,
            n_jobs=n_jobs
         )
       model.fit(X_train, y_train)

       # evaluate on training data because no split here
       pred = model.predict(X_train)

       metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_train, pred))),
        "r2": float(r2_score(y_train, pred)),
        "samples": len(X_train),
        "features": len(X_train.columns),
        "model_type": model_type
       }

       return {
        "model": model,
        "features": list(X.columns),
        "metrics": metrics
       }, metrics

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
          """
          Collect precomputed metrics.

          Priority:
            1. formula_outputs -> formula_report()
            2. fallback -> X dataframe (_build_raw output)

          Returns scalar metrics required by SRFO.
          """

          metric_map = {
              "alpha": "alpha_alpha",
              "beta": "alpha_beta",
              "ret": "market_ret",
              "corr": "basic_corr",
              "hit": "basic_hit_ratio",
              "tstat": "basic_tstat",
              "turnover": "execution_turnover",
              "tcost": "execution_transaction_cost",
              "ic": "intel_ic",
              "volatility": "risk_volatility",
              "cvar": "risk_cvar",
              "sharpe": "risk_sharpe",
              "drawdown": "risk_drawdown",
              "max_drawdown": "risk_max_drawdown",
              "score": "decision_score",
              "signal": "decision_psignal",
          }


          metrics = {}


          # =====================================================
          # PATH 1: formula_outputs -> formula_report
          # =====================================================

          formula_outputs = package.get("formula_outputs", [])


          if formula_outputs:

              for fo in formula_outputs:

                  if not hasattr(fo, "reporting"):
                      continue


                  try:
                      report_out = fo.reporting()

                      report = formula_report(
                          report_out,
                          mode="df_multi",
                          lookup="columns",
                      )

                  except Exception:
                      continue


                  if not isinstance(report, pd.DataFrame):
                      continue

                  if report.empty:
                      continue

                  if not isinstance(report.columns, pd.MultiIndex):
                      continue


                  for col in report.columns:

                      # FormulaOutput.reporting()
                      # columns are (category, metric)

                      if isinstance(col, tuple) and len(col) == 2:
                          source_key = f"{col[0]}_{col[1]}".lower()
                      else:
                          source_key = str(col).lower()


                      for output_key, lookup_key in metric_map.items():

                          if source_key != lookup_key.lower():
                              continue


                          value = report[col]


                          # symbol series -> scalar
                          if isinstance(value, pd.Series):

                              value = (
                                  value
                                  .replace([np.inf, -np.inf], np.nan)
                                  .dropna()
                              )


                              if value.empty:
                                  continue


                              value = value.mean()


                          if pd.isna(value):
                              continue


                          try:
                              metrics[output_key] = float(value)

                          except Exception:
                              pass



          # =====================================================
          # PATH 2: fallback from X after _build_raw
          # =====================================================

          X = package.get("X")


          if isinstance(X, pd.DataFrame):

              for output_key, column_name in metric_map.items():

                  # preserve formula_report values
                  if output_key in metrics:
                      continue


                  if column_name not in X.columns:
                      continue


                  value = (
                      X[column_name]
                      .replace([np.inf, -np.inf], np.nan)
                      .dropna()
                  )


                  if value.empty:
                      continue


                  # symbol metrics -> portfolio scalar
                  metrics[output_key] = float(value.mean())



          # =====================================================
          # FINAL VALIDATION
          # =====================================================

          if "score" not in metrics:
              print(
                  "[EVALUATE] Missing score. "
                  f"Available metrics={list(metrics.keys())}"
              )

          return metrics


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
                    #formula_outputs.append(s.formulaOutput.reporting())
                    report_out = formula_output.reporting()
                    report = formula_report( report_out,mode="df_easy",lookup="index",  )
                    formula_outputs.append(  report )
            ml = MLTrainEngine()
            X, y = ml.build_feature_matrix(strategy_results=engine.results, formula_outputs=formula_outputs, raw_data=stressed_data)
        except Exception as e:
            ml = MLTrainEngine()
            X,y = self._build_features_from_raw(stressed_data)
        if X.shape[0] == 0 or len(y) == 0:
            return None
        package = {"X": X, "y": y, "features": list(X.columns)}
        signal = self.backtest.signal(model_package, package)
        metrics = self.backtest.evaluate(package, signal)
        return {"scenario": scenario_name, "params": params, "metrics": metrics,
                "sharpe": metrics["sharpe"], "score": metrics["score"], "corr": metrics["corr"]}

    def _build_features_from_raw(self, data):
        """
        Fallback feature builder.

        Returns a DataFrame with MultiIndex columns
        (category, metric) so build_features() can
        flatten them to fo_category_metric and extract
        the ("market", "ret") target.
        Build fallback features using FormulaInfo and the
        same ML feature parser used by the SRFO engine.
        """
        run_fo = FormulaInfo(data)
        # Generate assembled formula output
        assemble = run_fo.assemble()
        foreport_out = formula_output.reporting()
        formula_outputs = formula_report( foreport_out,mode="df_easy",lookup="index",  )

        # Apply SRFO unique namespace
        #formula_outputs = [
        #   SRFORegistry.wrap_formula_output(assemble)
        #]

        # Use same parser as SRFO engine
        ml = MLTrainEngine()
        X, y = ml.build_feature_matrix(
          strategy_results=[],
          formula_outputs=formula_outputs,
          raw_data=data,
        )

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
# SRFO static registry of sr and fo instances
# =====================================================
class SRFORegistry:
    """
    SRFO uniqueness manager.

    Purpose:
    - Assign unique IDs to formulaOutput instances.
    - Prevent duplicate report column names.
    - No object storage.
    - No caching.
    - Objects live only in the current pipeline scope.
    """
    _counter = 0

    @classmethod
    def reset(cls):
        """
        Reset namespace for a new SRFO generation.
        """
        cls._counter = 0
    @classmethod
    def next_id(cls):
        """
        Generate unique formulaOutput namespace.
        """
        cls._counter += 1
        return f"fo_{cls._counter}"
    @classmethod
    def collect_formula_outputs(cls, strategies, reset=False):
        """
        Collect formulaOutput objects and assign unique IDs.
        Returns:
           [
                {
                    "id": "fo_1",
                    "strategy_id": 1,
                    "object": formulaOutput
                }
            ]
        """

        if reset:
            cls.reset()
        outputs = []
        for idx, strategy in enumerate(strategies, start=1):
            fo = getattr(
                strategy,
                "formulaOutput",
                getattr(strategy, "formula_output", None)
            )
            if fo is None:
                continue
            outputs.append(
                {
                    "id": cls.next_id(),
                    "strategy_id": idx,
                    "object": fo
                }
            )
        return outputs

    @classmethod
    def wrap_formula_output(cls, fo, strategy_id=None):
        """
        Used for raw FormulaInfo fallback path.
        """
        return {
            "id": cls.next_id(),
            "strategy_id": strategy_id,
            "object": fo
        }

    @classmethod
    def reports(cls, formula_outputs):
        """
        Convert formulaOutput.report() into uniquely named feature frames.
        Each report receives:
            fo_1_market_price
            fo_2_market_price
            ...

        preventing merge collisions.
        """
        reports = []
        for item in formula_outputs:
            fo_id = item["id"]
            fo = item["object"]
            if not hasattr(fo, "reporting"):
                continue
            df = fo.reporting().copy()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [
                    f"{fo_id}_{c[0]}_{c[1]}"
                    for c in df.columns
                ]
            else:
                df.columns = [
                    f"{fo_id}_{c}"
                    for c in df.columns
                ]
            reports.append(df)
        return reports
# =====================================================
# AGENTIC PIPELINE (MAIN ORCHESTRATOR)
# =====================================================

class AgenticPipeline:

    def __init__(self, data, base_dir="runs"):
        self.data = data
        self.base_dir = base_dir        
        # ... existing init code ...
        self.current_run_idx = None
        self.current_run_dir = None
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
        self._formula_outputs_raw = None

        # ... existing init code ...
        
        # === SRFO CACHE: generated once, used everywhere ===
        self._srfo_full = None
        self._Xy_train = None
        self._Xy_val = None

    def _generate_srfo(self, data):
      """
      Run QuantXEngine ONCE.
      Returns:
        {
            results,
            formula_outputs_raw,
            raw_data
        }

      Raw formulaOutput objects are preserved for downstream consumers.
      ML feature extraction should create namespaced .report() copies separately.
      """
      try:
        from ..qxEngine import QuantXEngine
        engine = QuantXEngine()
        strategies = engine.qxStrategyList(  data, interval="4y" )
        """
        formula_outputs_raw = ( SRFORegistry.collect_formula_outputs( engine.strategy ) )
        print(
            f"[SRFO] Engine: "
            f"{len(strategies)} strategies, "
            f"{len(formula_outputs_raw)} raw formulaOutput objects"
        )
        formula_outputs_report = ( SRFORegistry.reports( formula_outputs_raw )  )
        
        return {
            "results": engine.results,
            #"formula_outputs_raw": formula_outputs_raw,
            #"formula_outputs_report": formula_outputs_report,
            formula_outputs=formula_outputs
            "raw_data": data
        }
        """
        formula_outputs = []
        for strategy in engine.strategy:
           formula_output = getattr(strategy, "formulaOutput", None )
           formula_output.assemble()
           report_out = formula_output.reporting()
           report = formula_report( report_out,mode="df_easy",lookup="index",  )
                    
           if formula_output is None:
              continue
           formula_outputs.append( report )
        return {
            "results": engine.results,
            "formula_outputs": formula_outputs,
            "raw_data": data
        }

      except Exception as e:
        print( f"[SRFO] Engine failed: {e}" )
        return None


    def _ensure_srfo(self):
        """Lazy init: generate SRFO once. If engine output fails to parse, use fallback."""
        if self._srfo_full is not None:
            return

        print("[SRFO] Generating raw SRFO from full dataset...")
        self._srfo_full = self._generate_srfo(self.data)
        print("[DEBUG SRFO]",self._srfo_full)
        results, formula_outputs_raw = None, []
        X, y = None, None

        if self._srfo_full:
            results = self._srfo_full.get("results")
            #formula_outputs_raw = self._srfo_full.get("formula_outputs_raw", [])
            # ML needs .report() DataFrames
            #formula_outputs_report = ( self._srfo_full.get( "formula_outputs_report", [] ))
            formula_outputs = ( self._srfo_full.get( "formula_outputs", [] ))

            ml = MLTrainEngine()
            try:
                X, y = ml.build_feature_matrix(       
                    strategy_results=results,
                    formula_outputs=formula_outputs,   #####formula_outputs_report,
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
            results, formula_outputs_raw = None, []
            print(f"[SRFO] Fallback feature matrix: {X.shape}")

        # Temporal 50/50 split
        n = len(y)
        split_idx = n // 2

        self._Xy_train = {
            "X": X.iloc[:split_idx].copy(),
            "y": y.iloc[:split_idx].copy(),
            "features": list(X.columns),
            "results": results,
            "formula_outputs": formula_outputs
        }
        self._Xy_val = {
            "X": X.iloc[split_idx:].copy(),
            "y": y.iloc[split_idx:].copy(),
            "features": list(X.columns),
            "results": results,
            "formula_outputs": formula_outputs
        }
        # Cache raw objects for .assemble() in portfolio construction
        if formula_outputs_raw:
           self._formula_outputs_raw = formula_outputs_raw
        else:
           self._formula_outputs_raw = (
             self._srfo_full.get("formula_outputs", [])
             if self._srfo_full
              else []
           )
        print(f"[SRFO] Cached raw formulaOutput objects: {len(self._formula_outputs_raw)}")
        print(f"[SRFO] Split: train={len(self._Xy_train['y'])}, val={len(self._Xy_val['y'])}")

        # Store full X,y for potential scenario stress testing
        self._Xy_full = {"X": X.copy(), "y": y.copy(), "features": list(X.columns)}

    def _run_combinatorial_srfo(self, param_grid, max_combinations=500, scenarios=None):
        """Combinatorial search using cached SRFO X,y. No engine re-runs."""
        import time
        import gc

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
            d["n_jobs"] = 1          # ← FORCE SINGLE-THREAD
            combinations.append(d)

        print(f"[OPTIMIZER] {len(combinations)} combos × {len(scenarios)} scenarios = {len(combinations)*len(scenarios)} evals")

        # Shallow copies — we never mutate these frames
        X_train = self._Xy_train["X"].copy()
        y_train = self._Xy_train["y"].copy()
        X_val   = self._Xy_val["X"].copy()
        y_val   = self._Xy_val["y"].copy()

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

                # Train — single-threaded, small memory footprint
                ml = MLTrainEngine()
                trained = ml.train({"X": X_train, "y": y_tr_s, "params": params})
                model_pkg = {"model": trained["model"], "label_encoders": trained.get("label_encoders", {})}

                signal = self.backtest.signal(model_pkg, {"X": X_val, "y": y_val_s})
                metrics = self.backtest.evaluate({"X": X_val, "y": y_val_s}, signal)

                scenario_scores.append(metrics["score"])
                scenario_sharpes.append(metrics["sharpe"])
                all_results.append({
                    "scenario": scenario,
                    "params": {k: v for k, v in params.items() if k != "_hash"},
                    "sharpe": metrics["sharpe"],
                    "score": metrics["score"],
                    "corr": metrics["corr"]
                })

                # === CRITICAL: free memory immediately ===
                del trained, model_pkg, signal, metrics
                self.backtest.last_signal = None
                self.backtest.last_features = None
                gc.collect()

            # === PROGRESS ===
            resiliency = 0.6 * min(scenario_sharpes) + 0.4 * np.mean(scenario_scores) if scenario_scores else -999
            elapsed = time.time() - start_time
            avg_per_combo = elapsed / (idx + 1)
            remaining = avg_per_combo * (len(combinations) - idx - 1)

            new_best_flag = ""
            if resiliency > best_resiliency:
                best_resiliency = resiliency
                best_params = params
                new_best_flag = " *** NEW BEST ***"

            if (idx + 1) % 10 == 0 or idx == 0 or idx == len(combinations) - 1:
                print(f"  [{idx+1:3d}/{len(combinations)}] "
                      f"res={resiliency:.4f} best={best_resiliency:.4f} "
                      f"min_sharpe={min(scenario_sharpes):.4f} avg_score={np.mean(scenario_scores):.4f} "
                      f"elapsed={elapsed/60:.1f}m ETA={remaining/60:.1f}m{new_best_flag}")

        self.storage.save_json({
            "best_params": {k: v for k, v in best_params.items() if k != "_hash"} if best_params else None,
            "best_resiliency_score": float(best_resiliency),
            "total_combinations": len(combinations),
            "timestamp": datetime.now().isoformat()
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
                   s.formulaOutput.assemble()
                   report_out = s.formulaOutput.reporting()
                   report = formula_report( report_out,mode="df_easy",lookup="index",  )
                   formula_outputs.append(report)

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

    def _build_features_from_raw(self,data):
      # Create raw FormulaInfo
 
        run_fo = FormulaInfo(data)
        rpt_out, rpt_out_easy = run_fo.reporting()
        rpt = formula_report( (rpt_out, rpt_out_easy), "df_easy", "symbol_features")
        print(rpt.index)
        print(rpt.columns)
        y = rpt["market_ret"].copy()
        X = rpt.drop( columns=["market_ret"],  errors="ignore")
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        y = y.replace([np.inf, -np.inf], np.nan).fillna(0)

        print(y.dtype)   # should be float64
        return X, y



    def run_agent(self, params=None, max_iters=5, run_combinatorial=True, param_grid=None):
          if params is None:
            params = {}

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
            "max_features": params.get("max_features", "sqrt"), "model": params.get("model", "random_forest")
          }
          baseline_train = self.strategy.train(self.get_train_package(baseline_params))
          self.baseline_model_package = deepcopy(baseline_train)
          self.storage.save(self.baseline_model_package, f"{self.current_run_dir}/baseline/baseline_model.pkl")

          baseline_val_pkg = self.get_val_package()
          baseline_signal = self.backtest.signal(self.baseline_model_package, baseline_val_pkg)
          self.baseline_val_metrics = self.backtest.evaluate(baseline_val_pkg, baseline_signal)
          print(f"[BASELINE] Score: {self.baseline_val_metrics.get('score', 0):.4f}")
          self.storage.save(self.baseline_val_metrics, f"{self.current_run_dir}/baseline/baseline_val_metrics.pkl")

          # === PHASE 2: COMBINATORIAL ===
          if run_combinatorial:
            print(f"\n{'='*60}")
            print("PHASE 2: COMBINATORIAL SEARCH (SRFO-CACHED)")
            print(f"{'='*60}")
            if param_grid is None:
                param_grid = PARAM_GRID_INSIDE
            opt_result = self._run_combinatorial_srfo(param_grid, max_combinations=100)
            best_params = opt_result["best_params"]
            if best_params:
                for k in ["trees", "depth", "horizon", "min_samples_split", "max_features", "model"]:
                    if k in best_params:
                        params[k] = best_params[k]
                print(f"[OPTIMIZER] Best: {best_params}, Resiliency: {opt_result['best_resiliency_score']:.4f}")
            self.storage.save_json(opt_result, f"{self.current_run_dir}/combinatorial/combinatorial_optimization.json")

          # === PHASE 3: OPTIMIZED ===
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

            print(f"Train: Sharpe={train_metrics.get('sharpe', 0):.3f} Score={train_metrics.get('score', 0):.3f}")
            print(f"Val:   Sharpe={val_metrics.get('sharpe', 0):.3f} Score={val_metrics.get('score', 0):.3f}")

            review = self.review_engine.compare(val_metrics, previous_val)
            decision = self.evaluator.decide(review, train_metrics)

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

            # Save all iteration artifacts
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
                params = self._apply_mutations(params, review.get("recommendations", []))

            previous_val = val_metrics

        # ============================================================
        # PORTFOLIO HELPERS (self-contained, no dependency on _formula_outputs_raw)
        # ============================================================

          def _run_engine_and_assemble(data, label="engine"):
            """Run QuantXEngine on data and return assembled formulaOutput."""
            try:
                from ..qxEngine import QuantXEngine
                engine = QuantXEngine()
                strategies = engine.qxStrategyList(data, interval=params.get("interval", "4y"))
                for s in engine.strategy:
                    fo_obj = getattr(s, "formulaOutput", getattr(s, "formula_output", None))
                    if fo_obj is not None:
                        fo = fo_obj.assemble()
                        print(f"[PORTFOLIO] [{label}] Engine OK, formulaOutput assembled.", fo_obj)
                        return fo_obj
                print(f"[PORTFOLIO] [{label}] No formulaOutput found on strategies.")
                return None
            except Exception as e:
                print(f"[PORTFOLIO] [{label}] Engine failed: {e}")
                return None

          def _build_portfolio(fo, data_source, label="portfolio"):
            """Build, invoke, and persist a Portfolio from assembled formula output."""
            if fo is None:
                print(f"[PORTFOLIO] [{label}] No formulaOutput. Skipping.")
                return None

            try:
                # ---- returns -------------------------------------------------
                ret_df = fo.get("ret")
                if ret_df is None or not isinstance(ret_df, pd.DataFrame):
                    ret_parts = []
                    for sym, df in data_source.items():
                        if isinstance(df, pd.DataFrame) and "close" in df.columns:
                            ret_parts.append(df["close"].pct_change().fillna(0).rename(sym))
                    ret_df = pd.concat(ret_parts, axis=1).fillna(0) if ret_parts else None

                if ret_df is None or ret_df.empty:
                    raise ValueError("No return data.")

                # ---- weights from fo -----------------------------------------
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

                # ---- benchmark -----------------------------------------------
                benchmark = fo.get("benchmark")
                if benchmark is None:
                    benchmark = ret_df["BTCUSDT"] if "BTCUSDT" in ret_df.columns else ret_df.mean(axis=1)
                elif isinstance(benchmark, pd.DataFrame):
                    benchmark = benchmark.iloc[:, 0]
                benchmark = benchmark.squeeze()

                # ---- transaction_cost ----------------------------------------
                transaction_cost = fo.get("transaction_cost")
                if transaction_cost is None:
                    transaction_cost = 0.0005
                if hasattr(transaction_cost, "iloc"):
                    transaction_cost = float(transaction_cost.iloc[0]) if len(transaction_cost) > 0 else 0.0005
                try:
                    transaction_cost = float(transaction_cost)
                except (TypeError, ValueError):
                    transaction_cost = 0.0005

                # ---- align & sort chronologically ----------------------------
                common_idx = (
                    ret_df.index
                    .intersection(weights.index)
                    .intersection(benchmark.index)
                )
                ret_df    = ret_df.loc[common_idx].sort_index()
                weights   = weights.loc[common_idx].sort_index()
                benchmark = benchmark.loc[common_idx].sort_index()

                # ---- invoke Portfolio ----------------------------------------
                portfolio = Portfolio()
                result = portfolio.invoke(
                    fo=fo,
                    weights=weights,
                    returns=ret_df,
                    benchmark=benchmark,
                    transaction_cost=transaction_cost,
                    annualization=252
                )

                # ---- persist -------------------------------------------------
                chart_path = os.path.join(self.current_run_dir, f"{label}_chart.html")
                result["chart"].write_html(chart_path)
                print(f"[PORTFOLIO] [{label}] Chart saved: {chart_path}")

                self.storage.save(result["metrics"], f"{self.current_run_dir}/{label}_metrics.pkl")
                self.storage.save_json(result["metrics"], f"{self.current_run_dir}/{label}_metrics.json")
                self.storage.save(result["series"], f"{self.current_run_dir}/{label}_series.pkl")

                # ---- console metrics -----------------------------------------
                m = result["metrics"]
                print(f"[PORTFOLIO] [{label}] Metrics:")
                print(f"  Gross Return:      {m.get('gross_return', 0):.4f}")
                print(f"  Net Return:        {m.get('net_return', 0):.4f}")
                print(f"  Sharpe:            {m.get('sharpe', 0):.4f}")
                print(f"  Volatility:        {m.get('volatility', 0):.4f}")
                print(f"  Max Drawdown:      {m.get('max_drawdown', 0):.4f}")
                print(f"  BTC Correlation:   {m.get('btc_correlation', 0):.4f}")
                print(f"  Alpha:             {m.get('alpha', 0):.6f}")
                print(f"  Beta:              {m.get('beta', 0):.4f}")
                print(f"  Alpha t-stat:      {m.get('tstat_alpha', 0):.4f}")
                print(f"  Hit Ratio:         {m.get('hit_ratio', 0):.4f}")
                print(f"  IC:                {m.get('ic', 0):.4f}")
                print(f"  Mean Turnover:     {m.get('mean_turnover', 0):.4f}")
                print(f"  Total Costs:       {m.get('total_costs', 0):.4f}")

                return result

            except Exception as e:
                print(f"[PORTFOLIO] [{label}] Failed: {e}")
                import traceback
                traceback.print_exc()
                return None


          def _build_portfolio0(fo, data_source, label="portfolio"):
            """Build, invoke, and persist a Portfolio from assembled formula output."""
            if fo is None:
                print(f"[PORTFOLIO] [{label}] No formulaOutput. Skipping.")
                return None

            try:
                # ---- returns -------------------------------------------------
                ret_df = fo.get("ret")
                if ret_df is None or not isinstance(ret_df, pd.DataFrame):
                    ret_parts = []
                    for sym, df in data_source.items():
                        if isinstance(df, pd.DataFrame) and "close" in df.columns:
                            ret_parts.append(df["close"].pct_change().fillna(0).rename(sym))
                    ret_df = pd.concat(ret_parts, axis=1).fillna(0) if ret_parts else None

                if ret_df is None or ret_df.empty:
                    raise ValueError("No return data.")

                # ---- weights from FORMULA_CONFIG -----------------------------
                # ??? check if norm_weight or weight or executed_weight

                weights =  fo.get("executed_weight") 

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

                # ---- benchmark -----------------------------------------------
                benchmark = fo.get("benchmark")
                if benchmark is None:
                    benchmark = ret_df["BTCUSDT"] if "BTCUSDT" in ret_df.columns else ret_df.mean(axis=1)
                elif isinstance(benchmark, pd.DataFrame):
                    benchmark = benchmark.iloc[:, 0]

                # ---- transaction_cost (from FORMULA_CONFIG or fallback) ------
                transaction_cost = fo.get("transaction_cost")
                if transaction_cost is None:
                    transaction_cost = 0.0005
                if hasattr(transaction_cost, "iloc"):
                    transaction_cost = float(transaction_cost.iloc[0]) if len(transaction_cost) > 0 else 0.0005
                try:
                    transaction_cost = float(transaction_cost)
                except (TypeError, ValueError):
                    transaction_cost = 0.0005

                # ---- align ---------------------------------------------------
                common_idx = (
                    ret_df.index
                    .intersection(weights.index)
                    .intersection(benchmark.index)
                )
                ret_df = ret_df.loc[common_idx]
                weights = weights.loc[common_idx]
                benchmark = benchmark.loc[common_idx]

                # ---- invoke Portfolio ----------------------------------------
                portfolio = Portfolio()
                result = portfolio.invoke(
                    fo = fo,
                    weights=weights,
                    returns=ret_df,
                    benchmark=benchmark,
                    transaction_cost=transaction_cost,
                    annualization=252
                )

                # ---- persist (headless-safe) ---------------------------------
                chart_path = os.path.join(self.current_run_dir, f"{label}_chart.html")
                result["chart"].write_html(chart_path)
                print(f"[PORTFOLIO] [{label}] Chart saved: {chart_path}")
                print(result["metrics"])

                self.storage.save(result["metrics"], f"{self.current_run_dir}/{label}_metrics.pkl")
                self.storage.save_json(result["metrics"], f"{self.current_run_dir}/{label}_metrics.json")
                return result

            except Exception as e:
                print(f"[PORTFOLIO] [{label}] Failed: {e}")
                import traceback
                traceback.print_exc()
                return None

        # ============================================================
        # PORTFOLIO 1 — TRAINED DATA (1st 50%)
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
            # Try FormulaInfo first (user preference), fallback to engine
            try:
                from ..qxEngine import FormulaInfo
                trained_fo = FormulaInfo(trained_data)
                fo_train = trained_fo.assemble()
                chk_formula_output = trained_fo
                print("[PORTFOLIO] [train] Using FormulaInfo(trained_data).assemble()")
            except Exception as e:
                print(f"[PORTFOLIO] [train] FormulaInfo failed ({e}), falling back to QuantXEngine")
                fo_train = _run_engine_and_assemble(trained_data, label="train")
                chk_formula_output = fo_train
            _build_portfolio(chk_formula_output, trained_data, label="portfolio_train")

          except Exception as e:
            print(f"[PORTFOLIO] [train] Outer exception: {e}")
            import traceback
            traceback.print_exc()

        # ============================================================
        # PORTFOLIO 2 — MERGED / FULL DATA (complete dataset)
        # ============================================================
          print(f"\n{'='*60}")
          print("PORTFOLIO: MERGED / FULL DATA")
          print(f"{'='*60}")

          fo_merged = None

        # 1st try: instance cache (works if _ensure_srfo was updated)
          raw_cache = getattr(self, '_formula_outputs_raw', None)
          if raw_cache and len(raw_cache) > 0:
            try:
                fo_merged = raw_cache[0].assemble()
                chk_formula_output = fo_merged
                print("[PORTFOLIO] [merged] Using _formula_outputs_raw cache.")
            except Exception as e:
                print(f"[PORTFOLIO] [merged] Cache failed: {e}")

        # 2nd try: SRFO dict (works if _generate_srfo was updated)
          if fo_merged is None and getattr(self, '_srfo_full', None):
            srfo_raw = self._srfo_full.get('formula_outputs_raw')
            if srfo_raw and len(srfo_raw) > 0:
                try:
                    fo_merged = srfo_raw[0].assemble()
                    chk_formula_output = srfo_raw[0]
                    print("[PORTFOLIO] [merged] Using _srfo_full cache.")
                except Exception as e:
                    print(f"[PORTFOLIO] [merged] SRFO cache failed: {e}")

        # 3rd try: run engine directly (always works, just slower)
          if fo_merged is None:
            fo_merged = _run_engine_and_assemble(self.data, label="merged")
            chk_formula_output = fo_merged
          _build_portfolio(chk_formula_output, self.data, label="portfolio_merged")

        # === FINAL SUMMARY ===
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

        print(f"\n{'='*50}FINAL SUMMARY — RUN {self.current_run_idx:06d}{'='*50}")
        print(f"Run directory: {self.current_run_dir}")
        print(f"Total iterations: {self.iteration}")
        print(f"Best validation score: {self.best_score:.4f}")
        print(f"GoLive ready: {summary['golive_ready']}")
        return summary

# =====================================================
# DATA LOADER
# =====================================================

def build_data():
    #symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT"]
    symbols = ["BTCUSDT","ETHUSDT","XRPUSDT","BNBUSDT","SOLUSDT","DOGEUSDT","ADAUSDT","TRXUSDT","HYPEUSDT","SUIUSDT","LINKUSDT","AVAXUSDT","XLMUSDT","HBARUSDT","BCHUSDT","LTCUSDT","SHIBUSDT","DOTUSDT","AAVEUSDT","PEPEUSDT","NEARUSDT","APTUSDT","ICPUSDT","ETCUSDT","ONDOUSDT","POLUSDT","CROUSDT","TONUSDT", "UNIUSDT"]

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
# Portfolio class
# =====================================================
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly
import plotly.colors
import os
from typing import Dict, Any, Optional


class Portfolio:
    """
    Portfolio backtest engine.
    1. invoke()  — aligns data, reads fo metrics, produces chart + metrics dict.
    2. build()   — wrapper that assembles inputs from fo, calls invoke(),
                   and optionally persists to disk.
    """

    def __init__(self, run_dir: Optional[str] = None, storage: Optional[Any] = None):
        self.run_dir = run_dir
        self.storage = storage

    # =====================================================================
    # invoke — core backtest logic (reads from fo, minimal calculation)
    # =====================================================================
    def invoke(
        self,
        fo: Dict[str, Any],
        weights: pd.DataFrame,
        returns: pd.DataFrame,
        benchmark: pd.Series,
        transaction_cost: float = 0.0005,
        annualization: int = 252,
    ) -> Dict[str, Any]:
        # ------------------------------------------------------------------
        # 1. Align & sort chronologically
        # ------------------------------------------------------------------
        idx = (
            weights.index
            .intersection(returns.index)
            .intersection(benchmark.index)
        )
        weights   = weights.loc[idx].sort_index()
        returns   = returns.loc[idx].sort_index()
        benchmark = benchmark.loc[idx].sort_index()

        # ------------------------------------------------------------------
        # 2. Executed weights — from fo
        # ------------------------------------------------------------------
        executed_weight = fo.get("executed_weight")
        if isinstance(executed_weight, pd.DataFrame):
            executed_weight = (
                executed_weight
                .loc[executed_weight.index.intersection(idx)]
                .reindex(idx)
                .fillna(0)
                .sort_index()
            )
            for c in returns.columns:
                if c not in executed_weight.columns:
                    executed_weight[c] = 0.0
            executed_weight = executed_weight[returns.columns].fillna(0)
        else:
            executed_weight = weights.shift(1).fillna(0)

        lagged_weights = executed_weight.copy()

        # ------------------------------------------------------------------
        # 3. Gross return — PRIMARY: fo["strategy_ret"]
        # ------------------------------------------------------------------
        gross_return = fo.get("strategy_ret")
        if isinstance(gross_return, pd.Series):
            gross_return = gross_return.reindex(idx).fillna(0).sort_index()
        elif isinstance(gross_return, pd.DataFrame) and gross_return.shape[1] == 1:
            gross_return = gross_return.iloc[:, 0].reindex(idx).fillna(0).sort_index()
        else:
            gross_return = (executed_weight * returns).sum(axis=1)
        gross_return.name = "gross_return"

        # ------------------------------------------------------------------
        # 4. Costs — scalar rate from fo, series = turnover * rate
        # ------------------------------------------------------------------
        tc_rate = fo.get("transaction_cost", transaction_cost)
        if hasattr(tc_rate, "iloc"):
            tc_rate = float(tc_rate.iloc[0]) if len(tc_rate) > 0 else 0.0005
        try:
            tc_rate = float(tc_rate)
        except (TypeError, ValueError):
            tc_rate = 0.0005

        turnover_series = fo.get("turnover")
        if isinstance(turnover_series, pd.Series):
            turnover_series = turnover_series.reindex(idx).fillna(0).sort_index()
        else:
            turnover_series = weights.diff().abs().sum(axis=1)
            turnover_series.iloc[0] = weights.iloc[0].abs().sum()
        turnover_series.name = "turnover"

        costs = turnover_series * tc_rate
        costs.name = "costs"

        # ------------------------------------------------------------------
        # 5. Net return — PRIMARY: fo["net_ret"], fallback: gross - costs
        # ------------------------------------------------------------------
        net_return = fo.get("net_ret")
        if isinstance(net_return, pd.Series):
            net_return = net_return.reindex(idx).fillna(0).sort_index()
        elif isinstance(net_return, pd.DataFrame) and net_return.shape[1] == 1:
            net_return = net_return.iloc[:, 0].reindex(idx).fillna(0).sort_index()
        else:
            net_return = gross_return - costs
        net_return.name = "net_return"

        # ------------------------------------------------------------------
        # 6. Equity & Drawdown — PRIMARY from fo, fallback compute
        # ------------------------------------------------------------------
        equity = fo.get("equity")
        if isinstance(equity, pd.Series):
            net_equity = equity.reindex(idx).fillna(0).sort_index()
        else:
            net_equity = (1 + net_return).cumprod()

        gross_equity = (1 + gross_return).cumprod()
        benchmark_equity = (1 + benchmark).cumprod()

        drawdown = fo.get("drawdown")
        if isinstance(drawdown, pd.Series):
            drawdown = drawdown.reindex(idx).fillna(0).sort_index()
        else:
            running_max = net_equity.cummax()
            drawdown = net_equity / running_max - 1
        drawdown.name = "drawdown"

        # ------------------------------------------------------------------
        # 7. METRICS — READ DIRECTLY FROM fo (no calculation)
        # ------------------------------------------------------------------
        def _scalar(key, default=0.0):
            v = fo.get(key)
            if v is None:
                return default
            if isinstance(v, pd.Series):
                return float(v.iloc[-1]) if len(v) > 0 else default
            if isinstance(v, pd.DataFrame):
                return float(v.iloc[-1, 0]) if v.size > 0 else default
            try:
                return float(v)
            except (TypeError, ValueError):
                return default

        #sharpe       = _scalar("sharpe")
        #volatility   = _scalar("volatility")
        #max_drawdown = _scalar("max_drawdown") if fo.get("max_drawdown") is not None else float(drawdown.min())
        #btc_corr     = _scalar("corr_rm")
        #alpha        = _scalar("alpha")
        #beta         = _scalar("beta")
        #tstat_alpha  = _scalar("tstat_alpha")
        #hit_ratio    = _scalar("hit_ratio")
        #ic           = _scalar("ic")
        
        ic= fo.get("ic")
        hit_ratio = fo.get("hit_ratio")
        tstat_alpha = fo.get("tstat_alpha")
        beta = fo.get("beta").mean()
        alpha = fo.get("alpha").mean()
        btc_corr = fo.get("corr_rm").mean()
        max_drawdown = fo.get("max_drawdown")
        volatility  = fo.get("volatility")
        sharpe = fo.get("sharpe")

        gross_cumret = gross_equity.iloc[-1] - 1 if len(gross_equity) else 0.0
        net_cumret   = net_equity.iloc[-1] - 1   if len(net_equity)   else 0.0
        mean_turnover  = turnover_series.mean()
        total_turnover = turnover_series.sum()
        total_costs    = costs.sum()

        # ------------------------------------------------------------------
        # 8. Plottable metric series from fo (time-indexed only)
        # ------------------------------------------------------------------
        plot_keys = ["corr_rm", "drawdown", "sharpe", "alpha", "beta", "tstat_alpha", "weights", "turnover", "hit_ratio", "ic",  "volatility",]
        plot_items = []
        for key in plot_keys:
                if key == "drawdown":
                   obj = drawdown
                elif key == "turnover":
                   obj = turnover_series
                elif key == "weights":
                   obj = weights
                else:
                   obj = fo.get(key)
                if obj is None:
                   continue

                # ------------------------------
                # DataFrame
                # ------------------------------
                if isinstance(obj, pd.DataFrame):

                    obj = obj.copy()

                    if not isinstance(obj.index, pd.DatetimeIndex):
                        obj.index = pd.to_datetime(
                            obj.index,
                            errors="coerce"
                        )

                    obj = obj.dropna(how="all")

                    if not obj.empty:
                        plot_items.append((key, obj))


                # ------------------------------
                # Series
                # ------------------------------
                elif isinstance(obj, pd.Series):

                    obj = obj.copy()

                    if not isinstance(obj.index, pd.DatetimeIndex):
                        obj.index = pd.to_datetime(
                            obj.index,
                            errors="coerce"
                        )

                    obj = obj.dropna()

                    if len(obj):
                        plot_items.append((key, obj))


                # ------------------------------
                # Scalar metrics
                # ------------------------------
                else:

                    obj = pd.Series(
                        obj,
                        index=idx,
                        name=key
                    )

                    plot_items.append((key, obj))     
        # ------------------------------------------------------------------
        # 9. Plotly chart
        # ------------------------------------------------------------------
        n_metric_rows = len(plot_items)
        total_rows = 1 + max(n_metric_rows, 1)

        fig = make_subplots(
            rows=total_rows,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.55] + [0.45 / max(n_metric_rows, 1)] * max(n_metric_rows, 1),
            vertical_spacing=0.08,
            subplot_titles=["Portfolio Performance"] + [
              k.replace("_", " ").title() for k, _ in plot_items
            ],
        )

        fig.add_trace(
            go.Scatter(
                x=gross_equity.index, y=gross_equity,
                name="Gross Return", mode="lines",
                line=dict(color="#2E86AB", width=1.5),
            ), row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=net_equity.index, y=net_equity,
                name="Net Return", mode="lines",
                line=dict(color="#A23B72", width=1.5),
            ), row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=benchmark_equity.index, y=benchmark_equity,
                name="Benchmark", mode="lines",
                line=dict(color="#F18F01", width=1.5, dash="dash"),
            ), row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=drawdown.index, y=drawdown,
                name="Drawdown", fill="tozeroy",
                fillcolor="rgba(231, 76, 60, 0.12)",
                line=dict(color="rgba(231, 76, 60, 0.55)", width=1),
            ), row=1, col=1,
        )
    
        colors = plotly.colors.qualitative.Plotly
        for key, obj in plot_items:
           print(
             key,
             type(obj),
             obj.shape if hasattr(obj, "shape") else len(obj),
             obj.index[:3],
             obj.isna().sum() if isinstance(obj, pd.Series) else obj.isna().sum().to_dict()
        )
        for row, (key, obj) in enumerate(plot_items, start=2):
          print(" plotly ", obj ,type(obj))
          if isinstance(obj, pd.Series):

             fig.add_trace(
                go.Scatter(
                 x=obj.index,
                 y=obj,
                 mode="lines",
                 name=key.replace("_", " ").title(),
                 line=dict(width=1.5),
                ),
                row=row,
                col=1,
             )

          else:

            for j, col in enumerate(obj.columns):

             fig.add_trace(
                go.Scatter(
                    x=obj.index,
                    y=obj[col],
                    mode="lines",
                    name=f"{key}: {col}",
                    legendgroup=key,
                    line=dict(
                        width=1,
                        color=colors[j % len(colors)],
                    ),
                ),
                row=row,
                col=1,
             )

        fig.update_layout(
            template="plotly_white",
            hovermode="x unified",
            height=300 + 250 * total_rows,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom", y=1.02,
                xanchor="right", x=1,
            ),
            margin=dict(l=60, r=60, t=100, b=40),
        )
        fig.update_yaxes(title_text="Equity", row=1, col=1)
        for i in range(2, total_rows + 1):
            fig.update_yaxes(title_text="Value", row=i, col=1)
        fig.update_xaxes(title_text="Date", row=total_rows, col=1)

        # ------------------------------------------------------------------
        # 10. Output
        # ------------------------------------------------------------------
        metrics = {
            "gross_return":     gross_cumret,
            "net_return":       net_cumret,
            "sharpe":           _scalar("sharpe"),
            "volatility":       _scalar("volatility"),
            "max_drawdown":     _scalar("max_drawdown") if fo.get("max_drawdown") is not None else float(drawdown.min()),
            "btc_correlation":  _scalar("corr_rm"),
            "alpha":            _scalar("alpha"),
            "beta":             _scalar("beta"),
            "tstat_alpha":      _scalar("tstat_alpha"),
            "hit_ratio":        _scalar("hit_ratio"),
            "ic":               _scalar("ic"),
            "mean_turnover":    mean_turnover,
            "total_turnover":   total_turnover,
            "total_costs":      total_costs,
        }
        series = {
            "gross_return":     gross_return,
            "net_return":       net_return,
            "gross_equity":     gross_equity,
            "net_equity":       net_equity,
            "benchmark_equity": benchmark_equity,
            "drawdown":         drawdown,
            "weights":          weights,
            "executed_weight":  executed_weight,
            "lagged_weights":   lagged_weights,
            "turnover":         turnover_series,
            "costs":            costs,
        }

        return {"chart": fig, "metrics": metrics, "series": series}


class Portfolio0:

    def invoke(
        self,
        fo,
        weights: pd.DataFrame,
        returns: pd.DataFrame,
        benchmark: pd.Series,
        transaction_cost: float = 0.0005,
        annualization: int = 252,
    ):
        # -----------------------------------------
        # Align data
        # -----------------------------------------
        idx = (
            weights.index
            .intersection(returns.index)
            .intersection(benchmark.index)
        )

        weights   = weights.loc[idx]
        returns   = returns.loc[idx]
        benchmark = benchmark.loc[idx]

        # -----------------------------------------
        # Executed weights from fo
        # -----------------------------------------
        executed_weight = fo.get("executed_weight")
        if executed_weight is not None:
            executed_weight = executed_weight.loc[
                executed_weight.index.intersection(idx)
            ]
            executed_weight = executed_weight.reindex(idx).fillna(0)
        else:
            executed_weight = weights.shift(1).fillna(0)

        lagged_weights = weights.shift(1).fillna(0)

        # -----------------------------------------
        # Portfolio returns
        # -----------------------------------------
        gross_return = (executed_weight * returns).sum(axis=1)

        # -----------------------------------------
        # Turnover
        # -----------------------------------------
        turnover = weights.diff().abs().sum(axis=1)
        turnover.iloc[0] = 0

        # -----------------------------------------
        # Transaction cost
        # -----------------------------------------
        costs = turnover * transaction_cost

        # -----------------------------------------
        # Net return
        # -----------------------------------------
        net_return = gross_return - costs

        # -----------------------------------------
        # Equity curves
        # -----------------------------------------
        gross_equity     = (1 + gross_return).cumprod()
        net_equity       = (1 + net_return).cumprod()
        benchmark_equity = (1 + benchmark).cumprod()

        # -----------------------------------------
        # Drawdown
        # -----------------------------------------
        drawdown = net_equity / net_equity.cummax() - 1

        # -----------------------------------------
        # Risk stats — read from fo
        # -----------------------------------------
        sharpe       = fo.get("sharpe")
        volatility   = fo.get("volatility")
        max_drawdown = fo.get("max_drawdown") if fo.get("max_drawdown") is not None else drawdown.min()
        btc_corr     = fo.get("corr_rm")
        alpha        = fo.get("alpha")
        beta         = fo.get("beta")
        tstat_alpha  = fo.get("tstat_alpha")
        hit_ratio    = fo.get("hit_ratio")
        ic           = fo.get("ic")

        # -----------------------------------------
        # Gather plottable metric series from fo
        # -----------------------------------------
        metric_keys = [
            "sharpe", "alpha", "beta", "tstat_alpha",
            "hit_ratio", "ic", "volatility"
        ]

        plottable = {}  # key -> Series or DataFrame
        for k in metric_keys:
            v = fo.get(k)
            if v is None:
                continue
            if isinstance(v, pd.Series) and isinstance(v.index, pd.DatetimeIndex):
                plottable[k] = v
            elif isinstance(v, pd.DataFrame) and isinstance(v.index, pd.DatetimeIndex):
                plottable[k] = v

        # -----------------------------------------
        # Build subplots: row 1 = equity, rows 2+ = metrics
        # -----------------------------------------
        n_metric_rows = len(plottable)
        total_rows    = 1 + max(n_metric_rows, 1)

        fig = make_subplots(
            rows=total_rows,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.55] + [0.45 / max(n_metric_rows, 1)] * max(n_metric_rows, 1),
            vertical_spacing=0.08,
            subplot_titles=["Portfolio Performance"] + [
                k.replace("_", " ").title() for k in plottable.keys()
            ],
        )

        # --- Row 1: Equity curves ---
        fig.add_trace(
            go.Scatter(x=gross_equity.index, y=gross_equity,
                       name="Gross Return", mode="lines",
                       line=dict(color="#2E86AB", width=1.5)),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=net_equity.index, y=net_equity,
                       name="Net Return", mode="lines",
                       line=dict(color="#A23B72", width=1.5)),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=benchmark_equity.index, y=benchmark_equity,
                       name="Benchmark", mode="lines",
                       line=dict(color="#F18F01", width=1.5, dash="dash")),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=drawdown.index, y=drawdown,
                       name="Drawdown", fill="tozeroy",
                       fillcolor="rgba(231, 76, 60, 0.12)",
                       line=dict(color="rgba(231, 76, 60, 0.55)", width=1)),
            row=1, col=1,
        )

        # --- Rows 2+: Metric series ---
        colors = plotly.colors.qualitative.Plotly
        for i, (key, data) in enumerate(plottable.items(), start=2):
            if isinstance(data, pd.Series):
                fig.add_trace(
                    go.Scatter(
                        x=data.index, y=data,
                        name=key.replace("_", " ").title(),
                        mode="lines",
                        line=dict(width=1.2),
                    ),
                    row=i, col=1,
                )
            elif isinstance(data, pd.DataFrame):
                # Limit to top 5 symbols by final absolute value
                final_vals = data.iloc[-1].abs().sort_values(ascending=False)
                top_syms   = final_vals.head(5).index.tolist()
                sub_df     = data[top_syms]

                for j, sym in enumerate(sub_df.columns):
                    fig.add_trace(
                        go.Scatter(
                            x=sub_df.index, y=sub_df[sym],
                            name=f"{sym}",
                            mode="lines",
                            line=dict(width=1, color=colors[j % len(colors)]),
                            showlegend=True,
                            legendgroup=key,
                            legendgrouptitle_text=key.replace("_", " ").title(),
                        ),
                        row=i, col=1,
                    )

        # --- Layout ---
        fig.update_layout(
            template="plotly_white",
            hovermode="x unified",
            height=300 + 250 * total_rows,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom", y=1.02,
                xanchor="right", x=1,
            ),
            margin=dict(l=60, r=60, t=100, b=40),
        )

        fig.update_yaxes(title_text="Equity", row=1, col=1)
        for i in range(2, total_rows + 1):
            fig.update_yaxes(title_text="Value", row=i, col=1)
        fig.update_xaxes(title_text="Date", row=total_rows, col=1)

        # -----------------------------------------
        # Output
        # -----------------------------------------
        return {
            "chart": fig,
            "metrics": {
                "gross_return":     fo.get("ret"),
                "net_return":       fo.get("net_ret"),
                "sharpe":           sharpe,
                "volatility":       volatility,
                "max_drawdown":     max_drawdown,
                "btc_correlation":  btc_corr,
                "alpha":            alpha,
                "beta":             beta,
                "tstat_alpha":      tstat_alpha,
                "hit_ratio":        hit_ratio,
                "ic":               ic,
                "turnover":         fo.get("turnover"),
                "transaction_cost": costs,
            },
            "series": {
                "gross_return":     gross_return,
                "net_return":       net_return,
                "equity":           net_equity,
                "benchmark":        benchmark_equity,
                "drawdown":         drawdown,
                "weights":          weights,
                "executed_weight":  executed_weight,
                "lagged_weights":   lagged_weights,
            },
        }

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


# ======================================================
# END OF THE PIPELINE
# ======================================================