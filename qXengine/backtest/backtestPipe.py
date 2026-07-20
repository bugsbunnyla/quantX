import os
import pickle
import copy
import numpy as np
import pandas as pd
from scipy import stats

from ..qxEngine import QuantXEngine
from ..PickleDataManager import PickleDataManager
from pprint import pprint

def inspect_object(obj, indent=0):
    prefix = " " * indent

    if isinstance(obj, list):
        print(f"{prefix}List ({len(obj)} items)")
        for i, item in enumerate(obj):
            print(f"{prefix}[{i}]")
            inspect_object(item, indent + 4)
    elif isinstance(obj, dict):
        print(f"{prefix}Dict")
        pprint(obj)
    elif hasattr(obj, "__dict__"):
        print(f"{prefix}{obj.__class__.__name__}")
        for k, v in vars(obj).items():
            print(f"{prefix}{k}:")
            inspect_object(v, indent + 4)
    else:
        print(f"{prefix}{repr(obj)}")
        # train this data output for our required information

# =========================================================
# STORAGE ENGINE
# =========================================================
#class Storage:

#    def __init__(self, base_dir):
#        self.base_dir = base_dir
#        os.makedirs(base_dir, exist_ok=True)

#    def save(self, obj, path):
#        with open(path, "wb") as f:
#            pickle.dump(obj, f)

#    def load(self, path):
#        with open(path, "rb") as f:
#            return pickle.load(f)
import os
import pickle
from pathlib import Path


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

    # ---------------------------------------------------
    # Internal path resolver
    # ---------------------------------------------------
    def _resolve(self, path):
        path = Path(path)
        # Absolute path -> leave unchanged
        if path.is_absolute():
            return path
        # Already starts with base_dir (runs/...)
        if len(path.parts) > 0 and path.parts[0] == self.base.name:
            return path
        # Otherwise prepend base_dir
        return self.base / path

    # ---------------------------------------------------
    # Generic pickle
    # ---------------------------------------------------
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
    # ---------------------------------------------------
    # Dataset helpers
    # ---------------------------------------------------
    def save_dataset(self, name, data):
        return self.save(data, self.datasets / f"{name}.pkl")
    def load_dataset(self, name):
        return self.load(self.datasets / f"{name}.pkl")

    # ---------------------------------------------------
    # Run management
    # ---------------------------------------------------
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

    # ---------------------------------------------------
    # Run artifact helpers
    # ---------------------------------------------------
    def save_run_file(self, run_dir, name, obj):
        return self.save(obj, Path(run_dir) / f"{name}.pkl")

    def load_run_file(self, run_dir, name):
        return self.load(Path(run_dir) / f"{name}.pkl")

# =========================================================
# SPLIT ENGINE
# =========================================================
class SplitEngine:

    #def split(self, data):
    #    idx = int(len(data) * 0.7)
    #    return data.iloc[:idx], data.iloc[idx:]


    def split(self, data):
        train = copy.deepcopy(data)
        val = copy.deepcopy(data)
        return train, val

#
#
#
import os
import pickle
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# =========================================================
# ML TRAIN ENGINE
# =========================================================

import os
import joblib
import numpy as np
import pandas as pd


from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
class MLTrainEngine:


    def __init__(self):

        self.feature_schema = None

        self.model = None

        self.formula_outputs = None

        self.strategy_results = None

        self.raw_data = None

        self.params = {}



    # =====================================================
    # MAIN TRAIN ENTRY
    # RECEIVES PACKAGE FROM AgenticPipeline
    # =====================================================

    def train(
        self,
        train_package
    ):


        print(
            "[ML] Training from package"
        )


        if not isinstance(
            train_package,
            dict
        ):

            raise TypeError(
                "train() requires train_package dict"
            )



        required = [
            "X",
            "y"
        ]


        for key in required:

            if key not in train_package:

                raise KeyError(
                    f"Missing train_package key: {key}"
                )



        X = train_package["X"]

        y = train_package["y"]



        self.strategy_results = (
            train_package.get(
                "results"
            )
        )


        self.formula_outputs = (
            train_package.get(
                "formula_outputs"
            )
        )


        self.params = (
            train_package.get(
                "params",
                {}
            )
        )



        print(
            "[ML] Dataset:",
            X.shape,
            y.shape
        )



        trained_model, metrics = self.fit(

            X,

            y,

            self.params

        )



        self.model = trained_model


        self.feature_schema = list(
            X.columns
        )



        result = {


            # sklearn wrapper
            "model":
                trained_model,


            # feature matrix
            "X":
                X,


            # target
            "y":
                y,


            # schema
            "features":
                list(X.columns),


            # metrics
            "metrics":
                metrics,


            # preserve QuantX context
            "results":
                train_package.get(
                    "results"
                ),


            "formula_outputs":
                train_package.get(
                    "formula_outputs"
                ),


            "params":
                self.params

        }



        joblib.dump(
            result,
            "train.pkl"
        )


        return result




    # =====================================================
    # BUILD FEATURE MATRIX
    # USED BY AgenticPipeline.prepare()
    # =====================================================

    def build_feature_matrix(
        self,
        strategy_results,
        formula_outputs,
        raw_data,
        params=None
    ):


        X, y, schema = self.build_features(

            strategy_results,

            formula_outputs,

            raw_data,

            params or {}

        )


        return X, y




    # =====================================================
    # FEATURE BUILDER
    # =====================================================

    def build_features(
        self,
        strategy_results,
        formula_outputs,
        raw_data,
        params=None
    ):


        frames = []

        ret_series = None



        # -------------------------------------------------
        # FORMULA OUTPUTS
        # -------------------------------------------------

        if formula_outputs is not None:


            if isinstance(
                formula_outputs,
                pd.DataFrame
            ):

                formula_outputs = [
                    formula_outputs
                ]



            for fo in formula_outputs:


                if not isinstance(
                    fo,
                    pd.DataFrame
                ):

                    continue



                if not isinstance(
                    fo.index,
                    pd.MultiIndex
                ):

                    continue



                if (
                    ret_series is None
                    and
                    ("market","ret") in fo.index
                ):


                    ret_series = (
                        fo.loc[
                            ("market","ret")
                        ]
                        .copy()
                    )



                temp = (

                    fo
                    .reset_index()
                    .melt(

                        id_vars=[
                            "category",
                            "metric"
                        ],

                        var_name="symbol",

                        value_name="value"

                    )

                )



                temp["feature"] = (

                    temp["category"]
                    .astype(str)

                    +

                    "_"

                    +

                    temp["metric"]
                    .astype(str)

                )



                formula_df = (

                    temp
                    .pivot_table(

                        index="symbol",

                        columns="feature",

                        values="value",

                        aggfunc="last"

                    )
                    .reset_index()

                )



                formula_df = formula_df.drop(

                    columns=[
                        "market_ret"
                    ],

                    errors="ignore"

                )



                frames.append(
                    formula_df
                )



        # -------------------------------------------------
        # STRATEGY RESULTS
        # -------------------------------------------------

        for result in strategy_results or []:


            if getattr(
                result,
                "metrics",
                None
            ):


                rows = []


                for symbol, values in result.metrics.items():


                    if isinstance(
                        values,
                        dict
                    ):


                        row = {
                            "symbol": symbol
                        }


                        row.update(
                            values
                        )


                        rows.append(
                            row
                        )


                if rows:

                    frames.append(
                        pd.DataFrame(rows)
                    )



            if getattr(
                result,
                "signals",
                None
            ):


                signal_df = pd.DataFrame(

                    [

                        {
                            "symbol": symbol,

                            "signal": signal

                        }

                        for symbol, signal
                        in result.signals.items()

                    ]

                )


                frames.append(
                    signal_df
                )



        if not frames:

            raise ValueError(
                "No features generated"
            )



        # -------------------------------------------------
        # MERGE
        # -------------------------------------------------

        feature_df = frames[0]


        for frame in frames[1:]:


            feature_df = feature_df.merge(

                frame,

                on="symbol",

                how="outer"

            )



        # -------------------------------------------------
        # TARGET
        # -------------------------------------------------

        if ret_series is None:

            raise ValueError(
                "Missing ('market','ret') target"
            )



        y_df = (

            ret_series

            .rename_axis(
                "symbol"
            )

            .reset_index(
                name="y"
            )

        )



        feature_df = feature_df.merge(

            y_df,

            on="symbol",

            how="inner"

        )



        feature_df = feature_df.dropna(
            subset=["y"]
        )



        y = feature_df["y"]



        X = feature_df.drop(

            columns=[
                "symbol",
                "y"
            ],

            errors="ignore"

        )



        X = self.encode_features(
            X
        )



        X = X.replace(

            [np.inf, -np.inf],

            np.nan

        )


        X = X.fillna(0)



        return (

            X,

            y,

            list(X.columns)

        )




    # =====================================================
    # MODEL FIT
    # =====================================================

    def fit(
        self,
        X,
        y,
        params=None
    ):


        if params is None:

            params = {}



        X_train, X_test, y_train, y_test = train_test_split(

            X,

            y,

            test_size=0.2,

            random_state=42

        )



        model = RandomForestRegressor(

            n_estimators=params.get(
                "trees",
                500
            ),

            max_depth=params.get(
                "depth",
                12
            ),

            random_state=42

        )



        model.fit(

            X_train,

            y_train

        )



        pred = model.predict(
            X_test
        )



        metrics = {


            "rmse":

                float(
                    np.sqrt(
                        mean_squared_error(
                            y_test,
                            pred
                        )
                    )
                ),



            "r2":

                float(
                    r2_score(
                        y_test,
                        pred
                    )
                ),



            "samples":

                len(X),



            "features":

                len(X.columns)

        }



        wrapper = {


            "model":

                model,


            "features":

                list(X.columns),


            "metrics":

                metrics

        }



        return wrapper, metrics




    # =====================================================
    # ENCODER
    # =====================================================

    def encode_features(
        self,
        df
    ):


        result = df.copy()



        for col in result.columns:


            if pd.api.types.is_numeric_dtype(
                result[col]
            ):


                result[col] = (
                    result[col]
                    .fillna(0)
                )


            elif pd.api.types.is_datetime64_any_dtype(
                result[col]
            ):


                result[col] = (
                    result[col]
                    .astype("int64")
                )


            else:


                result[col] = (

                    result[col]
                    .fillna("UNKNOWN")
                    .astype("category")
                    .cat.codes

                )



        return result

# =========================================================
# PROCESS ENGINE - STRATEGY
# =========================================================
class ProcessEngine:


    def __init__(self):

        self.ml = MLTrainEngine()



    # =====================================================
    # TRAIN MODEL
    # RECEIVES PACKAGE FROM AgenticPipeline
    # =====================================================

    def train(
        self,
        train_package,
        params=None
    ):


        if params is None:
            params = {}



        if not isinstance(
            train_package,
            dict
        ):

            raise TypeError(
                "ProcessEngine.train expects package dict"
            )



        print(
            "[PROCESS] Training package received"
        )


        print(
            "[PROCESS] Keys:",
            train_package.keys()
        )



        # -------------------------------------------------
        # Validate package contract
        # -------------------------------------------------

        required = [
            "X",
            "y",
            "results",
            "formula_outputs"
        ]


        for key in required:


            if key not in train_package:


                raise KeyError(
                    f"Missing train package key: {key}"
                )



        # -------------------------------------------------
        # ML parameters
        # -------------------------------------------------

        ml_params = {


            "trees":

                params.get(
                    "trees",
                    500
                ),



            "depth":

                params.get(
                    "depth",
                    12
                ),



            "horizon":

                params.get(
                    "horizon",
                    21
                ),



            "model":

                params.get(
                    "model",
                    "random_forest"
                )

        }



        print(
            "[ML] Parameters:",
            ml_params
        )



        # -------------------------------------------------
        # attach parameters
        # -------------------------------------------------

        train_package["params"] = ml_params



        # -------------------------------------------------
        # Train ML model
        # -------------------------------------------------

        result = self.ml.train(

            train_package

        )



        print(
            "[ML] Training complete"
        )



        # -------------------------------------------------
        # Merge ML result back
        # -------------------------------------------------

        train_package["model"] = (
            result["model"]
        )


        train_package["metrics"] = (

            result.get(
                "metrics",
                {}
            )

        )



        train_package["features"] = (

            result.get(
                "features",
                train_package.get(
                    "features",
                    []
                )
            )

        )



        return train_package

# =========================================================
# BACKTEST ENGINE
# =========================================================
class BacktestEngine:


    def __init__(self):

        self.last_features = None
        self.last_signal = None



    # =====================================================
    # BUILD FEATURES FOR LIVE SIGNAL
    # OPTIONAL FUTURE USE
    # QuantX PREPARE CURRENTLY OWNS THIS
    # =====================================================

    def build_features_for_signal(
        self,
        raw_data
    ):

        rows = []


        for symbol, df in raw_data.items():


            if not isinstance(
                df,
                pd.DataFrame
            ):
                continue


            if len(df) < 20:
                continue



            row = {
                "symbol": symbol
            }



            close = df["close"]

            volume = df["volume"]



            ret = (

                df["ret"]

                if "ret" in df.columns

                else close.pct_change()

            )



            row["market_price"] = close.iloc[-1]

            row["market_volume"] = volume.iloc[-1]


            row["market_structure_liq_adj_vol"] = (

                volume
                .rolling(20)
                .mean()
                .iloc[-1]

            )



            volatility = (

                ret
                .rolling(20)
                .std()
                .iloc[-1]

            )


            row["risk_volatility"] = volatility


            row["risk_sharpe"] = (

                ret.mean()
                /
                (ret.std()+1e-9)

            )



            row["risk_drawdown"] = (

                close
                /
                close.cummax()

                -
                1

            ).iloc[-1]



            row["alpha_pure"] = (

                ret
                .rolling(5)
                .mean()
                .iloc[-1]

            )



            row["alpha_ts"] = (

                close.iloc[-1]
                -
                close
                .rolling(20)
                .mean()
                .iloc[-1]

            )



            row["alpha_beta"] = (

                ret
                .rolling(20)
                .mean()
                /
                (
                    ret
                    .rolling(20)
                    .std()
                    +
                    1e-9
                )

            ).iloc[-1]



            row["alpha_residual"] = (

                ret.iloc[-1]
                -
                ret
                .rolling(20)
                .mean()
                .iloc[-1]

            )


            row["alpha_xs"] = (

                close
                .pct_change(5)
                .iloc[-1]

            )



            mean20 = (

                close
                .rolling(20)
                .mean()
                .iloc[-1]

            )


            std20 = (

                close
                .rolling(20)
                .std()
                .iloc[-1]

            )


            row["transform_zscore"] = (

                close.iloc[-1]-mean20

            )/(std20+1e-9)



            row["transform_rank"] = (

                close
                .rolling(20)
                .rank()
                .iloc[-1]

            )



            row["transform_winsor"] = np.clip(

                ret.iloc[-1],

                -3*ret.std(),

                3*ret.std()

            )



            row["transform_tanh"] = np.tanh(
                ret.iloc[-1]
            )



            row["market_symbol"] = symbol



            row["transform_detrend"] = (

                close.iloc[-1]

                -
                close
                .rolling(20)
                .mean()
                .iloc[-1]

            )



            defaults = [

                "basic_corr",
                "basic_hit_ratio",
                "basic_r_squared",
                "basic_tstat",

                "decision_score",
                "decision_signal",

                "execution_impact",
                "execution_slippage",
                "execution_turnover",

                "intel_ic",

                "market_structure_regime",

                "portfolio_entropy",
                "portfolio_inv_vol",
                "portfolio_kelly",
                "portfolio_mvo",
                "portfolio_risk_parity",
                "portfolio_weight",

                "risk_cvar"

            ]



            for c in defaults:

                row[c] = 0



            rows.append(row)



        df = pd.DataFrame(rows)



        df = (

            df
            .replace(
                [np.inf,-np.inf],
                np.nan
            )
            .fillna(0)

        )



        self.last_features = df.copy()


        return df




    # =====================================================
    # ENCODER
    # =====================================================

    def encode_features(
        self,
        df
    ):


        result=df.copy()



        for col in result.columns:


            if pd.api.types.is_numeric_dtype(
                result[col]
            ):


                result[col]=(
                    result[col]
                    .fillna(0)
                )


            else:


                result[col]=(

                    result[col]
                    .fillna("UNKNOWN")
                    .astype("category")
                    .cat.codes

                )


        return result




    # =====================================================
    # GENERATE SIGNAL
    # TRAIN MODEL -> TRAIN OR VAL DATA
    # =====================================================

    def signal(
        self,
        model_package,
        dataset
    ):


        if "model" not in model_package:

            raise KeyError(
                "model_package missing model"
            )


        model = (

            model_package["model"]["model"]

        )



        if "X" not in dataset:

            raise KeyError(
                "dataset missing X"
            )



        X = dataset["X"].copy()



        required_features = (

            model_package["model"]["features"]

        )



        # Align validation features
        for col in required_features:


            if col not in X.columns:

                X[col]=0



        X = X[
            required_features
        ]



        prediction = model.predict(
            X
        )



        prediction=np.asarray(
            prediction,
            dtype=float
        )



        self.last_signal = prediction



        return prediction




    # =====================================================
    # EVALUATION
    # =====================================================

    def evaluate(
        self,
        package,
        signal
    ):


        if "y" not in package:

            raise KeyError(
                "Package missing y"
            )



        y=np.asarray(
            package["y"],
            dtype=float
        ).ravel()



        signal=np.asarray(
            signal,
            dtype=float
        ).ravel()



        if len(signal)!=len(y):

            raise ValueError(

                f"Signal length {len(signal)} "
                f"!= target length {len(y)}"

            )



        pnl = signal*y



        hit = np.mean(

            np.sign(signal)

            ==
            np.sign(y)

        )



        sharpe = (

            np.mean(pnl)

            /

            (
                np.std(pnl)
                +
                1e-9
            )

        )



        var=np.percentile(
            pnl,
            5
        )


        tail=pnl[pnl<=var]


        cvar=(

            tail.mean()

            if len(tail)>0

            else 0

        )



        if np.std(signal)==0 or np.std(y)==0:

            corr=0

        else:

            corr=np.corrcoef(
                signal,
                y
            )[0,1]



        slope, intercept, r, _, stderr = (

            stats.linregress(
                signal,
                y
            )

        )



        r2=r*r



        t_beta=(

            slope /
            (stderr+1e-9)

        )



        score=(

            0.4*sharpe

            +
            0.2*corr

            +
            0.2*hit

            +
            0.2*r2

        )



        return {


            "hit":float(hit),

            "sharpe":float(sharpe),

            "cvar":float(cvar),

            "corr":float(corr),

            "r2":float(r2),

            "t_beta":float(t_beta),

            "score":float(score),


            "signal":signal,

            "pnl":pnl

        }

    def encode_features(self, df):
        result = df.copy()

        for col in result.columns:

          if pd.api.types.is_numeric_dtype( result[col] ):
            result[col] = (
                result[col]
                .fillna(0)
            )
          else:
            result[col] = (
                result[col]
                .fillna("UNKNOWN")
                .astype("category")
                .cat.codes
            )


        return result

# =========================================================
# REVIEW ENGINE
# =========================================================
class ReviewEngine:

    def compare(self, current, previous):

        if previous is None:
            return {
                "decision": "BASELINE",
                "score": current["score"]
            }

        gap = current["score"] - previous["score"]

        return {
            "decision": "CONTINUE" if gap > -0.05 else "STOP",
            "gap": gap,
            "current": current["score"],
            "previous": previous["score"]
        }


# =========================================================
# EVALUATOR
# =========================================================
class Evaluator:

    def decide(self, review):

        return "CONTINUE" if review["decision"] != "STOP" else "STOP"


# =========================================================
# PIPELINE ENGINE
# =========================================================
class AgenticPipeline:


    def __init__(
        self,
        data,
        base_dir="runs"
    ):

        self.data = data
        self.base_dir = base_dir


        self.storage = Storage(
            base_dir
        )


        self.split_engine = SplitEngine()


        self.strategy = ProcessEngine()


        self.backtest = BacktestEngine()


        self.review_engine = ReviewEngine()


        self.evaluator = Evaluator()


        self.iteration = 0



    # =====================================================
    # FEATURE PREPARATION
    # QUANTX OWNER
    # SAME LOGIC TRAIN + VALIDATION
    # =====================================================

    def prepare(
        self,
        data,
        params=None
    ):

        if params is None:
            params = {}


        engine = QuantXEngine()


        strategy_names = params.get(
            "strategies",
            [
                "AlphaStrategy"
            ]
        )


        strategies = engine.qxStrategySelect(
            strategy_names,
            data,
            interval=params.get(
                "interval",
                "4y"
            )
        )


        print(
            "[ENGINE] Strategies:",
            len(strategies)
        )



        formula_outputs = []


        for s in engine.strategy:


            if hasattr(
                s,
                "formulaOutput"
            ):


                formula_outputs.append(

                    s
                    .formulaOutput
                    .assemble()

                )



        print(
            "[ML] Formula outputs:",
            len(formula_outputs)
        )



        if len(formula_outputs) == 0:

            raise ValueError(
                "No formula outputs generated"
            )



        ml = MLTrainEngine()



        X, y = ml.build_feature_matrix(

            strategy_results=
                engine.results,

            formula_outputs=
                formula_outputs,

            raw_data=
                data

        )



        package = {


            # -------------------------
            # ML INPUT
            # -------------------------

            "X":
                X,


            "y":
                y,


            "features":
                list(X.columns),



            # -------------------------
            # QUANTX OUTPUTS
            # -------------------------

            "engine":
                engine,


            "strategies":
                engine.strategy,


            "results":
                engine.results,


            "formula_outputs":
                formula_outputs


        }


        return package




    # =====================================================
    # AGENT LOOP
    # =====================================================

    def run_agent(
        self,
        params,
        max_iters=5
    ):


        train, val = (

            self.split_engine
            .split(
                self.data
            )

        )



        print(
            "[DEBUG] train symbols:",
            train.keys()
        )


        print(
            "[DEBUG] val symbols:",
            val.keys()
        )



        self.storage.save(
            train,
            f"{self.base_dir}/1a_train.pkl"
        )


        self.storage.save(
            val,
            f"{self.base_dir}/1b_val.pkl"
        )



        previous_val = None



        for i in range(max_iters):


            self.iteration += 1



            print(
                "\n=============================="
            )


            print(
                f"AGENT ITERATION {self.iteration}"
            )


            print(
                "=============================="
            )



            # =================================================
            # 1. BUILD TRAIN PACKAGE
            # =================================================

            raw_train_package = self.prepare(
                train,
                params
            )

            train_package = self.strategy.train(
              raw_train_package,
              params)

            print(
                "[AGENT] Train package:"
            )


            print(
                train_package.keys()
            )



            # =================================================
            # 2. TRAIN ML MODEL
            # =================================================
            #
            # NEXT CONNECTION POINT:
            #
            # train_package =
            # MLTrainEngine.train(train_package)
            #
            # or
            #
            # ProcessEngine.train(train_package)
            #
            # depending on final ownership decision.
            #
            # =================================================



            model_package = train_package



            # =================================================
            # 3. VALIDATION PACKAGE
            # =================================================


            val_package = self.prepare(
                val,
                params
            )



            print(
                "[DEBUG] val keys:",
                val_package.keys()
            )



            # =================================================
            # 4. SAVE PREPARED DATA
            # =================================================


            run_path = (

                f"{self.base_dir}/"
                f"run_{self.iteration:06d}"

            )


            os.makedirs(
                run_path,
                exist_ok=True
            )



            self.storage.save(
                train_package,
                f"{run_path}/train_package.pkl"
            )


            self.storage.save(
                val_package,
                f"{run_path}/val_package.pkl"
            )



            print(
                """
PACKAGE PREPARATION COMPLETE

TRAIN:
{}

VAL:
{}
""".format(
                    train_package.keys(),
                    val_package.keys()
                )
            )

            # =================================================
            # 5. TRAIN BACKTEST SIGNAL
            # =================================================


            print(
                "[BACKTEST] Generating TRAIN signal"
            )


            train_signal = (

                self.backtest
                .signal(
                    train_package,
                    train_package
                )

            )



            print(
                "[BACKTEST] TRAIN signal complete"
            )



            self.storage.save(
                train_signal,
                f"{run_path}/train_signal.pkl"
            )



            # =================================================
            # 6. TRAIN EVALUATION
            # =================================================


            train_metrics = (

                self.backtest
                .evaluate(
                    train_package,
                    train_signal
                )

            )



            print(
                """
==============================
TRAIN METRICS
==============================

{}
""".format(
                    train_metrics
                )
            )



            self.storage.save(
                train_metrics,
                f"{run_path}/train_metrics.pkl"
            )



            # =================================================
            # 7. VALIDATION BACKTEST SIGNAL
            # =================================================


            print(
                "[BACKTEST] Generating VALIDATION signal"
            )



            val_signal = (

                self.backtest
                .signal(
                    train_package,
                    val_package
                )

            )



            print(
                "[BACKTEST] VALIDATION signal complete"
            )



            self.storage.save(
                val_signal,
                f"{run_path}/val_signal.pkl"
            )



            # =================================================
            # 8. VALIDATION EVALUATION
            # =================================================


            val_metrics = (

                self.backtest
                .evaluate(
                    val_package,
                    val_signal
                )

            )



            print(
                """
==============================
VALIDATION METRICS
==============================

{}
""".format(
                    val_metrics
                )
            )



            self.storage.save(
                val_metrics,
                f"{run_path}/val_metrics.pkl"
            )



            # =================================================
            # 9. RESEARCH COMPARISON
            # =================================================


            research = {


                "iteration":
                    self.iteration,


                "train":
                    train_metrics,


                "validation":
                    val_metrics,


                "model":
                    train_package.get(
                        "metrics",
                        {}
                    ),


                "features":
                    train_package.get(
                        "features",
                        []
                    )

            }



            print(
                """
==============================
RESEARCH REPORT
TRAIN VS VALIDATION
==============================

TRAIN:
{}

VALIDATION:
{}

MODEL:
{}

""".format(
                    train_metrics,
                    val_metrics,
                    train_package.get(
                        "metrics",
                        {}
                    )
                )
            )



            self.storage.save(
                research,
                f"{run_path}/research.pkl"
            )



            # =================================================
            # 10. REVIEW ENGINE
            # =================================================


            review = (

                self.review_engine
                .compare(
                    val_metrics,
                    previous_val
                )

            )



            print(
                """
==============================
REVIEW
==============================

{}
""".format(
                    review
                )
            )



            self.storage.save(
                review,
                f"{run_path}/review.pkl"
            )



            # =================================================
            # 11. AGENT DECISION
            # =================================================


            decision = (

                self.evaluator
                .decide(
                    review
                )

            )



            print(
                """
==============================
AGENT DECISION
==============================

{}

==============================
"""
.format(
                    decision
                )
            )



            self.storage.save(
                {
                    "decision":
                        decision
                },
                f"{run_path}/eval.pkl"
            )



            # =================================================
            # 12. ITERATION CONTROL
            # =================================================


            if decision == "STOP":

                print(
                    "[AGENT] Stopping"
                )

                break



            previous_val = val_metrics

def build_data():

    #read pkl files to create data
    symbols = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","DOGEUSDT",]
    dm = PickleDataManager("backtest")

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
       raise RuntimeError("[DATA] Universe empty")
       print("[DATA] Assets:", list(data.keys()))
    return data



if __name__ == "__main__":

    data = build_data()

    pipeline = AgenticPipeline(
        data=data,
        base_dir="runs"
    )

    params = {
        "reg": 1e-5
    }

    pipeline.run_agent(params=params, max_iters=5)

