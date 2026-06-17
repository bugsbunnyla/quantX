import numpy as np
import pandas as pd
from qXengine.StrategyInit import StrategyInit


class qxEngineBackTest:

    def __init__(self):

        self.strategy_results = []
        self.dashboard = None

        self.decision = None
        self.metrics = None
        self.valid_metrics = None
        self.outcome = None

        self.portfolio_weights = None


    # =========================================================
    # 1. DECISION ENGINE
    # =========================================================
    def aggregate_decision(self, strategy_results, benchmark="benchmark"):

        scores = []

        for r in strategy_results:

            df = getattr(r, "chartdata", None)

            if df is None or df.empty:
                continue

            df = df.dropna()

            if benchmark not in df.columns:
                continue

            last = df.iloc[-1]

            strat_cols = [
                c for c in df.columns
                if c not in ["date", benchmark]
            ]

            if not strat_cols:
                continue

            strat_return = last[strat_cols].mean()
            bench_return = last[benchmark]

            excess = strat_return - bench_return

            volatility = df[strat_cols].pct_change().std().mean()

            sharpe_proxy = excess / (volatility + 1e-8)

            scores.append(sharpe_proxy)

        if len(scores) == 0:
            return "HOLD"

        avg_score = np.mean(scores)

        if avg_score > 0.25:
            return "BUY"
        elif avg_score < -0.25:
            return "SELL"

        return "HOLD"


    # =========================================================
    # 2. PORTFOLIO CONSTRUCTION
    # =========================================================
    def construct_portfolio(self, strategy_results):

        raw_scores = []
        names = []

        for r in strategy_results:

            df = getattr(r, "chartdata", None)

            if df is None or df.empty:
                continue

            df = df.dropna()

            cols = [c for c in df.columns if c != "date"]

            if not cols:
                continue

            returns = df[cols].pct_change().dropna()

            if returns.empty:
                continue

            total_return = (1 + returns).prod().mean()
            volatility = returns.std().mean()

            sharpe = total_return / (volatility + 1e-8)

            raw_scores.append(sharpe)
            names.append(r.name)

        raw_scores = np.array(raw_scores)

        if len(raw_scores) == 0 or raw_scores.sum() == 0:
            return {}

        weights = raw_scores / raw_scores.sum()

        return dict(zip(names, weights))


    # =========================================================
    # 3. PERFORMANCE EVALUATION
    # =========================================================
    def evaluate_portfolio(self, strategy_results):

        all_metrics = {}

        for r in strategy_results:

            df = getattr(r, "chartdata", None)

            if df is None or df.empty:
                continue

            df = df.dropna()

            cols = [c for c in df.columns if c != "date"]

            returns = df[cols].pct_change().dropna()

            if returns.empty:
                continue

            total_return = (1 + returns).prod().mean()
            vol = returns.std().mean()

            sharpe = total_return / (vol + 1e-8)

            all_metrics[r.name] = {

                "return": float(total_return),
                "volatility": float(vol),
                "sharpe": float(sharpe)
            }

        return all_metrics


    # =========================================================
    # 4. META ENGINE
    # =========================================================
    def run_meta_engine(self, strategy_results):

        decision = self.aggregate_decision(strategy_results)
        portfolio_weights = self.construct_portfolio(strategy_results)
        metrics = self.evaluate_portfolio(strategy_results)

        return {
            "decision": decision,
            "portfolio": portfolio_weights,
            "metrics": metrics
        }


    # =========================================================
    # 5. SETUP
    # =========================================================
    def run_setup(self):

        result = StrategyInit.run(interval="4y")

        self.dashboard = result["dashboard"]
        self.strategy_results = result["strategies"]

        print("\n======================================")
        print("[SYSTEM] BACKTEST ENGINE READY")
        print("======================================\n")

        print("Strategies:", len(self.strategy_results))


    # =========================================================
    # 6. PIPELINE EXECUTION
    # =========================================================
    def run_backtest_pipeline(self, strategy_results, benchmark="benchmark"):

        self.decision = self.aggregate_decision(
            strategy_results,
            benchmark=benchmark
        )

        self.portfolio_weights = self.construct_portfolio(
            strategy_results
        )

        self.metrics = self.evaluate_portfolio(
            strategy_results
        )

        self.valid_metrics = [
            m for m in self.metrics.values()
        ]

        if self.valid_metrics:

            avg_sharpe = np.mean([m["sharpe"] for m in self.valid_metrics])
            avg_return = np.mean([m["return"] for m in self.valid_metrics])
            avg_vol = np.mean([m["volatility"] for m in self.valid_metrics])

        else:
            avg_sharpe = 0
            avg_return = 0
            avg_vol = 0

        self.outcome = {

            "decision": self.decision,

            "portfolio_weights": self.portfolio_weights,

            "summary_metrics": {
                "avg_return": float(avg_return),
                "avg_volatility": float(avg_vol),
                "avg_sharpe": float(avg_sharpe),
                "num_strategies": len(strategy_results)
            },

            "strategy_metrics": self.metrics
        }


    # =========================================================
    # 7. MAIN RUN
    # =========================================================
    def run(self):

        self.run_setup()

        self.run_backtest_pipeline(self.strategy_results)