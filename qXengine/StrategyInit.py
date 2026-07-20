# ===============================================================
# StrategyInit : Core initialized engine driver
# Date: 2026/06/22 
# Author : bugsbunnyla
# Comment : initiates pipeline strategy and dashboard
# ===============================================================

from config import DEFAULT_SYMBOLS
from .PickleDataManager import PickleDataManager
from .qxEngine import QuantXEngine
from .StrategyCharts import QXDashboard

import pandas as pd


# =====================================================
# DATA LOADER
# =====================================================

def strategy_build_universe(dm, symbols):

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

    return data


# =====================================================
# STRATEGY PIPELINE (CLEAN ORCHESTRATOR ONLY)
# =====================================================

class StrategyInit:

    @staticmethod
    def run(interval="4y", run_option=None, symbols=None):

        # =====================================================
        # USE SINGLE SOURCE OF TRUTH
        # =====================================================

        if symbols is None:
            symbols = DEFAULT_SYMBOLS

        # -------------------------------------
        # DATA LOAD (ENV-AWARE)
        # -------------------------------------

        dm = PickleDataManager(run_option)
       
        data = strategy_build_universe(dm, symbols)

        if not data:
            raise RuntimeError("[DATA] Universe empty")

        #print("[DATA] Assets:", list(data.keys()))

        # -------------------------------------
        # ENGINE EXECUTION
        # -------------------------------------

        engine = QuantXEngine()

        strategies = engine.qxStrategyList(
            data,
            interval=interval
        )

        print("[ENGINE] Strategies:", len(strategies))

        # -------------------------------------
        # FILTER VALID STRATEGIES ONLY
        # -------------------------------------

        valid_strategies = [
            s for s in strategies
            if getattr(s, "signals", None) is not None
        ]

        print("[ENGINE] Valid strategies:", len(valid_strategies))

        # -------------------------------------
        # DASHBOARD ONLY (NO RENDERING HERE)
        # -------------------------------------

        dashboard = QXDashboard.get()

        # -------------------------------------
        # RETURN PIPELINE STATE ONLY
        # -------------------------------------

        return {
            "data": data,
            "strategies": strategies,
            "valid_strategies": valid_strategies,
            "dashboard": dashboard
        }


# =====================================================
# STANDALONE TEST
# =====================================================

if __name__ == "__main__":

    result = StrategyInit.run(
        interval="4y",
        run_option="production"
    )

    print(f"\n[COMPLETE] {len(result['strategies'])} strategies")
    print(f"[VALID] {len(result['valid_strategies'])}")

    dashboard = result["dashboard"]

    print("\n========== DASHBOARD ==========")
    print(dashboard.keys())

    print("\n========== STRATEGIES ==========")
    print(len(result["strategies"]))

    # ONLY visualization layer now
    QXDashboard.displayFrame()

# ========================================================
# END OF STRATEGY INIT
# ========================================================