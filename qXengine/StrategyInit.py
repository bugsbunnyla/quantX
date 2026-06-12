from .PickleDataManager import PickleDataManager
from .qxEngine import QuantXEngine
from .StrategyCharts import QXDashboard

import pandas as pd


# =====================================================
# SYMBOL UNIVERSE
# =====================================================
SYMBOLS = [
    "BTCUSDT.pkl",
    "ETHUSDT.pkl",
    "SOLUSDT.pkl",
    "BNBUSDT.pkl",
    "XRPUSDT.pkl",
    "DOGEUSDT.pkl",
    "SPY.pkl",
    "QQQ.pkl",
    "IWM.pkl",
    "TLT.pkl",
    "GLD.pkl"
]


# =====================================================
# DATA LOADER
# =====================================================
def strategy_build_universe(dm):

    data = {}

    for file_name in SYMBOLS:

        symbol = file_name.replace(".pkl", "")

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
    def run(interval="4y"):

        # -------------------------------------
        # DATA LOAD
        # -------------------------------------
        dm = PickleDataManager()
        data = strategy_build_universe(dm)

        if not data:
            raise RuntimeError("[DATA] Universe empty")

        print("[DATA] Assets:", list(data.keys()))

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

        # Feed engine outputs into dashboard state
        #dashboard["strategies"] = valid_strategies

        #print("[DASHBOARD] Updated with strategies")

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

    result = StrategyInit.run(interval="4y")

    print(f"\n[COMPLETE] {len(result['strategies'])} strategies")
    print(f"[VALID] {len(result['valid_strategies'])}")

    dashboard = result["dashboard"]

    print("\n========== DASHBOARD ==========")
    print(dashboard.keys())

    print("\n========== STRATEGIES ==========")
    print(len(dashboard.get("strategies", [])))

    # ONLY visualization layer now
    from .StrategyCharts import QXDashboard
    QXDashboard.displayFrame()