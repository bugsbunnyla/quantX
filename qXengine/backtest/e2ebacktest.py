# ===============================================================
# e2ebacktest : backtest pipeline
# Date: 2026/06/22 
# Author : bugsbunnyla
# Comment : process the non production pipeline as if live engine
# ===============================================================
import sys
import pandas as pd

from qXengine.StrategyInit import StrategyInit
from qXengine.PortfolioConstruct import PortfolioConstruct
from core.system import QuantX
from config import DEFAULT_SYMBOLS, DEFAULT_CAPITAL


# =========================================================
# GLOBAL OUTPUT FIX (CRITICAL)
# =========================================================

# Prevent pandas truncation ("...")
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 0)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.expand_frame_repr", False)


# Redirect ALL prints to file
LOG_FILE = open("backtest_run_output.txt", "w", encoding="utf-8")


class FileLogger:
    def __init__(self, file):
        self.file = file

    def write(self, msg):
        self.file.write(msg)

    def flush(self):
        self.file.flush()


# =========================================================
# APPLY GLOBAL REDIRECTION
# =========================================================
sys.stdout = FileLogger(LOG_FILE)
sys.stderr = FileLogger(LOG_FILE)


# =========================================================
# BACKTEST ENTRY FUNCTION
# =========================================================

def run_backtest(symbols=None, capital=None, view="both"):

    symbols = symbols or DEFAULT_SYMBOLS
    capital = capital or DEFAULT_CAPITAL

    print("\n=================================================")
    print("[BACKTEST] QUANT XPERT X STARTING")
    print("=================================================\n")

    print("Mode: backtest")
    print("Symbols:", symbols)
    print("Capital:", capital)
    print("View:", view)

    # =====================================================
    # ENGINE
    # =====================================================

    engine = QuantX(symbols=symbols)
    output = engine.run()

    # =====================================================
    # OUTPUT VIEW (FULL, NO TRUNCATION)
    # =====================================================

    if view in ["horizontal", "both"]:
        print("\n[BACKTEST] HORIZONTAL VIEW\n")
        print(output.to_string() if hasattr(output, "to_string") else output)

    if view in ["vertical", "both"]:
        print("\n[BACKTEST] VERTICAL VIEW\n")
        print(output.T.to_string() if hasattr(output, "T") else output.T)

    print("\n[BACKTEST] SIGNAL SUMMARY\n")

    signal = output[("decision", "signal")] if hasattr(output, "__getitem__") else None
    print(signal.to_string() if hasattr(signal, "to_string") else signal)

    # =====================================================
    # STRATEGY PIPELINE
    # =====================================================

    result = StrategyInit.run(interval="4y", run_option="backtest")

    dashboard = result["dashboard"]
    strategies = result["strategies"]

    # safe display (avoid console dependency)
    try:
        dashboard.displayFrame()
    except Exception as e:
        print(f"[WARN] dashboard display failed: {e}")

    # =====================================================
    # PORTFOLIO
    # =====================================================

    pc = PortfolioConstruct(capital=capital)
    portfolio_result = pc.invoke()

    # =====================================================
    # SUMMARY
    # =====================================================

    print("\n=================================================")
    print("[BACKTEST] SYSTEM SUMMARY")
    print("=================================================\n")

    print("Strategies:", len(strategies))

    print("\n[BACKTEST] COMPLETE\n")

    return {
        "output": output,
        "strategies": strategies,
        "portfolio": portfolio_result
    }


def main():
    run_backtest()


if __name__ == "__main__":
    main()

# ==============================================================
# END OF e2ebacktest
# ==============================================================