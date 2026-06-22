# ===============================================================
# e2ebacktest : backtest pipeline
# Date: 2026/06/22 
# Author : bugsbunnyla
# Comment : process the non production pipeline as if live engine
# ===============================================================
from qXengine.StrategyInit import StrategyInit
from qXengine.PortfolioConstruct import PortfolioConstruct
from core.system import QuantX

from config import DEFAULT_SYMBOLS, DEFAULT_CAPITAL


# =========================================================
# BACKTEST ENTRY FUNCTION (ONLY EXPOSED API)
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
    # ENGINE (same as production entry style)
    # =====================================================

    engine = QuantX(symbols=symbols)

    output = engine.run()

    # =====================================================
    # OUTPUT VIEW
    # =====================================================

    if view in ["horizontal", "both"]:
        print("\n[BACKTEST] HORIZONTAL VIEW\n")
        print(output)

    if view in ["vertical", "both"]:
        print("\n[BACKTEST] VERTICAL VIEW\n")
        print(output.T)

    print("\n[BACKTEST] SIGNAL SUMMARY\n")
    print(output[("decision", "signal")])

    # =====================================================
    # STRATEGY PIPELINE (BACKTEST CONTEXT MUST BE INFERRED)
    # =====================================================

    result = StrategyInit.run(interval="4y", run_option="backtest")

    dashboard = result["dashboard"]
    strategies = result["strategies"]

    dashboard.displayFrame()

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