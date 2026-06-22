# ===============================================================
# Main class defines the QuantXpert Solution driver
# Date: 2026/06/22 
# Author : bugsbunnyla
# Comment : execute pipeline for processing Quant Xpert portfolio
# ===============================================================
"""
=========================================================
 QUANT X
=========================================================

HEADLESS QUANT RESEARCH + ANALYSIS ENGINE

---------------------------------------------------------
USAGE EXAMPLES
---------------------------------------------------------

1. DEFAULT RUN
---------------------------------------------------------
python main.py

Uses:
    symbols = config.DEFAULT_SYMBOLS
    capital = config.DEFAULT_CAPITAL
    view = both

---------------------------------------------------------

2. CUSTOM SYMBOLS
---------------------------------------------------------
python main.py --symbols NVDA TSLA AAPL

---------------------------------------------------------

3. CUSTOM CAPITAL
---------------------------------------------------------
python main.py --capital 5000

---------------------------------------------------------

4. CUSTOM VIEW MODE
---------------------------------------------------------
python main.py --view horizontal

OPTIONS:
    horizontal
    vertical
    both

---------------------------------------------------------

5. FULL OVERRIDE
---------------------------------------------------------
python main.py \
    --symbols NVDA TSLA AAPL \
    --capital 10000 \
    --view both

=========================================================
📊 MULTIINDEX OUTPUT STRUCTURE
=========================================================

LEVEL 1 → market
---------------------------------
symbol
price
return
volume

LEVEL 2 → alpha
---------------------------------
ts              = time-series momentum
xs              = cross-sectional alpha
pure            = purified alpha
beta            = market beta
residual        = regression residual

LEVEL 3 → risk
---------------------------------
volatility
sharpe
drawdown
cvar
decorrelation
wiggle

LEVEL 4 → transform
---------------------------------
rank
zscore
winsor
tanh
detrend

LEVEL 5 → portfolio
---------------------------------
weight
risk_parity
kelly
entropy

LEVEL 6 → market_structure
---------------------------------
regime
liq_adj_vol

LEVEL 7 → execution
---------------------------------
slippage
impact
turnover

LEVEL 8 → intel
---------------------------------
ic

LEVEL 9 → decision
---------------------------------
score
signal

=========================================================
 SIGNAL INTERPRETATION
=========================================================

BUY:
    Positive alpha/risk structure

SELL:
    Negative alpha/risk structure

HOLD:
    Neutral market condition

=========================================================
"""

import argparse
import sys

from core.system import QuantX
from config import DEFAULT_SYMBOLS, DEFAULT_CAPITAL
from qXengine.StrategyInit import StrategyInit
from qXengine.PortfolioConstruct import PortfolioConstruct


# =========================================================
# CLI ARGUMENTS
# =========================================================
def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--symbols",
        nargs="+",
        help="List of symbols"
    )

    parser.add_argument(
        "--capital",
        type=float,
        help="Trade capital"
    )

    parser.add_argument(
        "--view",
        type=str,
        choices=["horizontal", "vertical", "both"],
        help="Output display mode"
    )

    parser.add_argument(
        "--run_option",
        type=str,
        choices=["production", "backtest"],
        default="production",
        help="Execution mode (data + cache + API routing)"
    )

    return parser.parse_args()


# =========================================================
# INTERACTIVE INPUTS
# =========================================================
def interactive_inputs(args):

    print("\n=================================================")
    print(" QUANT XPERT X INPUT CONFIGURATION")
    print("=================================================\n")

    # -----------------------------------------------------
    # SYMBOLS
    # -----------------------------------------------------
    if args.symbols:
        symbols = [s.upper() for s in args.symbols]
    else:
        user_symbols = input(
            f"""
Enter symbols separated by commas
(default = {','.join(DEFAULT_SYMBOLS)})

Symbols:
"""
        ).strip()

        if user_symbols == "":
            symbols = DEFAULT_SYMBOLS
        else:
            symbols = [s.strip().upper() for s in user_symbols.split(",")]

    # -----------------------------------------------------
    # CAPITAL
    # -----------------------------------------------------
    if args.capital is not None:
        capital = args.capital
    else:
        user_capital = input(
            f"""
Enter trade capital
(default = {DEFAULT_CAPITAL})

Capital:
"""
        ).strip()

        if user_capital == "":
            capital = DEFAULT_CAPITAL
        else:
            capital = float(user_capital)

    # -----------------------------------------------------
    # VIEW
    # -----------------------------------------------------
    if args.view:
        view = args.view
    else:
        user_view = input(
            """
Choose output mode:
    horizontal
    vertical
    both

(default = both)

View:
"""
        ).strip().lower()

        if user_view == "":
            view = "both"
        else:
            view = user_view

    # safety
    if not symbols:
        symbols = DEFAULT_SYMBOLS

    return symbols, capital, view


# =========================================================
# MAIN EXECUTION
# =========================================================
def main():

    args = parse_args()

    symbols, capital, view = interactive_inputs(args)

    print("\n=================================================")
    print("[MAIN] QUANT XPERT X STARTING")
    print("=================================================\n")

    print("Symbols:", symbols)
    print("Capital:", capital)
    print("View:", view)
    print("Run Mode:", args.run_option)

    # =====================================================
    # ENGINE
    # =====================================================
    engine = QuantX(symbols)
    output = engine.run()

    # =====================================================
    # HORIZONTAL VIEW
    # =====================================================
    if view in ["horizontal", "both"]:

        print("\n=================================================")
        print("[MAIN] HORIZONTAL MULTIINDEX VIEW")
        print("=================================================\n")

        print(output)

    # =====================================================
    # VERTICAL VIEW
    # =====================================================
    if view in ["vertical", "both"]:

        print("\n=================================================")
        print("[MAIN] VERTICAL STACKED VIEW")
        print("=================================================\n")

        print(output.T)

    # =====================================================
    # SIGNAL SUMMARY
    # =====================================================
    print("\n=================================================")
    print("[MAIN] SIGNAL SUMMARY")
    print("=================================================\n")

    print(output[("decision", "signal")])

    print("\n=================================================")
    print("[MAIN] ANALYSIS COMPLETE")
    print("=================================================\n")

    # =====================================================
    # STRATEGY PIPELINE (NOW RUN_OPTION WIRED)
    # =====================================================
    result = StrategyInit.run(
        interval="4y",
        run_option=args.run_option   # IMPORTANT FIX
    )

    dashboard = result["dashboard"]
    strategies = result["strategies"]

    dashboard.displayFrame()

    # =====================================================
    # SUMMARY
    # =====================================================
    print("\n=================================================")
    print("[MAIN] SYSTEM SUMMARY")
    print("=================================================\n")

    print("Strategies:", len(strategies))

    pc = PortfolioConstruct(capital=capital)
    pc.invoke()

    print("\n=================================================")
    print("[MAIN] COMPLETE")
    print("=================================================\n")


# =========================================================
# ENTRYPOINT
# =========================================================
if __name__ == "__main__":
    main()

# =========================================================
# END OF MAIN
# =========================================================