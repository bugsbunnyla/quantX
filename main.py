
import argparse
import sys

from core.system import QuantX
from config import DEFAULT_SYMBOLS, DEFAULT_CAPITAL
from qXengine.StrategyInit import StrategyInit

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
        symbols = args.symbols

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
            symbols = [
                s.strip().upper()
                for s in user_symbols.split(",")
            ]

    symbols = None
    # -----------------------------------------------------
    # CAPITAL
    # -----------------------------------------------------

    if args.capital:
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
    # VIEW MODE
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

    return symbols, capital, view


# =========================================================
# MAIN EXECUTION
# =========================================================

def main():

    args = parse_args()

    symbols, capital, view = interactive_inputs(args)

    print("\n=================================================")
    print("🚀 QUANT XPERT X STARTING")
    print("=================================================\n")

    #print("Symbols:", symbols)
    print("Capital:", capital)
    print("View:", view)

    # =====================================================
    # RUN ENGINE
    # =====================================================

    engine = QuantX(symbols)

    output = engine.run()

    # =====================================================
    # HORIZONTAL VIEW
    # =====================================================

    if view in ["horizontal", "both"]:

        print("\n=================================================")
        print("📊 HORIZONTAL MULTIINDEX VIEW")
        print("=================================================\n")

        print(output)

    # =====================================================
    # VERTICAL VIEW
    # =====================================================

    if view in ["vertical", "both"]:

        print("\n=================================================")
        print("📊 VERTICAL STACKED VIEW")
        print("=================================================\n")

        print(output.T)

    # =====================================================
    # SIGNAL SUMMARY
    # =====================================================

    print("\n=================================================")
    print("🧠 SIGNAL SUMMARY")
    print("=================================================\n")

    print(
        output[
            ("decision", "signal")
        ]
    )

    print("\n=================================================")
    print("✅ ANALYSIS COMPLETE")
    print("=================================================\n")


    # =====================================================
    # 2. STRATEGY PIPELINE
    # =====================================================
    result = StrategyInit.run(interval="4y")

    dashboard = result["dashboard"]
    strategies = result["strategies"]

    # =====================================================
    # 3. DASHBOARD (SINGLE SOURCE OF TRUTH)
    # =====================================================
    dashboard.displayFrame()

    # =====================================================
    # 4. SUMMARY (MINIMAL OUTPUT ONLY)
    # =====================================================
    print("\n=================================================")
    print("📊 SYSTEM SUMMARY")
    print("=================================================\n")

    #print("Engine output:", getattr(engine_output, "shape", None))
    print("Strategies:", len(strategies))
    

    print("\n=================================================")
    print("✅ COMPLETE")
    print("=================================================\n")


# =========================================================
# ENTRYPOINT
# =========================================================
if __name__ == "__main__":
    main()