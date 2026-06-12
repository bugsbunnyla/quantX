"""
=========================================================
🚀 QUANT XPERT X CONFIGURATION
=========================================================

DEFAULT_SYMBOLS:
    Default ETF / stock universe

DEFAULT_CAPITAL:
    Default trade capital

These values are used by main.py unless
the user overrides them through CLI args
or interactive prompts.

=========================================================
"""

# =========================================================
# DEFAULT SYMBOL UNIVERSE
# =========================================================

DEFAULT_SYMBOLS = [
    "SCHD",
    "VOO",
    "VOOG",
    "VTI",
    "IONQ",
    "RGTI",
    "MU",
    "PL"
]

# =========================================================
# DEFAULT USER TRADE CAPITAL
# =========================================================

DEFAULT_CAPITAL = 100.0

# =========================================================
# DEFAULT OUTPUT MODE
# =========================================================

DEFAULT_VIEW = "both"