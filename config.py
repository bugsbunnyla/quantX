# ===============================================================
# config : centralized configuration for Quant Xpert
# Date: 2026/06/22 
# Author : bugsbunnyla
# Comment : drives configurable settings for data
# ===============================================================
"""
=========================================================
 QUANT XPERT X CONFIGURATION
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
    "PL",
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "AAPL", 
    "MSFT", 
    "NVDA", 
    "AMD",
    "SPY",
    "QQQ",
    "IWM",
    "TLT",
    "GLD",
]

# =========================================================
# DEFAULT USER TRADE CAPITAL
# =========================================================

DEFAULT_CAPITAL = 100.0

# =========================================================
# DEFAULT OUTPUT MODE
# =========================================================

DEFAULT_VIEW = "both"

# =========================================================
# ENV
# =========================================================
ENVIRONMENTS = {

    "production": {

        "cache_path": "./data/cache/production",

        "market_data": {

            "equities": {
                "provider": "yahoo",
                "allow_api": True,
                "allow_cache": True
            },

            "crypto": {
                "provider": "binance",
                "allow_api": True,
                "allow_cache": True
            }
        }
    },

    "backtest": {

        "cache_path": "./data/cache/backtest",

        "market_data": {

            "equities": {
                "provider": "yahoo",
                "allow_api": False,
                "allow_cache": True
            },

            "crypto": {
                "provider": "coingecko",
                "allow_api": False,
                "allow_cache": True
            }
        }
    }
}

# ========================================================
# END OF config
# ========================================================