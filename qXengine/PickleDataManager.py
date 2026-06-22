# ===============================================================
# PickleDataManager : data engine to drive data in Quant Xpert
# Date: 2026/06/22 
# Author : bugsbunnyla
# Comment : creates, loads, stores data , cache data
# ===============================================================
import os
import pandas as pd
from datetime import datetime

from config import ENVIRONMENTS, DEFAULT_SYMBOLS


class PickleDataManager:

    # =========================================================
    # INIT
    # =========================================================
    def __init__(self, run_option="production"):

        if run_option not in ENVIRONMENTS:
            raise ValueError(f"Invalid run_option: {run_option}")

        self.run_option = run_option

        self.env = ENVIRONMENTS[run_option]

        # cache path from config
        self.path = self.env["cache_path"]
        os.makedirs(self.path, exist_ok=True)

        # provider config
        self.market_data = self.env.get("market_data", {})

        # default lookback (safe fallback)
        self.lookback_years = 4

    # =========================================================
    # DATE
    # =========================================================
    def _today(self):
        return datetime.now().strftime("%Y%m%d")

    # =========================================================
    # FILE PATH (SNAPSHOT)
    # =========================================================
    def _file_path(self, symbol: str):
        return os.path.join(
            self.path,
            f"{symbol}.{self._today()}.pkl"
        )

    # =========================================================
    # SYMBOL UNIVERSE (SINGLE SOURCE OF TRUTH)
    # =========================================================
    def get_all_symbols(self):
        return DEFAULT_SYMBOLS

    # =========================================================
    # NORMALIZE DATAFRAME
    # =========================================================
    def _normalize_columns(self, df: pd.DataFrame):

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [str(c).lower() for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]

        return df

    # =========================================================
    # SERIES EXTRACTOR
    # =========================================================
    def _extract_series(self, df, col):

        if col not in df.columns:
            return None

        data = df[col]

        if isinstance(data, pd.DataFrame):
            return data.iloc[:, 0]

        return data

    # =========================================================
    # ASSET TYPE DETECTION
    # =========================================================
    def _asset_type(self, symbol: str):

        if symbol.endswith("USDT"):
            return "crypto"

        return "equities"

    # =========================================================
    # PROVIDER CONFIG
    # =========================================================
    def _provider_config(self, symbol: str):

        asset_type = self._asset_type(symbol)

        return self.market_data.get(asset_type, {
            "provider": "yahoo",
            "allow_api": True,
            "allow_cache": True
        })

    # =========================================================
    # API ROUTER
    # =========================================================
    def _fetch_api(self, symbol: str):

        cfg = self._provider_config(symbol)
        provider = cfg["provider"]

        if provider == "yahoo":
            return self._fetch_yahoo(symbol)

        if provider == "binance":
            return self._fetch_binance(symbol)

        if provider == "coingecko":
            return self._fetch_binance(symbol)

        raise ValueError(f"Unsupported provider: {provider}")

    # =========================================================
    # YAHOO FINANCE
    # =========================================================
    def _fetch_yahoo(self, symbol: str):

        import yfinance as yf

        df = yf.download(
            symbol,
            period=f"{self.lookback_years}y",
            interval="1d",
            auto_adjust=False,
            progress=False
        )

        df = df.reset_index()
        return self._normalize_columns(df)

    # =========================================================
    # CRYPTO FALLBACK (BINANCE STYLE)
    # =========================================================
    def _fetch_binance(self, symbol: str):

        import yfinance as yf

        df = yf.download(
            symbol.replace("USDT", "-USD"),
            period=f"{self.lookback_years}y",
            interval="1d",
            auto_adjust=False,
            progress=False
        )

        df = df.reset_index()
        return self._normalize_columns(df)

    # =========================================================
    # STORE SNAPSHOT
    # =========================================================
    def _store(self, path: str, df: pd.DataFrame):

        df = self._normalize_columns(df)

        close = self._extract_series(df, "close")

        if close is not None:
            df["ret"] = close.pct_change()

        df.to_pickle(path)

    # =========================================================
    # FETCH + CACHE LOGIC
    # =========================================================
    def fetch_store(self, symbol: str, force_refresh: bool = False):

        path = self._file_path(symbol)

        cfg = self._provider_config(symbol)

        allow_api = cfg.get("allow_api", True)
        allow_cache = cfg.get("allow_cache", True)

        # -----------------------------------------------------
        # BACKTEST MODE → CACHE ONLY
        # -----------------------------------------------------
        if self.run_option == "backtest":

            if os.path.exists(path):
                df = pd.read_pickle(path)
                df = self._normalize_columns(df)

                close = self._extract_series(df, "close")
                if close is not None:
                    df["ret"] = close.pct_change()

                return df

            raise FileNotFoundError(f"[BACKTEST] Missing snapshot: {path}")

        # -----------------------------------------------------
        # PRODUCTION MODE → CACHE OR API
        # -----------------------------------------------------
        if os.path.exists(path) and not force_refresh and allow_cache:

            df = pd.read_pickle(path)
            df = self._normalize_columns(df)

            close = self._extract_series(df, "close")
            if close is not None:
                df["ret"] = close.pct_change()

            return df

        if not allow_api:
            raise RuntimeError(f"[PRODUCTION] API disabled for {symbol}")

        print(f"[DATA] Fetching {symbol} via API...")

        df = self._fetch_api(symbol)

        if df is None or df.empty:
            raise ValueError(f"Empty dataset: {symbol}")

        if allow_cache:
            self._store(path, df)

        return df

    # =========================================================
    # LOAD ALL TODAY FILES ONLY
    # =========================================================
    def load_all(self):

        data = {}
        today = self._today()

        for file in os.listdir(self.path):

            if file.endswith(f"{today}.pkl"):

                symbol = file.split(".")[0]

                df = pd.read_pickle(os.path.join(self.path, file))
                df = self._normalize_columns(df)

                data[symbol] = df

        return data

    # =========================================================
    # BOOTSTRAP ALL SYMBOLS
    # =========================================================
    def bootstrap_all(self, force_refresh: bool = False):

        dataset = {}

        for symbol in self.get_all_symbols():

            try:
                df = self.fetch_store(symbol, force_refresh)

                if df is not None and not df.empty:
                    dataset[symbol] = df
                    print(f"[DATA] Loaded {symbol} ({len(df)})")

            except Exception as e:
                print(f"[DATA] Failed {symbol}: {e}")

        print(f"[DATA] Bootstrap complete: {len(dataset)} assets")

        return dataset

# ================================================================
# END OF PICKLEDATAMANAGER
# ================================================================