import pandas as pd
import os
from .StrategyConfig import STRATEGY_CONFIG


class PickleDataManager:

    def __init__(self):

        self.path = "./data/cache/crypto/"
        os.makedirs(self.path, exist_ok=True)

        self.config = STRATEGY_CONFIG["data"]
        self.lookback_years = self.config["lookback_years"]

    # =========================================================
    # SYMBOLS FROM CONFIG
    # =========================================================
    def get_all_symbols(self):

        assets = self.config["assets"]

        symbols = []
        for k in assets:
            symbols.extend(assets[k])

        return list(set(symbols))

    # =========================================================
    # SAFE COLUMN NORMALIZER (CRITICAL FIX)
    # =========================================================
    def _normalize_columns(self, df: pd.DataFrame):

        # Handle MultiIndex (yfinance issue)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Force string + lowercase
        df.columns = [str(c).lower() for c in df.columns]

        # Remove duplicate columns (VERY IMPORTANT)
        df = df.loc[:, ~df.columns.duplicated()]

        return df

    # =========================================================
    # SAFE SERIES EXTRACTOR (FIXES ret BUG)
    # =========================================================
    def _extract_series(self, df, col):

        if col not in df.columns:
            return None

        data = df[col]

        # If DataFrame (multi-column issue) → take first column
        if isinstance(data, pd.DataFrame):
            return data.iloc[:, 0]

        return data

    # =========================================================
    # LOAD SINGLE SYMBOL (CACHE ONLY)
    # =========================================================
    def load_symbol(self, file):

        df = pd.read_pickle(os.path.join(self.path, file))

        df = self._normalize_columns(df)

        close = self._extract_series(df, "close")

        if close is not None:
            df["ret"] = close.pct_change()

        return df

    # =========================================================
    # LOAD ALL CACHE
    # =========================================================
    def load_all(self):

        data = {}

        for file in os.listdir(self.path):

            if file.endswith(".pkl"):

                symbol = file.replace(".pkl", "")

                df = self.load_symbol(file)

                data[symbol] = df

        return data

    # =========================================================
    # API ROUTER
    # =========================================================
    def _fetch_api(self, symbol: str):

        if symbol.endswith("USDT"):
            return self._fetch_crypto(symbol)
        else:
            return self._fetch_yfinance(symbol)

    # =========================================================
    # YFINANCE FETCH (SAFE)
    # =========================================================
    def _fetch_yfinance(self, symbol: str):

        import yfinance as yf

        df = yf.download(
            symbol,
            period=f"{self.lookback_years}y",
            interval="1d",
            auto_adjust=False,
            progress=False
        )

        df = df.reset_index()
        df = self._normalize_columns(df)

        return df

    # =========================================================
    # CRYPTO FETCH (SAFE FALLBACK)
    # =========================================================
    def _fetch_crypto(self, symbol: str):

        import yfinance as yf

        df = yf.download(
            symbol.replace("USDT", "-USD"),
            period=f"{self.lookback_years}y",
            interval="1d",
            auto_adjust=False,
            progress=False
        )

        df = df.reset_index()
        df = self._normalize_columns(df)

        return df

    # =========================================================
    # SAFE STORE
    # =========================================================
    def _store(self, symbol: str, df: pd.DataFrame):

        path = os.path.join(self.path, f"{symbol}.pkl")

        df = self._normalize_columns(df)

        close = self._extract_series(df, "close")

        if close is not None:
            df["ret"] = close.pct_change()

        df.to_pickle(path)

    # =========================================================
    # MAIN FETCH + CACHE
    # =========================================================
    def fetch_store(self, symbol: str, force_refresh: bool = False):

        path = os.path.join(self.path, f"{symbol}.pkl")

        # CACHE HIT
        if os.path.exists(path) and not force_refresh:

            df = pd.read_pickle(path)

            df = self._normalize_columns(df)

            close = self._extract_series(df, "close")

            if close is not None:
                df["ret"] = close.pct_change()

            return df

        # CACHE MISS → API
        print(f"[DATA] Fetching {symbol} from API...")

        df = self._fetch_api(symbol)

        if df is None or df.empty:
            raise ValueError(f"[DATA] Empty dataset for {symbol}")

        self._store(symbol, df)

        print(f"[DATA] Cached {symbol}.pkl")

        return df

    # =========================================================
    # BOOTSTRAP ALL SYMBOLS
    # =========================================================
    def bootstrap_all(self, force_refresh: bool = False):

        dataset = {}

        symbols = self.get_all_symbols()

        for symbol in symbols:

            try:
                df = self.fetch_store(symbol, force_refresh)
                dataset[symbol] = df

            except Exception as e:
                print(f"[DATA] Failed {symbol}: {e}")

        print(f"[DATA] Bootstrap complete: {len(dataset)} assets")

        return dataset