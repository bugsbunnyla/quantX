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
    def __init__(self, run_option):

        if run_option not in ENVIRONMENTS:
            raise ValueError(f"Invalid run_option: {run_option}")

        self.run_option = run_option
        self.env = ENVIRONMENTS[run_option]

        self.path = self.env["cache_path"]
        os.makedirs(self.path, exist_ok=True)

        self.market_data = self.env.get("market_data", {})

        self.lookback_years = 4

    # =========================================================
    # DATE
    # =========================================================
    def _today(self):
        return datetime.now().strftime("%Y%m%d")

    # =========================================================
    # FILE PATH
    # =========================================================
    def _file_path(self, symbol: str):
        return os.path.join(self.path, f"{symbol}.{self._today()}.pkl")

    # =========================================================
    # SYMBOLS
    # =========================================================
    def get_all_symbols(self):
        return DEFAULT_SYMBOLS

    # =========================================================
    # NORMALIZATION
    # =========================================================
    def _normalize_columns(self, df: pd.DataFrame):

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [str(c).lower() for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]

        return df

    # =========================================================
    # SERIES
    # =========================================================
    def _extract_series(self, df, col):

        if col not in df.columns:
            return None

        data = df[col]

        if isinstance(data, pd.DataFrame):
            return data.iloc[:, 0]

        return data

    # =========================================================
    # ASSET TYPE
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

       cfg = self.market_data.get(asset_type, {})

       # if nested provider exists (crypto case)
       provider_name = cfg.get("provider")

       if isinstance(cfg.get(provider_name), dict):
          return cfg[provider_name]

       return cfg

    # =========================================================
    # API ROUTER (SAFE + COMPLETE)
    # =========================================================
    def _fetch_api(self, symbol: str):

        cfg = self._provider_config(symbol)
        provider = cfg.get("provider")

        if provider == "yahoo":
            return self._fetch_yahoo(symbol)

        if provider == "binance":
            return self._fetch_binance(cfg)

        if provider == "kraken":
            return self._fetch_kraken(cfg)

        if provider == "coinbase":
            return self._fetch_coinbase(cfg)

        if provider == "coingecko":
            return self._fetch_coingecko(cfg)

        raise ValueError(f"Unsupported provider: {provider}")

    # =========================================================
    # YAHOO (EQUITIES)
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

        return self._normalize_columns(df.reset_index())

    # =========================================================
    # BINANCE
    # =========================================================
    def _fetch_binance(self, cfg):

        import requests

        if not cfg.get("allow_api", True):
            raise RuntimeError("API disabled for Binance")

        url = cfg["url"]
        keys = cfg.get("api_keys", {})

        headers = {}
        if keys.get("api_key"):
            headers["X-MBX-APIKEY"] = keys["api_key"]

        params = {
            "symbol": "BTCUSDT",
            "interval": "1d",
            "limit": 1000
        }

        r = requests.get(url, params=params, headers=headers)
        data = r.json()

        df = pd.DataFrame(data)

        return self._normalize_columns(df)

    # =========================================================
    # KRAKEN
    # =========================================================
    def _fetch_kraken(self, cfg):

        import requests

        if not cfg.get("allow_api", True):
            raise RuntimeError("API disabled for Kraken")

        url = cfg["url"]

        pair = "BTCUSD"

        r = requests.get(url, params={
            "pair": pair,
            "interval": 1440
        })

        data = r.json()

        if "result" not in data:
            raise ValueError(f"Kraken error: {data}")

        key = list(data["result"].keys())[0]

        df = pd.DataFrame(data["result"][key], columns=[
            "time","open","high","low","close","vwap","volume","count"
        ])

        df["time"] = pd.to_datetime(df["time"], unit="s")

        return self._normalize_columns(df[["time","open","high","low","close","volume"]])

    # =========================================================
    # COINBASE
    # =========================================================
    def _fetch_coinbase(self, cfg):

        import requests

        if not cfg.get("allow_api", True):
            raise RuntimeError("API disabled for Coinbase")

        base_url = cfg["url"]

        product = "BTC-USD"

        r = requests.get(
            f"{base_url}/{product}/candles",
            params={"granularity": 86400}
        )

        data = r.json()

        if not isinstance(data, list):
            raise ValueError(f"Coinbase error: {data}")

        df = pd.DataFrame(data, columns=[
            "time","low","high","open","close","volume"
        ])

        df["time"] = pd.to_datetime(df["time"], unit="s")

        return self._normalize_columns(df)

    # =========================================================
    # COINGECKO
    # =========================================================
    def _fetch_coingecko(self, cfg):

        import requests

        if not cfg.get("allow_api", True):
            raise RuntimeError("API disabled for CoinGecko")

        url = cfg["url"]

        r = requests.get(
            f"{url}/bitcoin/ohlc",
            params={
                "vs_currency": "usd",
                "days": 365
            }
        )

        data = r.json()

        if not isinstance(data, list):
            raise ValueError(f"CoinGecko error: {data}")

        df = pd.DataFrame(data, columns=[
            "time","open","high","low","close"
        ])

        df["time"] = pd.to_datetime(df["time"], unit="ms")

        df["volume"] = 0

        return self._normalize_columns(df)

    # =========================================================
    # STORE
    # =========================================================
    def _store(self, path: str, df: pd.DataFrame):

        df = self._normalize_columns(df)

        close = self._extract_series(df, "close")

        if close is not None:
            df["ret"] = close.pct_change()
        df = df.dropna(subset=["close"])

        df.to_pickle(path)

    # =========================================================
    # FETCH + CACHE LOGIC
    # =========================================================
    def fetch_store(self, symbol: str, force_refresh: bool = False):

        path = self._file_path(symbol)

        cfg = self._provider_config(symbol)

        allow_api = cfg.get("allow_api", True)
        allow_cache = cfg.get("allow_cache", True)

        # -------------------------
        # BACKTEST (UNCHANGED)
        # -------------------------
        if self.run_option == "backtest":

            if os.path.exists(path):

               df = pd.read_pickle(path)
               df = self._normalize_columns(df)

               close = self._extract_series(df, "close")
               if close is not None:
                  df["ret"] = close.pct_change()

               return df

            # auto fallback instead of crash
            print(f"[BACKTEST] Missing cache for {symbol}, falling back to production fetch...")

            df = self._fetch_api(symbol)

            if df is None or df.empty:
               raise ValueError(f"[BACKTEST] No data for {symbol}")

            # store snapshot so next run is stable
            self._store(path, df)

            return df

        # -------------------------
        # PRODUCTION CACHE FIRST
        # -------------------------
        if os.path.exists(path) and not force_refresh and allow_cache:

            df = pd.read_pickle(path)
            df = self._normalize_columns(df)

            close = self._extract_series(df, "close")
            if close is not None:
                df["ret"] = close.pct_change()

            return df

        if not allow_api:
            raise RuntimeError(f"[PRODUCTION] API disabled for {symbol}")

        print(f"[DATA] Fetching {symbol} via {cfg.get('provider')}...")

        df = self._fetch_api(symbol)

        if df is None or df.empty:
            raise ValueError(f"Empty dataset: {symbol}")

        if allow_cache:
            self._store(path, df)

        return df

    # =========================================================
    # LOAD ALL
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
    # BOOTSTRAP ALL
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
        if len(dataset) == 0:
           raise RuntimeError(f"[DATA] Universe empty in {self.run_option}")

        return dataset

# ================================================================
# END OF PICKLEDATAMANAGER
# ================================================================