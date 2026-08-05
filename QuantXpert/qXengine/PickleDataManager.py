# ===============================================================
# PickleDataManager : data engine to drive data in Quant Xpert
# Date: 2026/06/22 
# Author : bugsbunnyla
# Comment : creates, loads, stores data , cache data
# ===============================================================
import os
import pandas as pd
from datetime import datetime,timedelta

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
    # ASSET TYPE
    # =========================================================
    def _asset_type(self, symbol: str):
        return "crypto" if symbol.endswith("USDT") else "equities"

    # =========================================================
    # CONFIG
    # =========================================================
    def _provider_config(self, symbol: str):
        asset = self._asset_type(symbol)
        return self.market_data.get(asset, {})

    # =========================================================
    # NORMALIZE
    # =========================================================
    def _normalize(self, df):

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [str(c).lower() for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]

        return df

    # =========================================================
    # BACKTEST FETCH (SAFE: no url, no api keys)
    # =========================================================
    def _fetch_backtest(self, symbol: str):

        start = (datetime.utcnow() - timedelta(days=365 * self.lookback_years)).strftime("%Y-%m-%d")

        # -------------------------
        # CRYPTO → BINANCE (direct REST, no config)
        # -------------------------
        if symbol.endswith("USDT"):

            import requests

            url = "https://api.binance.com/api/v3/klines"

            params = {
                "symbol": symbol,
                "interval": "1d",
                "limit": 1000
            }

            all_data = []
            start_time = int(pd.Timestamp(start).timestamp() * 1000)

            while True:

                params["startTime"] = start_time

                r = requests.get(url, params=params)
                batch = r.json()

                if not isinstance(batch, list) or len(batch) == 0:
                    break

                all_data.extend(batch)
                start_time = batch[-1][0] + 1

                if len(batch) < 1000:
                    break

            df = pd.DataFrame(all_data, columns=[
                "time","open","high","low","close","volume",
                "close_time","qav","trades","tbb","tbq","ignore"
            ])

            df = df[["time","open","high","low","close","volume"]]
            df["time"] = pd.to_datetime(df["time"], unit="ms")

        # -------------------------
        # EQUITIES → YAHOO (direct library)
        # -------------------------
        else:

            import yfinance as yf

            df = yf.download(
                symbol,
                period=f"{self.lookback_years}y",
                interval="1d",
                auto_adjust=False,
                progress=False
            ).reset_index()

        return self._normalize(df)

    # =========================================================
    # PRODUCTION FETCH (USES CONFIG FULLY)
    # =========================================================
    def _fetch_production(self, symbol: str):

        cfg = self._provider_config(symbol)
        provider = cfg.get("provider")

        if provider == "yahoo":
            return self._fetch_yahoo(symbol)

        if provider == "binance":
            return self._fetch_binance(cfg, symbol)

        raise ValueError(f"Unsupported provider: {provider}")

    # =========================================================
    # YAHOO PRODUCTION
    # =========================================================
    def _fetch_yahoo(self, symbol):

        import yfinance as yf

        df = yf.download(
            symbol,
            period=f"{self.lookback_years}y",
            interval="1d",
            progress=False
        ).reset_index()

        return self._normalize(df)

    # =========================================================
    # BINANCE PRODUCTION (URL + KEYS)
    # =========================================================
    def _fetch_binance_api(self, cfg, symbol):

        import requests

        if not cfg.get("allow_api", True):
            raise RuntimeError("Binance API disabled")

        url = cfg["url"]
        headers = {}

        keys = cfg.get("api_keys", {})
        if keys.get("api_key"):
            headers["X-MBX-APIKEY"] = keys["api_key"]

        r = requests.get(url, params={
            "symbol": symbol,
            "interval": "1d",
            "limit": 1000
        }, headers=headers)

        data = r.json()

        if isinstance(data, dict) and "code" in data:
            raise ValueError(f"Binance error: {data}")

        df = pd.DataFrame(data)
        return self._normalize(df)

    # =========================================================
    # FETCH binance/yahoo
    # =========================================================
    def _load_backtest(self, symbol: str):

      import yfinance as yf

      path = self._file_path(symbol)

      # =========================================================
      # 1. LOAD FROM CACHE FIRST (HARD PRIORITY)
      # =========================================================
      if os.path.exists(path):

        df = pd.read_pickle(path)
        df = self._normalize(df)

        close = self._extract_series(df, "close")
        if close is not None:
            df["ret"] = close.pct_change()

        return df

      # =========================================================
      # 2. BUILD DATASET (NO API, ONLY YFINANCE)
      # =========================================================
      print(f"[BACKTEST] Building dataset {symbol}...")

      asset_type = self._asset_type(symbol)

      # =========================================================
      # EQUITIES
      # =========================================================
      if asset_type == "equities":

        df = yf.download(
            symbol,
            period=f"{self.lookback_years}y",
            interval="1d",
            auto_adjust=False,
            progress=False
        )

      # =========================================================
      # CRYPTO (USDT → -USD MAPPING)
      # =========================================================
      else:

        yf_symbol = symbol.replace("USDT", "-USD")

        df = yf.download(
            yf_symbol,
            period=f"{self.lookback_years}y",
            interval="1d",
            auto_adjust=False,
            progress=False
        )

      # =========================================================
      # 3. VALIDATION
      # =========================================================
      if df is None or df.empty:
        raise RuntimeError(f"[BACKTEST] No data for {symbol}")

      # =========================================================
      # 4. NORMALIZE
      # =========================================================
      df = df.reset_index()
      df = self._normalize(df)


      # =========================================================
      # 5. STORE SNAPSHOT (DETERMINISTIC BACKTEST CACHE)
      # =========================================================
      self._store(path, df)

      return df
    def _extract_series(self, df, col):

       if df is None or df.empty:
          return None

       col = col.lower()

       if col not in df.columns:
          return None

       data = df[col]

       if isinstance(data, pd.DataFrame):
          return data.iloc[:, 0]

       return data

    # =========================================================
    # STORE
    # =========================================================
    def _store(self, path, df):

        df = self._normalize(df)

        if "close" in df.columns:
            df["ret"] = df["close"].astype(float).pct_change()

        df.to_pickle(path)

    # =========================================================
    # MAIN FETCH
    # =========================================================
    def fetch_store(self, symbol: str, force_refresh=False):

        path = self._file_path(symbol)
        cfg = self._provider_config(symbol)

        allow_cache = cfg.get("allow_cache", True)
        allow_api = cfg.get("allow_api", True)

        # =====================================================
        # BACKTEST MODE
        # =====================================================
        if self.run_option == "backtest":

            if os.path.exists(path) and not force_refresh:
                df = pd.read_pickle(path)
                return self._normalize(df)

            print(f"[BACKTEST] Building dataset {symbol}...")

            df = self._load_backtest(symbol)

            if df is None or df.empty:
                raise ValueError(f"[BACKTEST] No data for {symbol}")

            self._store(path, df)
            return df

        # =====================================================
        # PRODUCTION MODE
        # =====================================================
        if os.path.exists(path) and allow_cache and not force_refresh:
            return self._normalize(pd.read_pickle(path))

        if not allow_api:
            raise RuntimeError(f"[PRODUCTION] API disabled for {symbol}")

        df = self._fetch_production(symbol)

        if allow_cache:
            self._store(path, df)

        return df

    # =========================================================
    # BOOTSTRAP
    # =========================================================
    def bootstrap_all(self, symbols):

        dataset = {}

        for s in symbols:
            try:
                df = self.fetch_store(s)
                dataset[s] = df
                print(f"[DATA] Loaded {s} ({len(df)})")
            except Exception as e:
                print(f"[DATA] Failed {s}: {e}")

        return dataset

# ================================================================
# END OF PICKLEDATAMANAGER
# ================================================================