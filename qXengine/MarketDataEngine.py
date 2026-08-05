class MarketDataEngine:

    def __init__(self, crypto_loader, equity_loader):

        self.crypto = crypto_loader
        self.equity = equity_loader

    def load(self):

        data = {}

        # =========================
        # CRYPTO LAYER (PICKLE)
        # =========================
        data.update(self.crypto.load_all())


        # =========================
        # EQUITY LAYER (YFINANCE OR PICKLE)
        # =========================
        data.update(self.equity.load_all())


        # =========================
        # STANDARDIZATION FIX
        # =========================
        for k, df in data.items():

            df["ret"] = df["close"].pct_change()

        return data
