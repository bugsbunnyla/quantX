# Pre requisite data
# intraday.pkl structure (per asset)
#
#{
#    "BTCUSDT": DataFrame,
#    "ETHUSDT": DataFrame,
#    "AAPL": DataFrame,
#    "SPY": DataFrame
#}
#Each DataFrame:
#
#datetime
#open
#high
#low
#close
#volume
#ret
# 2. INTRADAY LOADER (NEW FILE)
# IntradayDataManager.py
import pandas as pd
import os


class IntradayDataManager:

    def __init__(self, path="./data/cache/intraday/"):

        self.path = path


    def load(self):

        df = pd.read_pickle(os.path.join(self.path, "intraday.pkl"))

        for k, v in df.items():

            v["ret"] = v["close"].pct_change()

        return df
