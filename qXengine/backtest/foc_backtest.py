import numpy as np
import pandas as pd

from ..PickleDataManager import PickleDataManager
from ..strategies.FormulaOutput import FormulaOutput
from ..strategies.FormulaInfo import FormulaInfo

def run():
        # =====================================================
        # USE SINGLE SOURCE OF TRUTH
        # =====================================================

        #symbols = ["SCHD","VOO","VOOG","VTI","IONQ","RGTI","MU","PL",]
        #symbols = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","DOGEUSDT",]
        #symbols = ["AAPL","MSFT","NVDA","AMD","SPY","QQQ","IWM","TLT","GLD",]
        #symbols = ["SCHD","VOO","VOOG","VTI","IONQ","RGTI","MU","PL","BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT","DOGEUSDT","AAPL","MSFT","NVDA","AMD","SPY","QQQ","IWM","TLT","GLD"]


        # -------------------------------------
        # DATA LOAD (ENV-AWARE)
        # -------------------------------------

        dm = PickleDataManager("backtest")
       
        data = {}

        for symbol in symbols:

          try:
            df = dm.fetch_store(symbol)

            if df is not None and not df.empty:
                data[symbol] = df
                print(f"[DATA] Loaded {symbol} ({len(df)} rows)")
            else:
                print(f"[DATA] Empty dataset {symbol}")

          except Exception as e:
            print(f"[DATA] Failed {symbol}: {e}")

        if not data:
            raise RuntimeError("[DATA] Universe empty")

        print("[DATA] Assets:", list(data.keys()))
        return data 
  
def test1(data) :
      legacy = FormulaOutput(data)
      legacy_out = legacy.assemble()

      new = cls(data)
      new_out = new.report_new()

      legacy_out, new_out = legacy_out.align(new_out)

      comparison = {}

      for idx in legacy_out.index:
        comparison[idx] = {}

        for sym in legacy_out.columns:
            a = legacy_out.loc[idx, sym]
            b = new_out.loc[idx, sym]

            comparison[idx][sym] = (
                np.isclose(a, b, atol=tol, rtol=tol)
                if isinstance(a, (int, float, np.number))
                else a == b
            )
      return {
        "legacy": legacy_out,
        "dag": new_out,
        "comparison": pd.DataFrame(comparison),
      }
def test(data, tol=1e-6):

    legacy = FormulaOutput(data)
    legacy_out = legacy.assemble()

    new = FormulaInfo(data)
    new_out = new.report_new()

    print("legacy shape :", legacy_out.shape)
    print("new shape    :", new_out.shape)

    print("legacy index names :", legacy_out.index.names)
    print("new index names    :", new_out.index.names)

    # Make index metadata identical
    new_out.index = new_out.index.set_names(legacy_out.index.names)
    new_out.columns = legacy_out.columns

    # Align
    #legacy_out, new_out = legacy_out.align(new_out)

    legacy_out, new_out = legacy_out.align(new_out, join="outer")

    is_num = legacy_out.apply(lambda col: col.map(lambda x: isinstance(x, (int, float, np.number)))).all()

    comparison = pd.DataFrame(index=legacy_out.index, columns=legacy_out.columns)

    for col in legacy_out.columns:

        a = legacy_out[col]
        b = new_out[col]

        if pd.api.types.is_numeric_dtype(a) or pd.api.types.is_numeric_dtype(b):

            comparison[col] = np.isclose(a.astype(float), b.astype(float),
                                          atol=tol, rtol=tol, equal_nan=True)

        else:
            comparison[col] = a.eq(b) | (a.isna() & b.isna())

    return {
        "legacy": legacy_out,
        "new": new_out,
        "comparison": comparison
    }

def test0(data, tol=1e-6):

    legacy = FormulaOutput(data)
    legacy_out = legacy.assemble()
    #print("[DEBUG] legacy_out",legacy_out)

    new = FormulaInfo(data)
    new_out = new.report_new()

    legacy_out, new_out = legacy_out.align(new_out)

    comparison = {}

    for idx in legacy_out.index:
        comparison[idx] = {}

        for sym in legacy_out.columns:
            a = legacy_out.loc[idx, sym]
            b = new_out.loc[idx, sym]

            comparison[idx][sym] = (
                np.isclose(a, b, atol=tol, rtol=tol)
                if isinstance(a, (int, float, np.number))
                else a == b
            )

    return {
        "legacy": legacy_out,
        "dag": new_out,
        "comparison": pd.DataFrame(comparison),
    }

def main():
    testdata = run()
    result = test(testdata)
    print("[TEST] completed result", result)

if __name__ == "__main__":
    main()
