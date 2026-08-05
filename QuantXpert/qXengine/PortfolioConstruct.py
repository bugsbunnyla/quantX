# ===============================================================
# PortfolioConstruct class defines the portfolio construction
# Date: 2026/06/22 
# Author : bugsbunnyla
# Comment : Portfolio constructor QuantXpert structure data model
# ===============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .strategies.FormulaOutput import FormulaOutput


class PortfolioConstruct:

    def __init__(self, capital: float = 100000, debug: bool = True):
        self.capital = capital
        self.debug = debug
        self._last_store_id = None
        self._last_result = None

    # ==================================================
    # ENTRY
    # ==================================================
    def invoke(self):

        print("\n[PORTFOLIO CONSTRUCT] starting...")

        store = getattr(FormulaOutput, "STATIC_STORE", None)
        if not store:
            print("NOT READY: STATIC_STORE empty")
            return None

        latest = store[-1]
        store_id = id(latest)

        if self._last_store_id == store_id:
            return self._last_result

        self._last_store_id = store_id

        df = self._build(latest)
        if df.empty:
            return None

        df = self._fix(df)
        df = self._score(df)
        df = self._state(df)

        universe, state = self._select(df)

        if universe.empty:
            universe = df.sort_values("score", ascending=False).head(5)
            state = "FALLBACK"

        top = self._top5(universe)
        weighted = self._weight(top)

        img = self._pie(weighted, state)

        self._report(weighted, state, img)

        self._last_result = {
            "state": state,
            "table": weighted,
            "chart": img
        }

        return self._last_result

    # ==================================================
    # BUILD SAFE TABLE (ROBUST MULTIINDEX HANDLING)
    # ==================================================
    def _build(self, df):

        if not isinstance(df, pd.DataFrame):
            return pd.DataFrame()

        if not isinstance(df.index, pd.MultiIndex):
            return pd.DataFrame()

        symbols = df.columns
        out = {}

        for idx in df.index.unique():
            try:
                cat, metric = idx
                out[f"{cat}__{metric}"] = df.loc[idx]
            except:
                continue

        return pd.DataFrame(out, index=symbols).T

    # ==================================================
    # FIX FEATURES (STRICT CONTRACT ENFORCEMENT)
    # ==================================================
    def _fix(self, df):

        df = df.copy()
        symbols = df.columns

        def get(row):
            if row in df.index:
                return pd.to_numeric(df.loc[row], errors="coerce").fillna(0)
            return pd.Series(0.0, index=symbols)

        ts = get("alpha__ts")
        xs = get("alpha__xs")

        if xs.abs().sum() == 0:
            xs = (ts - ts.mean()) / (ts.std() + 1e-9)

        df.loc["alpha__xs"] = xs
        df.loc["alpha__pure"] = ts + xs

        # ALWAYS SAFE HIT RATIO
        if "performance__hit_ratio" not in df.index:
            df.loc["performance__hit_ratio"] = 0.5
        else:
            df.loc["performance__hit_ratio"] = (
                pd.to_numeric(df.loc["performance__hit_ratio"], errors="coerce")
                .fillna(0.5)
            )

        return df

    # ==================================================
    # SCORE (NOW PORTFOLIO-AWARE)
    # ==================================================
    def _score(self, df):

        symbols = df.columns

        def g(x):
            if x in df.index:
                return pd.to_numeric(df.loc[x], errors="coerce").fillna(0)
            return pd.Series(0.0, index=symbols)

        ts = g("alpha__ts")
        xs = g("alpha__xs")
        pure = g("alpha__pure")

        sharpe = g("risk__sharpe")
        cvar = g("risk__cvar")
        vol = g("risk__volatility")
        hit = g("performance__hit_ratio")

        kelly = g("portfolio__kelly") if "portfolio__kelly" in df.index else 0
        risk_parity = g("portfolio__risk_parity") if "portfolio__risk_parity" in df.index else 0

        def norm(x):
            x = x.replace([np.inf, -np.inf], np.nan).fillna(0)
            return (x - x.mean()) / (x.std() + 1e-9)

        portfolio_boost = 0
        if isinstance(kelly, pd.Series):
            portfolio_boost = norm(kelly) * 0.15 + norm(risk_parity) * 0.10

        score = (
            0.18 * norm(ts) +
            0.18 * norm(xs) +
            0.10 * norm(pure) +
            0.22 * norm(sharpe) +
            0.12 * norm(hit) -
            0.10 * norm(cvar) -
            0.05 * norm(vol) +
            portfolio_boost
        )

        df.loc["score"] = score
        return df

    # ==================================================
    # STATE
    # ==================================================
    def _state(self, df):

        score = df.loc["score"].replace([np.inf, -np.inf], np.nan).fillna(0)

        q = score.quantile([0.25, 0.5, 0.75])

        df.loc["state"] = pd.cut(
            score,
            bins=[-np.inf, q[0.25], q[0.5], q[0.75], np.inf],
            labels=["WEAK", "MEDIUM", "STRONG", "VERY_STRONG"]
        ).astype(str)

        return df

    # ==================================================
    # SELECT
    # ==================================================
    def _select(self, df):

        state = df.loc["state"]

        vs = state[state == "VERY_STRONG"].index
        if len(vs):
            return df[vs].T, "VERY_STRONG"

        s = state[state == "STRONG"].index
        if len(s):
            return df[s].T, "STRONG"

        return pd.DataFrame(), "NOT_READY"

    # ==================================================
    # TOP
    # ==================================================
    def _top5(self, df):
        return df.sort_values("score", ascending=False).head(5)

    # ==================================================
    # WEIGHT (SAFE NORMALIZATION)
    # ==================================================
    def _weight(self, df):

        df = df.copy()

        scores = pd.to_numeric(df["score"], errors="coerce").fillna(0).abs()

        if scores.sum() <= 0:
            df["weight"] = 1 / len(df)
        else:
            df["weight"] = scores / scores.sum()

        df["allocation"] = df["weight"] * self.capital

        # normalize again (IMPORTANT FIX)
        df["weight"] = df["weight"] / df["weight"].sum()

        return df

    # ==================================================
    # PIE (FIXED)
    # ==================================================
    def _pie(self, df, state):

        fig, ax = plt.subplots(figsize=(6, 6))

        weights = pd.to_numeric(df["weight"], errors="coerce").fillna(0)
        weights = np.clip(weights.values, 0, None)

        if weights.sum() == 0:
            weights = np.ones(len(weights)) / len(weights)

        ax.pie(weights, labels=df.index.astype(str), autopct="%1.1f%%")
        ax.set_title(f"Portfolio ({state})")

        path = "portfolio_pie.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return path

    # ==================================================
    # REPORT
    # ==================================================
    def _report(self, df, state, img):

        print("\n==============================")
        print(" PORTFOLIO CONSTRUCT FINAL")
        print("==============================")
        print(f"STATE: {state}")
        print(f"CAPITAL: {self.capital}")
        print(f"ASSETS: {len(df)}")

        for i, r in df.iterrows():
            print(f"{i} | {r['weight']:.2%} | ${r['allocation']:.2f}")

        print(f"\nPie saved -->{img}")

# ===================================================================
# END OF PORTFOLIO CONSTRUCT
# ===================================================================