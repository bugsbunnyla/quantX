#!/usr/bin/env python3
"""
QuantXpert Final Cleanup Script
================================
1. Deletes root folders: alpha, backtest, execution, pairs, portfolio,
   regime, risk, stats, runs  (preserves qXengine/ and core/)
2. Deletes root main.py
3. Strips ALL top-level functions/classes with suffix '0' or '_0' from
   every .py file recursively.
4. Removes leftover references to deleted *_0 fallback engines.
   EXCEPTION: SplitEngine.split() fallback is INLINED (not removed).
5. Injects 1a_train.pkl / 1b_val.pkl persistence into
   qXengine/backtest/pipeline_backtest.py after the split.
6. Changes Storage base_dir from "runs" → "quantx_runs".
7. Replaces core/system.py with a self-contained, pruned version that
   no longer imports deleted folders.
8. Deletes ALL files in qXengine/backtest/ EXCEPT:
      __init__.py, e2ebacktest.py, pipeline_backtest.py
9. Keeps qXengine/strategies/ fully intact.
10. NEVER deletes __init__.py files (even if empty).
"""

import os
import re
import shutil
from pathlib import Path

SRC_DIR = Path("../qX")
DST_DIR = Path("QuantXpert")

DELETE_ROOT_FOLDERS = [
    "alpha", "backtest", "execution", "pairs",
    "portfolio", "regime", "risk", "stats", "runs"
]
DELETE_ROOT_FILES = ["main.py"]

TOPLEVEL_0_RE = re.compile(
    r'^(class\s+\w+_0\b|class\s+\w+0\b|def\s+\w+_0\s*\(|def\s+\w+0\s*\()'
)

SKIP_COPY_NAMES = {".git", "__pycache__", ".pytest_cache", ".mypy_cache"}


def copy_tree(src: Path, dst: Path):
    if src.name in SKIP_COPY_NAMES:
        return
    if src.is_dir():
        dst.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            copy_tree(item, dst / item.name)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def safe_read(path: Path) -> str:
    """Read text with fallback for encoding errors."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def remove_toplevel_zero_definitions(text: str) -> str:
    lines = text.splitlines(keepends=True)
    result = []
    i = 0
    while i < len(lines):
        if TOPLEVEL_0_RE.match(lines[i]):
            block_indent = len(lines[i]) - len(lines[i].lstrip())
            i += 1
            while i < len(lines):
                if lines[i].strip() == "":
                    i += 1
                    continue
                curr_indent = len(lines[i]) - len(lines[i].lstrip())
                if curr_indent <= block_indent and lines[i].strip():
                    break
                i += 1
            continue
        result.append(lines[i])
        i += 1
    return "".join(result)


def inline_split_fallback(text: str) -> str:
    """
    Replace the SplitEngine_0 fallback call inside SplitEngine.split()
    with inline per-symbol row-split logic.
    """
    # Find and replace the exact fallback block
    old_fallback = '''        if not ranges:
            print("[SPLIT] No valid dates found; falling back to per-symbol row split.")
            return SplitEngine_0().split(data, split_ratio)'''
    
    new_fallback = '''        if not ranges:
            print("[SPLIT] No valid dates found; falling back to per-symbol row split.")
            train, val = {}, {}
            for symbol, df in data.items():
                if not isinstance(df, pd.DataFrame):
                    continue
                split_idx = int(len(df) * split_ratio)
                if split_idx < 10:
                    split_idx = len(df) // 2
                train[symbol] = df.iloc[:split_idx].copy()
                val[symbol] = df.iloc[split_idx:].copy()
            return train, val'''
    
    return text.replace(old_fallback, new_fallback)


def remove_zero_fallback_refs(text: str) -> str:
    """
    Remove generic leftover references to deleted *_0 classes.
    (SplitEngine fallback is handled separately by inline_split_fallback.)
    """
    text = re.sub(
        r'^\s*return\s+\w+_0\(\)\.\w+\([^)]*\)\s*$',
        '        raise RuntimeError("Fallback engine removed in QuantXpert cleanup")',
        text,
        flags=re.MULTILINE,
    )
    return text


def change_storage_base_dir(text: str) -> str:
    return text.replace(
        'def __init__(self, base_dir="runs"):',
        'def __init__(self, base_dir="quantx_runs"):',
    )


def inject_pickle_dump(text: str) -> str:
    lines = text.splitlines(keepends=True)
    out = []
    i = 0
    injected = False
    while i < len(lines):
        out.append(lines[i])
        if not injected and "train_data, val_data = self.split_engine.split(" in lines[i]:
            j = i + 1
            while j < len(lines):
                out.append(lines[j])
                if lines[j].strip().startswith("print(") and "[SRFO] Temporal split:" in lines[j]:
                    j += 1
                    break
                j += 1
            inject_lines = [
                "\n",
                "        # === QUANTXPERT: Persist split datasets ===\n",
                '        os.makedirs("quantx_runs", exist_ok=True)\n',
                '        with open("quantx_runs/1a_train.pkl", "wb") as f:\n',
                '            pickle.dump(train_data, f)\n',
                '        with open("quantx_runs/1b_val.pkl", "wb") as f:\n',
                '            pickle.dump(val_data, f)\n',
                "        # === End persistence ===\n",
                "\n",
            ]
            out.extend(inject_lines)
            injected = True
            i = j
            continue
        i += 1
    return "".join(out)


CLEAN_SYSTEM_PY = '''\
# ===============================================================
# core/system : core data structures of Quant Xpert
# Date: 2026/06/22
# Author : bugsbunnyla
# Comment : initiates the multi index output of data structure
# ===============================================================
import numpy as np
import pandas as pd
import requests


class SymbolAssembler:
    def assemble_symbol_report(
        self,
        symbol,
        price,
        ret,
        volume,
        alpha_ts,
        alpha_xs,
        alpha_pure,
        beta,
        residual,
        volatility,
        sharpe,
        drawdown,
        cvar,
        decorrelation,
        wiggle,
        rank,
        zscore,
        winsor,
        tanh,
        detrend,
        weight,
        risk_parity,
        kelly,
        entropy,
        regime,
        liquidity_adj_vol,
        slippage,
        impact,
        turnover,
        ic,
        score,
        signal
    ):
        return {
            ("market", "symbol"): symbol,
            ("market", "price"): float(price),
            ("market", "return"): float(ret),
            ("market", "volume"): float(volume),
            ("alpha", "ts"): float(alpha_ts),
            ("alpha", "xs"): float(alpha_xs),
            ("alpha", "pure"): float(alpha_pure),
            ("alpha", "beta"): float(beta),
            ("alpha", "residual"): float(residual),
            ("risk", "volatility"): float(volatility),
            ("risk", "sharpe"): float(sharpe),
            ("risk", "drawdown"): float(drawdown),
            ("risk", "cvar"): float(cvar),
            ("risk", "decorrelation"): float(decorrelation),
            ("risk", "wiggle"): float(wiggle),
            ("transform", "rank"): float(rank),
            ("transform", "zscore"): float(zscore),
            ("transform", "winsor"): float(winsor),
            ("transform", "tanh"): float(tanh),
            ("transform", "detrend"): float(detrend),
            ("portfolio", "weight"): float(weight),
            ("portfolio", "risk_parity"): float(risk_parity),
            ("portfolio", "kelly"): float(kelly),
            ("portfolio", "entropy"): float(entropy),
            ("market_structure", "regime"): regime,
            ("market_structure", "liq_adj_vol"): float(liquidity_adj_vol),
            ("execution", "slippage"): float(slippage),
            ("execution", "impact"): float(impact),
            ("execution", "turnover"): float(turnover),
            ("intel", "ic"): float(ic),
            ("decision", "score"): float(score),
            ("decision", "signal"): signal,
        }


class QuantX:
    """
    ==========================================================
    QUANT XPERT X
    ==========================================================
    Full institutional-style research engine
    OUTPUT:
        MultiIndex tensor report
    ==========================================================
    """

    def __init__(self, symbols):
        self.symbols = symbols if symbols is not None else self.getSymbols(6)
        if self.symbols is None:
            raise ValueError("getSymbols() returned None")
        self.assembler = SymbolAssembler()

    def pad_nan(self, x, target_len):
        if x is None:
            return [np.nan] * target_len
        if not isinstance(x, (list, np.ndarray)):
            x = list(x)
        if len(x) > target_len:
            x = x[:target_len]
        if len(x) < target_len:
            x = x + [np.nan] * (target_len - len(x))
        return x

    def getSymbols(self, howMany):
        url = "https://api.binance.us/api/v3/exchangeInfo"
        info = requests.get(url).json()
        if "symbols" not in info:
            raise Exception(f"Binance error: {info}")
        tickers = list(
            dict.fromkeys(
                s["symbol"].strip().upper()
                for s in info["symbols"]
                if s["status"] == "TRADING" and s["quoteAsset"] == "USDT"
            )
        )[:howMany]
        return tickers

    def getSPV(self, sym, interval="1m"):
        BASE = "https://api.binance.us/api/v3/klines"
        if not isinstance(sym, str):
            return [], []
        sym = sym.strip().upper()
        if not sym.isalnum():
            raise ValueError(f"Bad symbol: {sym}")
        params = {"symbol": sym, "interval": interval}
        responsePV = requests.get(BASE, params=params).json()
        if not isinstance(responsePV, list):
            raise ValueError(responsePV)
        closes = [float(c[4]) for c in responsePV]
        volumes = [float(c[5]) for c in responsePV]
        return closes, volumes

    def data(self):
        prices_dict = {}
        volumes_dict = {}
        valid_symbols = []
        max_len = 0
        for sym in self.symbols:
            try:
                closes, volumes = self.getSPV(sym)
                if len(closes) == 0:
                    continue
                prices_dict[sym] = closes
                volumes_dict[sym] = volumes
                valid_symbols.append(sym)
                max_len = max(max_len, len(closes))
            except Exception:
                continue
        self.symbols = valid_symbols
        for k in prices_dict:
            prices_dict[k] = self.pad_nan(prices_dict[k], max_len)
        for k in volumes_dict:
            volumes_dict[k] = self.pad_nan(volumes_dict[k], max_len)
        P = pd.DataFrame.from_dict(prices_dict, orient="index").T
        V = pd.DataFrame.from_dict(volumes_dict, orient="index").T
        P = P.interpolate(limit=5)
        V = V.interpolate(limit=5)
        return P, V

    def analyze_symbol(self, P, V, R, s):
        EPS = 1e-12
        R = R.fillna(0.0)
        V = V.fillna(0.0)
        R = R.clip(-1, 1)
        V = V.clip(lower=EPS)
        price = P[s]
        volume = V[s]
        ret = R[s]
        market = R.mean(axis=1).rolling(10).mean()

        ts = (
            ret.rolling(20).mean().iloc[-1]
            if len(ret) >= 20
            else ret.mean()
        )
        xs = market.iloc[-1] if len(market) > 0 else 0.0
        pure = ts * 0.5
        beta = 1.0
        alpha = ret.iloc[-1] - beta * (
            market.iloc[-1] if len(market) > 0 else 0.0
        )
        residual = ret - ret.mean()

        vol = ret.std() if len(ret) > 1 else 0.0
        sharpe = (
            (ret.mean() / (vol + EPS) * np.sqrt(252))
            if len(ret) > 1
            else 0.0
        )
        dd = (
            (price / price.cummax() - 1).min()
            if len(price) > 0
            else 0.0
        )
        cvar = ret.quantile(0.05) if len(ret) > 0 else 0.0
        decor = 0.0
        wiggle = vol

        rank = 0.0
        z = 0.0
        win = (
            ret.clip(-0.1, 0.1).iloc[-1]
            if len(ret) > 0
            else 0.0
        )
        tanh = np.tanh(ret.iloc[-1]) if len(ret) > 0 else 0.0
        detrend = ret.iloc[-1] - ret.mean() if len(ret) > 0 else 0.0

        weight = 1.0 / len(self.symbols) if self.symbols else 0.0
        rp = weight
        kelly = 0.0
        entropy = 0.0

        slip = 0.0
        impact = 0.0
        liq_adj = vol
        turnover = np.abs(weight) * 0.01

        regime = "NEUTRAL"

        future_returns = ret.shift(-1).fillna(0)
        ic = 0.0

        score = alpha + sharpe + ic - cvar - abs(dd)

        if score > 0.5:
            signal = "BUY"
        elif score < -0.5:
            signal = "SELL"
        else:
            signal = "HOLD"

        return self.assembler.assemble_symbol_report(
            symbol=s,
            price=price.iloc[-1] if len(price) > 0 else 0.0,
            ret=ret.iloc[-1] if len(ret) > 0 else 0.0,
            volume=volume.iloc[-1] if len(volume) > 0 else 0.0,
            alpha_ts=ts,
            alpha_xs=xs,
            alpha_pure=pure,
            beta=beta,
            residual=np.mean(residual) if len(residual) > 0 else 0.0,
            volatility=vol,
            sharpe=sharpe,
            drawdown=dd,
            cvar=cvar,
            decorrelation=decor,
            wiggle=wiggle,
            rank=rank,
            zscore=z,
            winsor=win,
            tanh=tanh,
            detrend=detrend,
            weight=weight,
            risk_parity=rp,
            kelly=kelly,
            entropy=entropy,
            regime=regime,
            liquidity_adj_vol=liq_adj,
            slippage=slip,
            impact=impact,
            turnover=turnover,
            ic=ic,
            score=score,
            signal=signal,
        )

    def run(self):
        P, V = self.data()
        R = P.pct_change().fillna(0)
        rows = []
        for s in self.symbols:
            row = self.analyze_symbol(P, V, R, s)
            rows.append(row)
        df = pd.DataFrame(rows)
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        return df
'''


def main():
    if not SRC_DIR.exists():
        print(f"ERROR: Source directory '{SRC_DIR}' not found.")
        return

    # 1. Clean copy
    print("[1/9] Copying quantX → QuantXpert …")
    if DST_DIR.exists():
        shutil.rmtree(DST_DIR)
    copy_tree(SRC_DIR, DST_DIR)

    # 2. Delete root folders/files
    print("[2/9] Deleting unused root folders and files …")
    for sub in DELETE_ROOT_FOLDERS:
        target = DST_DIR / sub
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
            print(f"   Removed {target}")
    for fname in DELETE_ROOT_FILES:
        target = DST_DIR / fname
        if target.exists():
            target.unlink()
            print(f"   Removed {target}")

    # 3. Strip 0-suffix definitions
    print("[3/9] Stripping '0'-suffixed top-level definitions …")
    for py_file in DST_DIR.rglob("*.py"):
        original = safe_read(py_file)
        cleaned = remove_toplevel_zero_definitions(original)
        if cleaned != original:
            py_file.write_text(cleaned, encoding="utf-8")

    # 4. Inline SplitEngine fallback (CRITICAL FIX)
    print("[4/9] Inlining SplitEngine fallback …")
    pipeline = DST_DIR / "qXengine" / "backtest" / "pipeline_backtest.py"
    if pipeline.exists():
        text = safe_read(pipeline)
        text = inline_split_fallback(text)
        pipeline.write_text(text, encoding="utf-8")

    # 5. Remove other leftover *_0 references
    print("[5/9] Removing other leftover *_0 references …")
    for py_file in DST_DIR.rglob("*.py"):
        original = safe_read(py_file)
        cleaned = remove_zero_fallback_refs(original)
        if cleaned != original:
            py_file.write_text(cleaned, encoding="utf-8")

    # 6. Modify pipeline_backtest.py
    print("[6/9] Modifying pipeline_backtest.py …")
    if pipeline.exists():
        text = safe_read(pipeline)
        text = change_storage_base_dir(text)
        text = inject_pickle_dump(text)
        pipeline.write_text(text, encoding="utf-8")

    # 7. Replace core/system.py
    print("[7/9] Replacing core/system.py …")
    system_py = DST_DIR / "core" / "system.py"
    system_py.write_text(CLEAN_SYSTEM_PY, encoding="utf-8")

    # 8. Delete empty .py files BUT NEVER delete __init__.py
    print("[8/9] Removing empty Python files (preserving __init__.py) …")
    removed = 0
    for py_file in list(DST_DIR.rglob("*.py")):
        if py_file.name == "__init__.py":
            continue
        text = safe_read(py_file).strip()
        if not text:
            py_file.unlink()
            removed += 1
    print(f"   Removed {removed} empty files")

    # 9. Delete ALL files in qXengine/backtest EXCEPT the 3 keepers
    print("[9/9] Pruning qXengine/backtest/ to coverage files only …")
    bt_dir = DST_DIR / "qXengine" / "backtest"
    keep = {"__init__.py", "e2ebacktest.py", "pipeline_backtest.py"}
    for f in list(bt_dir.iterdir()):
        if f.name not in keep:
            if f.is_dir():
                shutil.rmtree(f)
            else:
                f.unlink()
            print(f"   Deleted qXengine/backtest/{f.name}")

    print("\n✅ QuantXpert/ is ready.")
    print("\nFinal structure:")
    for py_file in sorted(DST_DIR.rglob("*.py")):
        rel = py_file.relative_to(DST_DIR)
        print(f"   {rel}")

    print("\nNext steps:")
    print("   cd QuantXpert")
    print("   git init")
    print("   git add -A")
    print('   git commit -m "QuantXpert clean: coverage-only, split pkls, pruned system.py"')


if __name__ == "__main__":
    main()