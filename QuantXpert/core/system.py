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
