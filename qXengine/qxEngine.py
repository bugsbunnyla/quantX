import pandas as pd

from .StrategyConfig import STRATEGY_CONFIG
from .StrategyResult import StrategyResult

# strategies
from .strategies.alpha import AlphaStrategy
from .strategies.beta import BetaNeutralStrategy
from .strategies.momentum import UMDMomentum, TimeSeriesMomentum
from .strategies.reversal import STREV, IntradayReversal
from .strategies.pairs import PairTrading
from .strategies.pair_trading_fallback import PairTradingFallback
from .strategies.volatility import VolatilityStrategy
from .strategies.correlation import CorrelationStrategy
from .strategies.correlation_fallback import CorrelationFallback
from .strategies.dispersion import DispersionStrategy
from .strategies.breadth import BreadthStrategy
from .strategies.forecast import ForecastStrategy
from .strategies.error_strategy import ErrorStrategy
from .strategies.intraday import IntradayStrategy
from .strategies.industry_momentum import IndustryMomentumStrategy


# =====================================================
# CONTEXT
# =====================================================
class Context:
    def __init__(self, data, cfg, runtime_cfg, factor_engine=None, logger=None):
        self.data = data
        self.cfg = cfg
        self.runtime_cfg = runtime_cfg
        self.factor_engine = factor_engine
        self.logger = logger


# =====================================================
# RESOLVER
# =====================================================
class StrategyResolver:

    STRATEGY_MAP = {
        "AlphaStrategy": AlphaStrategy,
        "BetaNeutralStrategy": BetaNeutralStrategy,
        "UMDMomentum": UMDMomentum,
        "TimeSeriesMomentum": TimeSeriesMomentum,
        "STREV": STREV,
        "IntradayReversal": IntradayReversal,
        "PairTrading": PairTrading,
        "CorrelationStrategy": CorrelationStrategy,
        "DispersionStrategy": DispersionStrategy,
        "BreadthStrategy": BreadthStrategy,
        "VolatilityStrategy": VolatilityStrategy,
        "ForecastStrategy": ForecastStrategy,
        "IndustryMomentumStrategy": IndustryMomentumStrategy,
        "IntradayStrategy": IntradayStrategy,
        "PairTradingFallback": PairTradingFallback,
        "CorrelationFallback": CorrelationFallback,
    }

    @staticmethod
    def resolve(name):
        return StrategyResolver.STRATEGY_MAP.get(name, ErrorStrategy)


# =====================================================
# ENGINE
# =====================================================
class QuantXEngine:

    def __init__(self):
        self.fe = None  # kept for backward compatibility if strategies expect it
        self.results = []
        self.strategy = []

    # =====================================================
    # MAIN PIPELINE (CLEAN)
    # =====================================================
    def qxStrategyList(self, data, interval="4y"):

        runtime_cfg = {
            "data": data,
            "interval": interval,
            "benchmark": STRATEGY_CONFIG.get("data", {}).get("benchmark", "SPY"),
        }

        
        enabled = STRATEGY_CONFIG.get("strategies", {})

        for name, params in enabled.items():

            if not params.get("enabled", False):
                continue

            StrategyClass = StrategyResolver.resolve(name)

            context = Context(
                data=data,
                cfg=params,
                runtime_cfg=runtime_cfg,
                factor_engine=self.fe,
                logger=None
            )

            # =================================================
            # EXECUTION LAYER
            # =================================================
            try:
                strategy = StrategyClass(context)
                result = strategy.run()

                # attach metadata if needed
                result.add(params.get("tab"))
 
                self.results.append(result)
                self.strategy.append(strategy)
                print(f"[QuantX] {name} OK")

            except Exception as e:
                self.results.append(
                    ErrorStrategy(name=name, error=str(e))
                )
                continue

        return self.results

    def qxStrategySelect(self, strategy_names, data, interval="4y"):
      """
      Execute one or more strategies by name.

      Parameters
      ----------
      strategy_names : str | Iterable[str]
        Strategy name or collection of strategy names.
      data : Any
        Market data.
      interval : str
        Runtime interval.

      Returns
      -------
      list
        List of strategy results.
      """

      runtime_cfg = {
        "data": data,
        "interval": interval,
        "benchmark": STRATEGY_CONFIG.get("data", {}).get("benchmark", "SPY"),
      }

      # Accept a single string or an iterable
      if isinstance(strategy_names, str):
        strategy_names = [strategy_names]

      strategies_cfg = STRATEGY_CONFIG.get("strategies", {})

      for name in strategy_names:

        params = strategies_cfg.get(name)

        if params is None:
            self.results.append(
                ErrorStrategy(
                    name=name,
                    error=f"Strategy '{name}' not found in STRATEGY_CONFIG."
                )
            )
            continue

        if not params.get("enabled", False):
            continue

        StrategyClass = StrategyResolver.resolve(name)

        context = Context(
            data=data,
            cfg=params,
            runtime_cfg=runtime_cfg,
            factor_engine=self.fe,
            logger=None,
        )

        # =================================================
        # EXECUTION LAYER
        # =================================================
        try:
            strategy = StrategyClass(context)
            result = strategy.run()

            result.add(params.get("tab"))
            self.results.append(result)
            self.strategy.append(strategy)

            print(f"[QuantX] {name} OK")

        except Exception as e:
            self.results.append(
                ErrorStrategy(
                    name=name,
                    error=str(e)
                )
            )

      return self.results