from .BaseStrategy import BaseStrategy


class FallbackStrategy(BaseStrategy):

    def __init__(self, registry, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.registry = registry

    def run(self):

        fallback_name = self.get_cfg(
            "fallback_strategy",
            "pair_trading_fallback"
        )

        if fallback_name not in self.registry:

            return self.build_result(

                signal={},

                score=0.0,

                metadata={

                    "error":
                        f"Unknown fallback strategy: {fallback_name}"
                }
            )

        StrategyClass = self.registry[fallback_name]

        strategy = StrategyClass(
            data=self.data,
            cfg=self.cfg,
            runtime_cfg=self.runtime_cfg,
            factor_engine=self.factor_engine,
            logger=self.logger,
        )

        return strategy.run()