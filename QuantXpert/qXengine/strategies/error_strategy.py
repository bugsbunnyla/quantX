# strategies/error_strategy.py

from .BaseStrategy import BaseStrategy


class ErrorStrategy(BaseStrategy):

    def __init__(self, name, error, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.strategy_name = name
        self.error = error

    def run(self):

        return self.build_result(

            signal={},

            score=0.0,

            metadata={

                "strategy": self.strategy_name,
                "error": str(self.error),
                "status": "failed"
            }
        )