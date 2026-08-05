class StrategyBacktestEngine:

    def run(self, strategy_results):

        outputs = []

        for res in strategy_results:

            pnl = {}

            if isinstance(res.signals, dict):

                for sym, sig in res.signals.items():

                    df = res.data[sym]

                    pnl[sym] = (sig * df["ret"]).cumsum()

            else:

                pnl["strategy"] = (
                    res.signals * 1.0
                ).cumsum()

            outputs.append({
                "name": res.name,
                "pnl": pnl
            })

        return outputs
