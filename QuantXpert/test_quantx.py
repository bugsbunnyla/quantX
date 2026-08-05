#test_quants.py
from qXengine.StrategyCharts import QXDashboard
from qXengine.StrategyInit import StrategyInit


# =====================================================
# BACKTEST ENGINE (RESEARCH LAYER)
# =====================================================
def run_backtest(strategy_results):

    results = []

    for res in strategy_results:

        try:

            signals = res.signals
            pnl_series = {}

            # Cross-sectional strategy
            if isinstance(signals, dict):

                for sym, sig in signals.items():

                    if sym not in res.data:
                        continue

                    df = res.data[sym]

                    if "ret" not in df.columns:
                        continue

                    pnl = (
                        sig.fillna(0) * df["ret"].fillna(0)
                    ).cumsum()

                    pnl_series[sym] = pnl

            # Single stream strategy
            else:
                pnl_series["strategy"] = signals.fillna(0).cumsum()

            results.append({
                "name": res.name,
                "pnl": pnl_series,
                "metrics": res.metrics
            })

        except Exception as e:
            print(f"[BACKTEST] {res.name} failed: {e}")

    return results
from qXengine.qxEngine import QuantXEngine
from qXengine.StrategyCharts import QXDashboard


def main():

    # 1. INIT ENGINE
    engine = QuantXEngine()

    # 2. LOAD DATA (your pickle or loader)
    

# =====================================================
# MAIN TEST RUNNER
# =====================================================
    result = StrategyInit.run(
        interval="4y",
        render_charts=False
    )

    print("[ENGINE]", result.keys())
    print("[STRATEGIES]", len(result["strategies"]))


# =====================================================
# BACKTEST EXECUTION (NOW HERE)
# =====================================================
    backtests = run_backtest(result["strategies"])

    print("[BACKTEST]", len(backtests))


# =====================================================
# DASHBOARD RENDER
# =====================================================
    # 3. RUN STRATEGIES
    #results = engine.qxStrategyList(data)

    #print(f"[MAIN] strategies executed: {len(results)}")

    # 4. GET DASHBOARD SNAPSHOT
    dashboard = QXDashboard.get()
    #dashboard.display()
    #dashboard.displayJSON()
    dashboard.displayFrame()
    # 5. INSPECT OUTPUT (CRITICAL DEBUG STEP)
    #print("\n========== DASHBOARD KEYS ==========")
    #print(dashboard.keys())


    # 6. OPTIONAL: PRINT SAMPLE STRUCTURE
    # add data first (your pipeline already does this)
    # dashboard.addData(...)

    figures = dashboard.render()

    #  THIS IS THE MISSING PIECE
    #for tab_name, fig_list in figures.items():
    #    print(f"\n=== {tab_name.upper()} ===")

    #   for i, fig in enumerate(fig_list):
    #         print(f"Rendering figure {i+1}")
    #        fig.show()
    

if __name__ == "__main__":
    main()
