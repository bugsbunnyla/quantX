from .StrategyCharts import StrategyChart,QXDashboard
class StrategyResult:

    def __init__(self, name, data, metrics, signals, chart):
        self.chart = chart
        self.name = name
        self.data = data
        self.metrics = metrics
        self.signals = signals

    def add(self,tab):
        QXDashboard.get().addData(
            tab,
            self
        )
