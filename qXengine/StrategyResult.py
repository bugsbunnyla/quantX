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
    
    def keys(self):
        return self.data.keys()

    def __getitem__(self, key):
        return self.data[key]

    def get(self, key, default=None):
        return self.data.get(key, default)
