class WalkForward:

    """
    WALK FORWARD BACKTEST

    splits data into rolling train/test windows
    """

    def run(self, data, window=50):
        results = []

        for i in range(window, len(data)):
            train = data[i-window:i]
            test = data[i]

            results.append(test)

        return results