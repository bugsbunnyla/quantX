import plotly.graph_objects as go
Class MomentumCharts:
  def __init__(self):
      pass
def qxMomentumCharts(self, data):

    signals = self._enrichMomentum(data)

    import plotly.graph_objects as go

    for sym, sig in signals.items():

        fig = go.Figure()

        price = data[sym]["close"]

        fig.add_trace(
            go.Scatter(y=price, name="Price")
        )

        fig.add_trace(
            go.Scatter(y=sig, name="Momentum Signal")
        )

        # highlight strong momentum
        strong = sig[abs(sig) > sig.std() * 2]

        fig.add_trace(
            go.Scatter(
                x=strong.index,
                y=strong.values,
                mode="markers",
                marker=dict(size=10, color="green"),
                name="Momentum Events"
            )
        )

        fig.update_layout(
            title=f"Momentum Chart: {sym}",
            template="plotly_dark"
        )

        fig.show()
