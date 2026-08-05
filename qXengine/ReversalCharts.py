import plotly.graph_objects as go
Class ReversalCharts:
  def __init__(self):
      pass

  def qxReversalCharts(self, data):

    signals = self._enrichReversal(data)

    import plotly.graph_objects as go

    for sym, sig in signals.items():

        fig = go.Figure()

        price = data[sym]["close"]

        fig.add_trace(
            go.Scatter(y=price, name="Price")
        )

        fig.add_trace(
            go.Scatter(y=sig, name="Reversal Signal")
        )

        # highlight reversal spikes
        spikes = sig[abs(sig) > sig.std() * 2]

        fig.add_trace(
            go.Scatter(
                x=spikes.index,
                y=spikes.values,
                mode="markers",
                marker=dict(size=10, color="red"),
                name="Reversal Events"
            )
        )

        fig.update_layout(
            title=f"Reversal Chart: {sym}",
            template="plotly_dark"
        )

        fig.show()
