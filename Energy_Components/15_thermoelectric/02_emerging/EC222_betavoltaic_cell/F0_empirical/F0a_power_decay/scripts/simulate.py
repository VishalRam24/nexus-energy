"""EC222 Betavoltaic Cell F0a - optional Plotly decay report."""
import os
import numpy as np
from model import BetavoltaicPowerDecay


def main():
    c = BetavoltaicPowerDecay()
    ts = np.linspace(0, 500, 60)
    ps = [c.power_nW(t) for t in ts]
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts, y=ps, mode="lines", name="power"))
        fig.update_layout(title="EC222 Betavoltaic F0a output power vs time",
                          xaxis_title="time (years)", yaxis_title="power (nW)")
        out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
        fig.write_html(out)
        print("wrote", out)
    except Exception as e:
        print("plotly unavailable, text dump:", e)
        for t, v in zip(ts[::10], ps[::10]):
            print(round(float(t), 1), round(v, 4))


if __name__ == "__main__":
    main()
