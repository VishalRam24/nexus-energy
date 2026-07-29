"""EC216 TEG F0a - optional Plotly efficiency curve report."""
import os
import numpy as np
from model import TEGEfficiencyCurve


def main():
    c = TEGEfficiencyCurve()
    ts = np.linspace(50, 300, 60)
    effs = [c.efficiency(t) for t in ts]
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts, y=effs, mode="lines", name="efficiency"))
        fig.update_layout(title="EC216 TEG F0a efficiency vs hot-side temperature",
                          xaxis_title="T_hot (degC)", yaxis_title="efficiency")
        out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
        fig.write_html(out)
        print("wrote", out)
    except Exception as e:
        print("plotly unavailable, text dump:", e)
        for t, e2 in zip(ts[::10], effs[::10]):
            print(round(float(t), 1), round(e2, 4))


if __name__ == "__main__":
    main()
