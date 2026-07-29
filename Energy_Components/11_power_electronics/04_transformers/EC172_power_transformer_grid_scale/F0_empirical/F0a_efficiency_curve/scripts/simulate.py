"""OPTIONAL Plotly efficiency-curve report for EC172 Power Transformer (Grid-Scale)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import EfficiencyCurveModel  # noqa: E402


def main():
    import numpy as np
    m = EfficiencyCurveModel()
    lf = np.linspace(0.0, float(m.load_fraction[-1]), 200)
    eff = m.efficiency_at(lf)
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly not installed; skipping HTML report.")
        for x in (0.1, 0.3, 0.5, 0.7, 1.0):
            print("  load=%.2f  eff=%.4f" % (x, float(m.efficiency_at(x))))
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lf * 100, y=eff * 100, mode="lines",
                             name="efficiency"))
    fig.add_trace(go.Scatter(x=m.load_fraction * 100, y=m.efficiency * 100,
                             mode="markers", name="breakpoints"))
    fig.update_layout(title="EC172 Power Transformer (Grid-Scale) -- F0a efficiency vs load",
                      xaxis_title="Load (% of rated)",
                      yaxis_title="Efficiency (%)")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                       "simulation_report.html")
    fig.write_html(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
