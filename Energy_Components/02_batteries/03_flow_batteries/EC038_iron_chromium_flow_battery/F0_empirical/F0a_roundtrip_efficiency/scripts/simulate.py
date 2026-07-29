"""Optional Plotly report for Iron-Chromium Flow Battery (ICFB) (EC038) F0a. Wrapped so absence of plotly is harmless."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import ComponentModel


def main():
    m = ComponentModel()
    try:
        import numpy as np
        import plotly.graph_objects as go
    except Exception as e:  # noqa: BLE001
        print(f"[simulate] plotly/numpy unavailable ({e}); skipping report.")
        return
    xs = np.linspace(0, 5, 200)
    ys = m.curve.efficiency(xs)
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines", name="round-trip eff"))
    fig.update_layout(title="Iron-Chromium Flow Battery (ICFB) (EC038) F0a round-trip efficiency",
                      xaxis_title="C-rate (1/h)", yaxis_title="efficiency")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "simulation_report.html")
    fig.write_html(out)
    print(f"[simulate] wrote {out}")


if __name__ == "__main__":
    main()
