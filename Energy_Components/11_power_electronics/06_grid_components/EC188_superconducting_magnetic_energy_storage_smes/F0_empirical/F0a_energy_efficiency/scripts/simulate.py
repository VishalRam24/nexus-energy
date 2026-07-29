"""F0a optional Plotly report for EC188 Superconducting Magnetic Energy Storage (SMES). Safe if Plotly absent."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import ComponentModel  # noqa: E402


def main():
    m = ComponentModel()
    try:
        import plotly.graph_objects as go
    except Exception:
        print("Plotly not available; skipping HTML report.")
        return
    xs = np.linspace(0.1, 1.0, 50)
    ys = [m.predict({"power_fraction": float(x)})["round_trip_efficiency"] for x in xs]
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines+markers"))
    fig.update_layout(title="EC188 Superconducting Magnetic Energy Storage (SMES) — F0a round_trip_efficiency",
                      xaxis_title="power_fraction", yaxis_title="round_trip_efficiency")
    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
