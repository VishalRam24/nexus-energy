"""F0a optional Plotly report for EC179 Wound Rotor Synchronous Generator. Safe if Plotly absent."""
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
    ys = [m.predict({"load_fraction": float(x)})["efficiency"] for x in xs]
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines+markers"))
    fig.update_layout(title="EC179 Wound Rotor Synchronous Generator — F0a efficiency",
                      xaxis_title="load_fraction", yaxis_title="efficiency")
    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
