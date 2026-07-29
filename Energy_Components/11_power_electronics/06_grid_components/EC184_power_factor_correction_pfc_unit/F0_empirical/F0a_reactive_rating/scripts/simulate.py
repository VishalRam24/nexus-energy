"""F0a optional Plotly report for EC184 Power Factor Correction Unit. Safe if Plotly absent."""
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
    xs = np.linspace(m.q_min, m.q_max, 50)
    ys = [m.predict({"q_demand": float(x)})["loss"] for x in xs]
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines+markers"))
    fig.update_layout(title="EC184 Power Factor Correction Unit — F0a converter loss",
                      xaxis_title="Q demand", yaxis_title="loss")
    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
