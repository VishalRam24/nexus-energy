"""F0a optional Plotly report for EC174 Instrument Transformer (CT/PT). Safe if Plotly absent."""
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
    xs = np.linspace(0.05, 2.0, 50)
    ys = [m.predict({"load_fraction": float(x)})["ratio_error_pct"] for x in xs]
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines+markers"))
    fig.update_layout(title="EC174 Instrument Transformer (CT/PT) — F0a ratio_error_pct",
                      xaxis_title="load_fraction", yaxis_title="ratio_error_pct")
    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
