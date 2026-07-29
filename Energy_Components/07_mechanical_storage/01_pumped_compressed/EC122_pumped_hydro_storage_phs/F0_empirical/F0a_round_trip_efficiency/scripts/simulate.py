"""Optional Plotly report for EC122 F0a RTE curve."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import ComponentModel


def main():
    m = ComponentModel()
    fr = np.linspace(0.1, 1.0, 50)
    rte = m.curve.round_trip_efficiency(fr)
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly not installed; RTE @ rated =", m.predict({"power_fraction": 1.0}))
        return
    fig = go.Figure(go.Scatter(x=fr, y=rte, mode="lines"))
    fig.update_layout(title=f"{m.component_id} F0a Round-Trip Efficiency vs Part-Load",
                      xaxis_title="Power fraction", yaxis_title="RTE")
    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
