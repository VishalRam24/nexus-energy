"""Optional Plotly report for EC107 Micro-CHP (Stirling-based) F0a CHP."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import ComponentModel


def main():
    m = ComponentModel()
    plr = np.linspace(0.0, 1.0, 101)
    ee = [m.predict({"part_load_ratio": x})["electrical_efficiency"] for x in plr]
    et = [m.predict({"part_load_ratio": x})["thermal_efficiency"] for x in plr]
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly not installed; eta_el(1)=", ee[-1], "eta_th(1)=", et[-1])
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plr, y=ee, mode="lines", name="eta_el"))
    fig.add_trace(go.Scatter(x=plr, y=et, mode="lines", name="eta_th"))
    fig.update_layout(title="EC107 Micro-CHP (Stirling-based) - F0a part-load efficiencies",
                      xaxis_title="Part-load ratio", yaxis_title="Efficiency")
    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
