"""Optional Plotly report for EC098 Organic Rankine Cycle (ORC), R245fa F0a."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict import ComponentModel


def main():
    m = ComponentModel()
    plr = np.linspace(0.0, 1.0, 101)
    eta = [m.predict({"part_load_ratio": x})["electrical_efficiency"] for x in plr]
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly not installed; eta(PLR=1) =", eta[-1])
        return
    fig = go.Figure(go.Scatter(x=plr, y=eta, mode="lines", name="eta"))
    fig.update_layout(title="EC098 Organic Rankine Cycle (ORC), R245fa - F0a efficiency curve",
                      xaxis_title="Part-load ratio", yaxis_title="Electrical efficiency")
    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
