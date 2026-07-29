"""
EC028 -- Lead-Acid Battery -- F0a optional Plotly report.
Run:  python3 scripts/simulate.py   (no-op if plotly absent)
"""

import json
import os

import numpy as np

from model import LeadAcidBatteryF0a


def main():
    p = os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")
    m = LeadAcidBatteryF0a(json.load(open(p)))
    cs = np.linspace(0.0, m.crate_max, 50)
    etas = m.round_trip_efficiency(cs)
    try:
        import plotly.graph_objects as go
        fig = go.Figure(go.Scatter(x=cs, y=etas, mode="lines"))
        fig.update_layout(title="EC028 F0a round-trip efficiency vs C-rate",
                          xaxis_title="C-rate", yaxis_title="round-trip efficiency")
        out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
        fig.write_html(out)
        print("wrote", out)
    except Exception as e:
        print("plotly unavailable, skipping report:", e)
        for c, e2 in zip(cs[::10], etas[::10]):
            print(f"  C={c:.2f}  eta={e2:.4f}")


if __name__ == "__main__":
    main()
