"""Optional Plotly report for EDLC Supercapacitor (EC041) F0a. Wrapped so absence of plotly is harmless."""
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
    vs = np.linspace(m.curve.V_min, m.curve.V_max, 200)
    es = m.curve.energy(vs)
    fig = go.Figure(go.Scatter(x=vs, y=es, mode="lines", name="stored energy"))
    fig.update_layout(title="EDLC Supercapacitor (EC041) F0a stored energy 0.5CV^2",
                      xaxis_title="voltage (V)", yaxis_title="energy (J)")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "simulation_report.html")
    fig.write_html(out)
    print(f"[simulate] wrote {out}")


if __name__ == "__main__":
    main()
