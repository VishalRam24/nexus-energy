"""
EC047 -- Thin-Film CIGS PV -- F2a Physics-Lumped
Optional Plotly report: I-V/P-V curves, MPP vs irradiance/temperature, and a
diurnal thermal-ODE transient. Plotly import is guarded; absence won't crash.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def build_report(out_html=None):
    cm = ComponentModel()
    m = cm._model

    # I-V / P-V at a few irradiances
    iv_data = {}
    for G in [200, 600, 1000]:
        V, I, P = m.iv_curve(G, 298.15, n=200)
        iv_data[G] = (V, I, P)

    # MPP vs temperature
    temps = np.linspace(0, 70, 30)
    p_vs_T = [m.mpp(1000.0, t + 273.15)["p_mp"] for t in temps]

    # Diurnal transient (bell-shaped irradiance, warming ambient)
    def G_diurnal(t):
        h = t / 3600.0
        return max(0.0, 1000.0 * np.sin(np.pi * (h - 6) / 12)) if 6 <= h <= 18 else 0.0

    def Tamb(t):
        h = t / 3600.0
        return 18.0 + 10.0 * np.sin(np.pi * (h - 8) / 14) if 0 <= h <= 24 else 18.0

    tr = m.simulate(G_diurnal, Tamb, wind=1.5, dt=300.0, duration_s=86400.0)

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        print(f"[simulate] Plotly unavailable ({e}); skipping HTML, printing summary.")
        print(f"  STC MPP: {m.mpp(1000.0,298.15)['p_mp']:.1f} W")
        print(f"  Peak transient power: {np.max(tr['power_W']):.1f} W, "
              f"max T_cell: {np.max(tr['T_cell_c']):.1f} C")
        return None

    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "I-V curves", "P-V curves", "MPP power vs cell temperature",
        "Diurnal thermal-ODE transient"))
    for G, (V, I, P) in iv_data.items():
        fig.add_trace(go.Scatter(x=V, y=I, name=f"I-V {G} W/m2"), row=1, col=1)
        fig.add_trace(go.Scatter(x=V, y=P, name=f"P-V {G} W/m2"), row=1, col=2)
    fig.add_trace(go.Scatter(x=temps, y=p_vs_T, name="P_mp(T)"), row=2, col=1)
    fig.add_trace(go.Scatter(x=tr["t"] / 3600.0, y=tr["power_W"], name="Power [W]"),
                  row=2, col=2)
    fig.add_trace(go.Scatter(x=tr["t"] / 3600.0, y=tr["T_cell_c"], name="T_cell [C]",
                             yaxis="y2"), row=2, col=2)
    fig.update_layout(title="EC047 CIGS PV F2a -- Physics-Lumped", height=800)

    if out_html is None:
        out_html = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out_html)
    print(f"[simulate] wrote {out_html}")
    return out_html


if __name__ == "__main__":
    build_report()
