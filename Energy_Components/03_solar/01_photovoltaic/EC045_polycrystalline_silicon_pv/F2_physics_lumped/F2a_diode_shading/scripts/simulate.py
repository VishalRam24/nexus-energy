"""
EC045 -- Poly-Si PV -- F2a Physics-Lumped : Plotly simulation report.
Generates simulation_report.html (I-V/P-V curves, partial-shading curve,
and a diurnal thermal+power dynamic run). Plotly import is optional.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAVE_PLOTLY = True
except Exception:
    HAVE_PLOTLY = False


def main():
    cm = ComponentModel()
    m = cm._model

    # 1. I-V / P-V at STC and reduced irradiance
    curves = {}
    for G in [1000.0, 600.0, 300.0]:
        V, I, P = m.iv_curve(G, 25.0, n_points=200)
        curves[G] = (V, I, P)

    # 2. Partial shading curve
    sh = m.mpp_partial_shade(1000.0, 25.0, 0.6)

    # 3. Diurnal dynamic run (bell-shaped irradiance, warming ambient)
    day = 6 * 3600.0
    def G_t(t):
        x = (t - day / 2) / (day / 2)
        return max(1000.0 * (1 - x * x), 0.0)
    def Tamb_t(t):
        return 18.0 + 10.0 * np.sin(np.pi * t / day)
    dyn = m.simulate(G_t, T_amb=Tamb_t, wind=1.5, dt=300.0, duration_s=day)

    print(f"STC P_mp = {curves[1000.0][2].max():.1f} W")
    print(f"60% shade on one substring: P_mp = {sh['p_mp']:.1f} W")
    print(f"Diurnal peak T_cell = {dyn['T_cell'].max():.1f} C, "
          f"peak P_mp = {dyn['p_mp'].max():.1f} W")

    if not HAVE_PLOTLY:
        print("Plotly not available; skipping HTML report.")
        return

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("I-V curves (T=25C)", "P-V curves (T=25C)",
                        "Partial shading P-V (60% on 1 substring)",
                        "Diurnal thermal + power dynamics"),
        specs=[[{}, {}], [{}, {"secondary_y": True}]],
    )
    for G, (V, I, P) in curves.items():
        fig.add_trace(go.Scatter(x=V, y=I, name=f"{G:.0f} W/m2"), row=1, col=1)
        fig.add_trace(go.Scatter(x=V, y=P, name=f"{G:.0f} W/m2", showlegend=False),
                      row=1, col=2)
    fig.add_trace(go.Scatter(x=sh["V_curve"], y=sh["P_curve"], name="60% shade"),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=dyn["t"] / 3600, y=dyn["T_cell"], name="T_cell (C)"),
                  row=2, col=2, secondary_y=False)
    fig.add_trace(go.Scatter(x=dyn["t"] / 3600, y=dyn["p_mp"], name="P_mp (W)"),
                  row=2, col=2, secondary_y=True)
    fig.update_layout(title="EC045 Poly-Si PV -- F2a Physics-Lumped", height=800)

    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print(f"Report written to {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
