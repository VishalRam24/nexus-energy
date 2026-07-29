"""
EC051 -- DSSC F2a -- Plotly simulation report (optional).

Generates an interactive HTML report: I-V/P-V curves across irradiance, the
low-light efficiency advantage, temperature dependence of Voc, and the lumped
thermal warm-up transient. Plotly import is guarded so absence does not crash.

Run:  python3 scripts/simulate.py
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

    irradiances = [100.0, 300.0, 600.0, 1000.0]

    if not HAVE_PLOTLY:
        print("Plotly not installed -- printing summary table instead.")
        print(f"{'G[W/m2]':>9} {'Voc[V]':>8} {'Isc[mA]':>9} {'Pmp[mW]':>9} {'FF':>6} {'eta[%]':>7}")
        for G in irradiances:
            r = m.iv_curve(G, 298.15)
            print(f"{G:9.0f} {r['Voc_V']:8.3f} {r['Isc_A']*1e3:9.2f} "
                  f"{r['Pmp_W']*1e3:9.2f} {r['FF']:6.3f} {r['eta']*100:7.2f}")
        return

    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "I-V curves vs irradiance", "P-V curves vs irradiance",
        "Efficiency vs irradiance (low-light strength)", "Thermal warm-up (1 sun)"))

    for G in irradiances:
        r = m.iv_curve(G, 298.15)
        fig.add_trace(go.Scatter(x=r["V"], y=r["I"] * 1e3, name=f"{G:.0f} W/m2"), row=1, col=1)
        fig.add_trace(go.Scatter(x=r["V"], y=r["P"] * 1e3, name=f"{G:.0f} W/m2",
                                 showlegend=False), row=1, col=2)

    Gs = np.linspace(50, 1200, 30)
    etas = [m.iv_curve(G, 298.15)["eta"] * 100 for G in Gs]
    fig.add_trace(go.Scatter(x=Gs, y=etas, name="eta(G)", showlegend=False), row=2, col=1)

    sim = m.simulate(1000.0, T0=298.15, T_amb=298.15, dt=5.0, duration_s=1200.0)
    fig.add_trace(go.Scatter(x=sim["t"], y=sim["temperature"], name="T(t)",
                             showlegend=False), row=2, col=2)

    fig.update_xaxes(title_text="V [V]", row=1, col=1)
    fig.update_yaxes(title_text="I [mA]", row=1, col=1)
    fig.update_xaxes(title_text="V [V]", row=1, col=2)
    fig.update_yaxes(title_text="P [mW]", row=1, col=2)
    fig.update_xaxes(title_text="G [W/m2]", row=2, col=1)
    fig.update_yaxes(title_text="eta [%]", row=2, col=1)
    fig.update_xaxes(title_text="t [s]", row=2, col=2)
    fig.update_yaxes(title_text="T [K]", row=2, col=2)
    fig.update_layout(title="EC051 DSSC F2a -- Physics-Lumped Single-Diode + Thermal ODE",
                      height=800)

    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print(f"Report written to {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
