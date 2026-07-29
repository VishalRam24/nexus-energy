"""
EC050 -- OPV F2a -- Simulation scenarios + optional Plotly HTML report.
Plotly import is wrapped so its absence does not crash the run.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel

_OUT = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")


def run():
    cm = ComponentModel()
    m = cm._model

    # Scenario 1: I-V / P-V curves at several irradiances (25C cell)
    curves = {}
    for G in [100.0, 400.0, 700.0, 1000.0]:
        curves[G] = m.iv_curve(G, 25.0 + 273.15, n_points=200)

    # Scenario 2: efficiency & FF vs irradiance (low-light story)
    G_sweep = np.linspace(20.0, 1200.0, 60)
    eta = np.array([m.mpp(G, 25.0 + 273.15)["eta"] for G in G_sweep])
    ff = np.array([m.mpp(G, 25.0 + 273.15)["FF"] for G in G_sweep])

    # Scenario 3: thermal-ODE transient with a step in irradiance
    def G_step(t):
        return 300.0 if t < 300.0 else 1000.0
    ts = m.simulate(G_step, 25.0, T0_C=25.0, dt=5.0, duration_s=900.0)

    print("OPV F2a simulation scenarios computed.")
    print(f"  STC eta = {m.mpp(1000.0, 298.15)['eta']*100:.2f} %, "
          f"FF = {m.mpp(1000.0, 298.15)['FF']:.3f}")
    print(f"  Final cell temp after step = {ts['T_cell_C'][-1]:.1f} C")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        print(f"[plotly unavailable: {e}] skipping HTML report.")
        return

    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "I-V curves", "P-V curves",
        "Efficiency & FF vs irradiance", "Thermal-ODE transient (G step)"))

    for G, c in curves.items():
        fig.add_trace(go.Scatter(x=c["V"], y=c["I"], name=f"{int(G)} W/m2"), row=1, col=1)
        fig.add_trace(go.Scatter(x=c["V"], y=c["P"], name=f"{int(G)} W/m2",
                                 showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=G_sweep, y=eta * 100, name="eta [%]"), row=2, col=1)
    fig.add_trace(go.Scatter(x=G_sweep, y=ff, name="FF", yaxis="y4"), row=2, col=1)
    fig.add_trace(go.Scatter(x=ts["t"], y=ts["T_cell_C"], name="T_cell [C]"), row=2, col=2)
    fig.add_trace(go.Scatter(x=ts["t"], y=ts["power"], name="P [W]"), row=2, col=2)

    fig.update_layout(title="EC050 OPV -- F2a Physics-Lumped Single-Diode + Thermal ODE",
                      height=800, width=1100)
    fig.write_html(_OUT)
    print(f"Report written to {_OUT}")


if __name__ == "__main__":
    run()
