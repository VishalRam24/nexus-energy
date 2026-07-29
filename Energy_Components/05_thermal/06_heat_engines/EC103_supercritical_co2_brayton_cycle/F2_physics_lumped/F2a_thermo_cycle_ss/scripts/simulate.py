"""
EC103 -- sCO2 Brayton F2a -- optional Plotly simulation report.
Plotly import is wrapped so its absence never crashes the module.
Run: python3 scripts/simulate.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run():
    cm = ComponentModel()

    # 1) efficiency vs turbine inlet temperature
    T4 = np.linspace(773.15, 1073.15, 25)
    eta = np.array([cm._model.cycle(T_turb_in=t)["eta_thermal"] for t in T4])
    carnot = np.array([cm._model.cycle(T_turb_in=t)["eta_carnot"] for t in T4])

    # 2) compressibility / density vs temperature (near-critical advantage)
    Tline = np.linspace(305.15, 700.0, 40)
    Z = np.array([cm._model.compressibility(t, 7.7e6) for t in Tline])
    rho = np.array([cm._model.density(t, 7.7e6) for t in Tline])

    # 3) transient warm-up
    sim = cm._model.simulate(T_metal0=400.0, dt=5.0, duration_s=2000.0)

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:  # pragma: no cover
        print(f"[simulate] plotly unavailable ({e}); computed arrays only.")
        print(f"  eta range: {eta.min():.3f}-{eta.max():.3f}")
        print(f"  Z range:   {Z.min():.3f}-{Z.max():.3f}")
        print(f"  T_final:   {sim['T_turbine_inlet'][-1]:.1f} K")
        return

    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "Thermal efficiency vs turbine inlet T",
        "Compressibility Z @ 7.7 MPa (near-critical liquid-like)",
        "Density @ 7.7 MPa",
        "Transient hot-section warm-up (lumped ODE)"))
    fig.add_trace(go.Scatter(x=T4 - 273.15, y=eta, name="eta_thermal"), 1, 1)
    fig.add_trace(go.Scatter(x=T4 - 273.15, y=carnot, name="Carnot"), 1, 1)
    fig.add_trace(go.Scatter(x=Tline - 273.15, y=Z, name="Z"), 1, 2)
    fig.add_trace(go.Scatter(x=Tline - 273.15, y=rho, name="rho"), 2, 1)
    fig.add_trace(go.Scatter(x=sim["t"], y=sim["T_turbine_inlet"],
                             name="T_turbine_inlet"), 2, 2)
    fig.update_layout(title="EC103 sCO2 Brayton F2a — Physics-Lumped Report",
                      height=800, showlegend=True)
    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print(f"[simulate] wrote {os.path.abspath(out)}")


if __name__ == "__main__":
    run()
