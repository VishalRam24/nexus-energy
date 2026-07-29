"""
EC003 -- Alkaline Fuel Cell (AFC) -- F2a Electrochemical
Optional Plotly report. Plotly import is guarded so absence does not crash.

Run:  python3 scripts/simulate.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run_scenarios():
    cm = ComponentModel()
    m = cm._model

    # 1) Polarization curve (steady, at T_ref)
    j_grid = np.linspace(0.005, 0.98, 80)
    T = m.T_ref
    V = np.array([m.cell_voltage(j, T, 1.0, 0.21) for j in j_grid])
    P = j_grid * V

    # 2) Cold-start thermal transient under load
    tr = m.simulate(0.5, 313.15, 1.0, 0.21, 1.0, 600.0)

    # 3) KOH conductivity vs concentration sweep
    c_grid = np.linspace(0.5, 12.0, 60)
    kappa = np.array([m.koh_conductivity(T, c) for c in c_grid])

    return j_grid, V, P, tr, c_grid, kappa


def build_report(path="simulation_report.html"):
    j_grid, V, P, tr, c_grid, kappa = run_scenarios()
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:  # plotly absent -> text summary only
        print(f"[simulate] Plotly unavailable ({e}); printing text summary.")
        print(f"  Polarization: V@0.4A/cm2 = "
              f"{np.interp(0.4, j_grid, V):.4f} V, "
              f"peak power density = {P.max():.4f} W/cm2 at "
              f"j={j_grid[np.argmax(P)]:.3f} A/cm2")
        print(f"  Cold-start: T 313.15 -> {tr['temperature'][-1]:.2f} K over 600 s")
        print(f"  KOH peak conductivity = {kappa.max():.4f} S/cm at "
              f"c={c_grid[np.argmax(kappa)]:.2f} mol/L")
        return None

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Polarization curve (V-j)",
            "Power density (W/cm2)",
            "Cold-start thermal transient",
            "KOH conductivity vs concentration (Gilliam 2007)",
        ),
    )
    fig.add_trace(go.Scatter(x=j_grid, y=V, name="V_cell"), row=1, col=1)
    fig.add_trace(go.Scatter(x=j_grid, y=P, name="P_density"), row=1, col=2)
    fig.add_trace(go.Scatter(x=tr["t"], y=tr["temperature"], name="T_stack"),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=c_grid, y=kappa, name="kappa_KOH"), row=2, col=2)
    fig.update_layout(title_text="EC003 AFC F2a — Electrochemical + Thermal",
                      showlegend=False, height=720)
    out = os.path.join(os.path.dirname(__file__), "..", path)
    fig.write_html(out)
    print(f"[simulate] Report written to {out}")
    return out


if __name__ == "__main__":
    build_report()
