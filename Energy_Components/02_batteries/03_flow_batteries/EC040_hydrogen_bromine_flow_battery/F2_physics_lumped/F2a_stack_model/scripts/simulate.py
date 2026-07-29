"""
EC040 -- HBrFB F2a -- Plotly simulation report (optional).
Generates simulation_report.html for visual verification. Plotly import is
wrapped so its absence does not crash.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run_scenarios():
    cm = ComponentModel()
    m = cm._model

    # 1. Discharge profile (SOC 0.9 -> down) at 80 A
    r_dis = m.simulate(80.0, soc0=0.9, T0=298.15, dt=60.0, duration_s=3000.0)

    # 2. Polarization curve at SOC=0.5, T=25 C
    I = np.linspace(-250, 250, 101)
    Vdis = np.array([m.stack_voltage(i, 0.5, 298.15) for i in I])

    # 3. Round-trip efficiency vs current
    Iabs = np.linspace(10, 250, 25)
    rte = np.array([m.round_trip_efficiency(i, 0.5, 298.15) for i in Iabs])

    return r_dis, (I, Vdis), (Iabs, rte)


def build_report(path=None):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        print(f"[simulate] Plotly unavailable ({e}); skipping HTML report.")
        return None

    r_dis, pol, eff = run_scenarios()
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Discharge @80 A: stack voltage & SOC",
            "Discharge: temperature rise",
            "Stack polarization curve (SOC=0.5, 25 C)",
            "Round-trip efficiency vs current",
        ),
        specs=[[{"secondary_y": True}, {}], [{}, {}]],
    )
    fig.add_trace(go.Scatter(x=r_dis["t"], y=r_dis["stack_voltage"], name="V_stack"),
                  row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=r_dis["t"], y=r_dis["soc"], name="SOC"),
                  row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=r_dis["t"], y=r_dis["temperature"], name="T [K]"),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=pol[0], y=pol[1], name="V_stack(I)"), row=2, col=1)
    fig.add_trace(go.Scatter(x=eff[0], y=eff[1] * 100, name="RTE [%]"), row=2, col=2)
    fig.update_layout(title="EC040 HBrFB F2a -- Physics-Lumped Stack Report", height=800)
    fig.write_html(path)
    print(f"[simulate] Report written to {path}")
    return path


if __name__ == "__main__":
    build_report()
