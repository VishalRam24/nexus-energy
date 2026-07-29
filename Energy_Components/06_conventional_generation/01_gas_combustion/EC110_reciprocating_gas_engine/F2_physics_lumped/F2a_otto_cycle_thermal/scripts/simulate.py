"""
EC110 -- Reciprocating Gas Engine -- F2a Otto/Miller Cycle + Thermal ODE
Optional Plotly simulation report. Plotly import is wrapped so its absence
does not crash; run via `python3 scripts/simulate.py`.
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

    # 1. Efficiency vs part-load
    plrs = np.linspace(0.4, 1.0, 25)
    eta_b = np.array([m.operating_point(p)["eta_brake"] for p in plrs])
    eta_i = np.array([m.operating_point(p)["eta_indicated"] for p in plrs])
    P_brake = np.array([m.operating_point(p)["P_brake_w"] / 1e3 for p in plrs])

    # 2. Cold-start thermal transient at rated load
    r = cm.predict({"part_load_ratio": 1.0, "T0_K": 298.15, "dt": 5.0, "duration_s": 2400.0})

    # 3. Load-step schedule
    def sched(t):
        return 0.5 if t < 1000 else 1.0
    rs = cm.predict({"part_load_ratio": sched, "T0_K": 330.0, "dt": 5.0, "duration_s": 2400.0})

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:  # noqa: BLE001
        print(f"Plotly unavailable ({e}); skipping HTML report.")
        op = m.operating_point(1.0)
        print(f"Rated eta_brake={op['eta_brake']*100:.1f}%  T_final={r['temperature'][-1]:.1f} K")
        return

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Brake vs Indicated Efficiency vs Part Load",
            "Brake Power vs Part Load",
            "Cold-Start Engine-Block Thermal Transient",
            "Block Temperature under 0.5->1.0 Load Step",
        ),
    )
    fig.add_trace(go.Scatter(x=plrs, y=eta_i * 100, name="eta_indicated"), 1, 1)
    fig.add_trace(go.Scatter(x=plrs, y=eta_b * 100, name="eta_brake"), 1, 1)
    fig.add_trace(go.Scatter(x=plrs, y=P_brake, name="P_brake"), 1, 2)
    fig.add_trace(go.Scatter(x=r["t"] / 60.0, y=r["temperature"], name="T_block"), 2, 1)
    fig.add_trace(go.Scatter(x=rs["t"] / 60.0, y=rs["temperature"], name="T_block step"), 2, 2)
    fig.update_xaxes(title_text="Part-load ratio [-]", row=1, col=1)
    fig.update_xaxes(title_text="Part-load ratio [-]", row=1, col=2)
    fig.update_xaxes(title_text="Time [min]", row=2, col=1)
    fig.update_xaxes(title_text="Time [min]", row=2, col=2)
    fig.update_yaxes(title_text="Efficiency [%]", row=1, col=1)
    fig.update_yaxes(title_text="Brake power [kW]", row=1, col=2)
    fig.update_yaxes(title_text="T_block [K]", row=2, col=1)
    fig.update_yaxes(title_text="T_block [K]", row=2, col=2)
    fig.update_layout(title="EC110 Reciprocating Gas Engine — F2a Otto/Miller + Thermal ODE",
                      height=800)
    fig.write_html(_OUT)
    print(f"Report written to {_OUT}")


if __name__ == "__main__":
    run()
