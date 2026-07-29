"""
EC192 -- Gas Pressure Regulator -- F2a
Optional Plotly simulation report. Plotly import is wrapped so its absence
does not crash. Run: python3 scripts/simulate.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run_scenarios():
    cm = ComponentModel()
    m = cm._model

    # 1) Load-step transient
    def load(t):
        rho = m.rho_std
        if t < 20.0:
            return 8000.0 * rho / 3600.0
        elif t < 40.0:
            return 30000.0 * rho / 3600.0
        return 15000.0 * rho / 3600.0
    transient = m.simulate(50.0, load, 288.15, duration_s=60.0, dt=0.02)

    # 2) Droop / lockup curve
    loads = np.linspace(0.0, 40000.0, 12)
    droop = []
    for q in loads:
        s = m.steady_state(50.0, q * m.rho_std / 3600.0, 288.15, duration_s=120.0)
        droop.append(s["P_down_bar"])

    return transient, loads, np.array(droop), m.P_set / 1e5


def build_report(path="simulation_report.html"):
    transient, loads, droop, P_set = run_scenarios()
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        print(f"[simulate] Plotly unavailable ({e}); skipping HTML report.")
        print(f"  Transient final P_down = {transient['P_down_bar'][-1]:.3f} bar")
        print(f"  Droop band = {droop[0]-droop[-1]:.3f} bar")
        return None

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Downstream pressure (load-step)",
                        "Valve travel",
                        "JT downstream temperature",
                        "Droop / lockup curve"))

    t = transient["t"]
    fig.add_trace(go.Scatter(x=t, y=transient["P_down_bar"], name="P_down"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=transient["P_set_bar"], name="setpoint",
                             line=dict(dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=transient["valve_travel_frac"] * 100,
                             name="travel %"), row=1, col=2)
    fig.add_trace(go.Scatter(x=t, y=transient["T_downstream_K"], name="T_down"),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=loads, y=droop, name="droop", mode="lines+markers"),
                  row=2, col=2)
    fig.add_hline(y=P_set, line_dash="dash", row=2, col=2)

    fig.update_layout(title="EC192 Gas Pressure Regulator -- F2a Diaphragm Dynamics",
                      height=720, showlegend=True)
    out = os.path.join(os.path.dirname(__file__), "..", path)
    fig.write_html(out)
    print(f"[simulate] wrote {out}")
    return out


if __name__ == "__main__":
    build_report()
