"""
EC026 -- Li-Air F2a (Pore Passivation) -- simulation scenarios + optional Plotly report.
Plotly import is guarded so its absence never crashes the run.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run_scenarios():
    cm = ComponentModel()
    # Scenario A: deep discharge of a fresh cell at 1 A -> pore saturation cutoff
    dis = cm.predict({"current_A": 1.0, "soc_0": 1.0, "theta_0": 0.0,
                      "dt": 30.0, "duration_s": 3600.0})
    # Scenario B: recharge a passivated cell at 0.5 A -> high OER plateau
    chg = cm.predict({"current_A": -0.5, "soc_0": 0.2, "theta_0": 0.7,
                      "dt": 30.0, "duration_s": 2400.0})
    return dis, chg


def build_report(path=None):
    dis, chg = run_scenarios()
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:  # plotly not installed -- skip plotting, no crash
        print(f"[simulate] Plotly unavailable ({e}); scenarios computed, report skipped.")
        return None

    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "Discharge V & E_eq vs time", "Pore fill theta vs time",
        "Charge V vs E_eq (OER plateau)", "Cell temperature"))
    fig.add_trace(go.Scatter(x=dis["t"], y=dis["voltage"], name="V_dis"), 1, 1)
    fig.add_trace(go.Scatter(x=dis["t"], y=dis["equilibrium_voltage"], name="E_eq"), 1, 1)
    fig.add_trace(go.Scatter(x=dis["t"], y=dis["theta"], name="theta"), 1, 2)
    fig.add_trace(go.Scatter(x=chg["t"], y=chg["voltage"], name="V_chg"), 2, 1)
    fig.add_trace(go.Scatter(x=chg["t"], y=chg["equilibrium_voltage"], name="E_eq_chg"), 2, 1)
    fig.add_trace(go.Scatter(x=dis["t"], y=dis["temperature"], name="T_dis"), 2, 2)
    fig.update_layout(title="EC026 Li-Air F2a -- Pore Passivation + Thermal ODE", height=750)

    if path is None:
        path = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(path)
    print(f"[simulate] Report written to {path}")
    return path


if __name__ == "__main__":
    build_report()
