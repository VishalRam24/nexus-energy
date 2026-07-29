"""
EC100 -- Brayton Cycle Gas Turbine -- F2a Physics-Lumped
Optional Plotly report: efficiency & net work vs pressure ratio, and a spool
load-rejection transient. Plotly import is wrapped so its absence never crashes.
Run:  python3 scripts/simulate.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run():
    cm = ComponentModel()
    m = cm._model

    PRs = np.linspace(2, 40, 80)
    eta = np.array([m.cycle(PR=pr)["eta_thermal"] for pr in PRs])
    wnet = np.array([m.cycle(PR=pr)["w_net_J_kg"] / 1e3 for pr in PRs])  # kJ/kg
    carnot = np.array([m.cycle(PR=pr)["eta_carnot"] for pr in PRs])
    PR_opt = m.optimal_pressure_ratio()

    # Load-rejection: drop electrical load to 50% at t=5 s
    base = m.cycle()
    def load(t):
        return base["W_net_W"] * (1.0 if t < 5.0 else 0.5)
    tr = m.simulate_spool(load, t_end=20.0)

    print(f"PR_opt (analytic) = {PR_opt:.1f}")
    print(f"eta @ PR=18: {m.cycle(PR=18.0)['eta_thermal']:.4f}  Carnot: {m.cycle(PR=18.0)['eta_carnot']:.4f}")
    print(f"load rejection: rpm {tr['rpm'][0]:.0f} -> {tr['rpm'][-1]:.0f}")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        print(f"[simulate] Plotly unavailable ({e}); numeric summary only.")
        return

    fig = make_subplots(rows=1, cols=3, subplot_titles=(
        "Thermal eff. vs PR", "Net specific work vs PR", "Spool load-rejection"))
    fig.add_trace(go.Scatter(x=PRs, y=eta, name="eta_thermal"), 1, 1)
    fig.add_trace(go.Scatter(x=PRs, y=carnot, name="eta_Carnot", line=dict(dash="dash")), 1, 1)
    fig.add_trace(go.Scatter(x=PRs, y=wnet, name="w_net [kJ/kg]"), 1, 2)
    fig.add_vline(x=PR_opt, line_dash="dot", row=1, col=2)
    fig.add_trace(go.Scatter(x=tr["t"], y=tr["rpm"], name="shaft rpm"), 1, 3)
    fig.update_layout(title="EC100 Brayton F2a -- Air-Standard Cycle + Spool Dynamics")

    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print(f"[simulate] wrote {out}")


if __name__ == "__main__":
    run()
