"""
EC177 -- BLDC F2a -- simulation report.
Generates startup transient + speed-torque curve plots (Plotly optional).
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run():
    cm = ComponentModel()
    m = cm._model

    # Startup transient
    tr = cm.predict({"T_load_Nm": 0.5, "duration_s": 0.4, "dt": 1e-4})

    # Speed-torque curve (simulated steady states) vs analytic line
    T_grid = np.linspace(0.0, 0.9 * m.stall_torque(), 12)
    w_sim = []
    for T in T_grid:
        r = m.simulate(T_load=float(T), dt=2e-4, duration_s=1.5)
        w_sim.append(r["omega_final"])
    w_sim = np.array(w_sim)
    w_analytic = m.speed_at_torque(T_grid)

    print(f"No-load speed : {m.no_load_speed()*60/(2*np.pi):.0f} rpm")
    print(f"Stall torque  : {m.stall_torque():.2f} Nm")
    print(f"Startup final : {tr['speed_rpm'][-1]:.0f} rpm, eff={tr['efficiency']:.3f}")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=2, cols=2,
                            subplot_titles=("Speed startup", "Current startup",
                                            "Torque startup", "Speed-torque curve"))
        fig.add_trace(go.Scatter(x=tr["t"], y=tr["speed_rpm"], name="rpm"), 1, 1)
        fig.add_trace(go.Scatter(x=tr["t"], y=tr["current"], name="i [A]"), 1, 2)
        fig.add_trace(go.Scatter(x=tr["t"], y=tr["torque_e"], name="T_e [Nm]"), 2, 1)
        fig.add_trace(go.Scatter(x=T_grid, y=w_sim*60/(2*np.pi),
                                 mode="markers", name="sim"), 2, 2)
        fig.add_trace(go.Scatter(x=T_grid, y=w_analytic*60/(2*np.pi),
                                 mode="lines", name="analytic"), 2, 2)
        out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
        fig.write_html(out)
        print(f"Report written: {out}")
    except Exception as e:
        print(f"[plotly unavailable, skipping HTML] {e}")


if __name__ == "__main__":
    run()
