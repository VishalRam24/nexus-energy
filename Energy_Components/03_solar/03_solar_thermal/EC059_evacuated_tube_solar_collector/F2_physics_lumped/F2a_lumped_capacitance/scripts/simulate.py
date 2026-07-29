"""
EC059 — Evacuated Tube Solar Collector — F2a Lumped-Capacitance
Optional Plotly HTML report. Plotly import is guarded so absence never crashes.
Run: python3 scripts/simulate.py
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


def diurnal_run():
    cm = ComponentModel()
    m = cm._model

    def G_of_t(t):
        # daylight half-sine peaking at solar noon (t in seconds, 0 = midnight)
        frac = t / 86400.0
        x = (frac - 0.25) / 0.5  # sunrise 6h, sunset 18h
        return 1000.0 * max(0.0, np.sin(np.pi * x)) if 0.0 <= x <= 1.0 else 0.0

    def Ta_of_t(t):
        return 12.0 + 8.0 * np.sin(2 * np.pi * (t / 86400.0 - 0.35))

    return m.simulate(G_of_t, Ta_of_t, T_inlet_c=40.0, dt=300.0, duration_s=86400.0)


def efficiency_curve():
    cm = ComponentModel()
    m = cm._model
    G, Ta = 800.0, 15.0
    xs, etas = [], []
    for Tin in np.linspace(15.0, 160.0, 25):
        eta = m.efficiency_steady(G, Tin, Ta)
        xs.append((Tin - Ta) / G)  # approximate reduced temperature at inlet
        etas.append(eta)
    return np.array(xs), np.array(etas), m.optical_eff


def build_report(out_path):
    r = diurnal_run()
    xs, etas, eta0 = efficiency_curve()

    if not HAVE_PLOTLY:
        print("plotly not installed — skipping HTML report (model still runs).")
        print(f"Diurnal peak useful heat: {np.max(r['useful_heat_w']):.0f} W")
        print(f"Efficiency intercept ~ {etas[0]:.3f} (eta_0 = {eta0})")
        return

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Diurnal temperatures", "Heat flows",
                        "Efficiency curve (EN 12975)", "Effective U_L"),
    )
    th = r["t"] / 3600.0
    fig.add_trace(go.Scatter(x=th, y=r["T_absorber_c"], name="T_absorber"), 1, 1)
    fig.add_trace(go.Scatter(x=th, y=r["T_outlet_c"], name="T_outlet"), 1, 1)
    fig.add_trace(go.Scatter(x=th, y=r["T_ambient_c"], name="T_ambient"), 1, 1)
    fig.add_trace(go.Scatter(x=th, y=r["q_absorbed_w"], name="Q_absorbed"), 1, 2)
    fig.add_trace(go.Scatter(x=th, y=r["q_loss_w"], name="Q_loss"), 1, 2)
    fig.add_trace(go.Scatter(x=th, y=r["useful_heat_w"], name="Q_useful"), 1, 2)
    fig.add_trace(go.Scatter(x=xs, y=etas, name="eta(x)", mode="lines+markers"), 2, 1)
    fig.add_trace(go.Scatter(x=r["reduced_temp"], y=r["U_L_w_m2k"],
                             name="U_L", mode="markers"), 2, 2)
    fig.update_layout(title="EC059 Evacuated Tube Collector — F2a Lumped-Capacitance",
                      height=800, showlegend=True)
    fig.write_html(out_path)
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    build_report(os.path.abspath(out))
