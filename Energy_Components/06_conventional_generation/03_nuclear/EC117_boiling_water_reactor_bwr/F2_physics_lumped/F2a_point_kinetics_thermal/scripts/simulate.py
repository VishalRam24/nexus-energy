"""
EC117 -- BWR F2a -- simulation scenarios + optional Plotly HTML report.

Runs a +0.3$ reactivity-insertion transient and writes simulation_report.html
showing power, fuel/coolant temperature, void fraction and reactivity. Plotly is
optional -- if unavailable the script still prints a text summary.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run():
    cm = ComponentModel()
    scenarios = {
        "+0.3$ insertion": cm.predict({"reactivity_dollars": 0.3, "duration_s": 80.0, "dt": 0.02}),
        "-1.0$ insertion": cm.predict({"reactivity_dollars": -1.0, "duration_s": 80.0, "dt": 0.02}),
        "steady (0$)": cm.predict({"reactivity_dollars": 0.0, "duration_s": 80.0, "dt": 0.02}),
    }

    for name, r in scenarios.items():
        print(f"{name:18s}: P0={r['power_mw'][0]:7.1f} MW  "
              f"peak={r['power_mw'].max():7.1f}  final={r['power_mw'][-1]:7.1f}  "
              f"Tf_final={r['T_fuel'][-1]:6.1f} K  void={r['void_fraction'][-1]:.3f}")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception:
        print("[plotly not available -- skipping HTML report]")
        return

    r = scenarios["+0.3$ insertion"]
    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "Thermal power [MW]", "Fuel & coolant temperature [K]",
        "Core-average void fraction [-]", "Reactivity [$]"))
    fig.add_trace(go.Scatter(x=r["t"], y=r["power_mw"], name="P_th"), row=1, col=1)
    fig.add_trace(go.Scatter(x=r["t"], y=r["T_fuel"], name="T_fuel"), row=1, col=2)
    fig.add_trace(go.Scatter(x=r["t"], y=r["T_coolant"], name="T_coolant"), row=1, col=2)
    fig.add_trace(go.Scatter(x=r["t"], y=r["void_fraction"], name="void"), row=2, col=1)
    fig.add_trace(go.Scatter(x=r["t"], y=r["reactivity_dollars"], name="rho [$]"), row=2, col=2)
    fig.update_layout(title="EC117 BWR F2a -- +0.3$ reactivity insertion (point kinetics + TH feedback)",
                      height=720, showlegend=True)
    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print(f"[report written -> {os.path.abspath(out)}]")


if __name__ == "__main__":
    run()
