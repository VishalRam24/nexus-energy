"""
EC133 -- Tidal Lagoon -- F2a Physics-Lumped Water-Level ODE
Optional Plotly report. Plotly import is guarded so its absence never crashes.
Run: python3 scripts/simulate.py  (writes simulation_report.html if plotly present)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def main():
    cm = ComponentModel()
    r = cm.predict({"n_cycles": 3})

    print("EC133 Tidal Lagoon F2a -- simulation summary")
    print(f"  energy/cycle : {r['energy_per_cycle_MWh']:.1f} MWh")
    print(f"  avg power    : {r['avg_power_MW']:.1f} MW")
    print(f"  cap. factor  : {r['capacity_factor']:.3f}")
    print(f"  peak |head|  : {abs(r['head']).max():.2f} m")

    best_h, best_E, grid, energies = cm._model.optimal_hold_head(n_cycles=2)
    print(f"  optimal hold : {best_h:.2f} m -> {best_E:.1f} MWh/cycle")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        print(f"  [plotly unavailable: {e}] -- skipping HTML report")
        return

    t_h = r["t"] / 3600.0
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=("Water levels (sea vs lagoon)",
                        "Head & turbine flow",
                        "Generated power"),
    )
    fig.add_trace(go.Scatter(x=t_h, y=r["z_sea"], name="sea", line=dict(color="navy")), 1, 1)
    fig.add_trace(go.Scatter(x=t_h, y=r["z_lagoon"], name="lagoon", line=dict(color="teal")), 1, 1)
    fig.add_trace(go.Scatter(x=t_h, y=r["head"], name="head H", line=dict(color="firebrick")), 2, 1)
    fig.add_trace(go.Scatter(x=t_h, y=r["flow"], name="flow Q", line=dict(color="green"), yaxis="y4"), 2, 1)
    fig.add_trace(go.Scatter(x=t_h, y=r["power_MW"], name="power", line=dict(color="darkorange")), 3, 1)
    fig.update_xaxes(title_text="time [h]", row=3, col=1)
    fig.update_yaxes(title_text="level [m]", row=1, col=1)
    fig.update_yaxes(title_text="head [m] / flow", row=2, col=1)
    fig.update_yaxes(title_text="P [MW]", row=3, col=1)
    fig.update_layout(title="EC133 Tidal Lagoon F2a -- Two-Way Generation Dynamics", height=800)

    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print(f"  report -> {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
