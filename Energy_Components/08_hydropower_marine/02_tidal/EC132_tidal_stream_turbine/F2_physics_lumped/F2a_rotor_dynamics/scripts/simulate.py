"""
EC132 -- Tidal Stream Turbine -- F2a Physics-Lumped Rotor Dynamics
Optional Plotly report. Plotly import is guarded so absence does not crash.
Run: python3 scripts/simulate.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def main():
    cm = ComponentModel()
    m = cm._model

    # Scenario 1: full M2 semidiurnal tidal cycle
    sim = cm.predict({"v_mean": 2.0, "v_amp": 1.0,
                      "tidal_period_s": 44712.0, "duration_s": 44712.0, "dt": 60.0})

    # Scenario 2: static Cp-lambda curve
    lam = np.linspace(0.0, m.lambda_zero, 300)
    cp = m.cp(lam)

    print("=== EC132 F2a Rotor-Dynamics Simulation ===")
    print(f"Mean P_elec    : {sim['power_electrical_kw'].mean():.1f} kW")
    print(f"Peak P_elec    : {sim['power_electrical_kw'].max():.1f} kW")
    print(f"Capacity factor: {sim['capacity_factor']:.3f}")
    print(f"Energy / tide  : {sim['energy_electrical_wh']/1000:.0f} kWh")
    print(f"Peak Cp        : {cp.max():.3f} (Betz {16/27:.3f})")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as exc:  # plotly absent -> skip plotting silently
        print(f"[plotly unavailable: {exc}] -- numeric summary only.")
        return

    t_h = sim["t"] / 3600.0
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Tidal current speed", "Rotor speed (rpm)",
                        "Electrical power", "Cp-lambda curve"),
    )
    fig.add_trace(go.Scatter(x=t_h, y=sim["v"], name="v [m/s]"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_h, y=sim["rpm"], name="rpm"), row=1, col=2)
    fig.add_trace(go.Scatter(x=t_h, y=sim["power_electrical_kw"], name="P [kW]"),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=lam, y=cp, name="Cp(lambda)"), row=2, col=2)
    fig.add_hline(y=16 / 27, line_dash="dash", row=2, col=2)
    fig.update_layout(title="EC132 Tidal Stream Turbine -- F2a Rotor Dynamics",
                      height=720)

    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print(f"Report written: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
