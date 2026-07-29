"""
EC129 -- Run-of-River Hydropower -- F2a Physics-Lumped Headpond Transient
Simulation scenarios + optional Plotly HTML report.

Run: python3 scripts/simulate.py
Plotly import is wrapped so its absence does not crash the run.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def diurnal_inflow(t):
    """Mild diurnal river inflow variation around 45 m3/s [s -> m3/s]."""
    return 45.0 + 10.0 * np.sin(2.0 * np.pi * t / 86400.0)


def peaking_demand(t):
    """Operator runs harder during the evening peak (16-22 h)."""
    hour = (t / 3600.0) % 24.0
    return 55.0 if 16.0 <= hour <= 22.0 else 35.0


def main():
    cm = ComponentModel()
    print("Model:", cm.get_info()["fidelity"])

    # 48-hour run-of-river operation with diurnal inflow and peaking demand.
    r = cm.predict({
        "Q_inflow_m3s": diurnal_inflow,
        "Q_demand_m3s": peaking_demand,
        "z0_m": 8.0,
        "dt": 300.0,
        "duration_s": 2 * 86400.0,
    })

    hrs = r["t"] / 3600.0
    print(f"  level range: {r['z'].min():.2f} - {r['z'].max():.2f} m")
    print(f"  net head range: {r['H_net'].min():.2f} - {r['H_net'].max():.2f} m")
    print(f"  power range: {r['power_kw'].min():.0f} - {r['power_kw'].max():.0f} kW")
    energy_MWh = np.trapz(r["power_kw"], r["t"]) / 3600.0 / 1000.0
    print(f"  energy over 48 h: {energy_MWh:.1f} MWh")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:  # plotly absent -> skip report gracefully
        print(f"  [plotly unavailable: {e}] skipping HTML report")
        return

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=("Forebay level & net head", "Flows", "Electrical power"),
    )
    fig.add_trace(go.Scatter(x=hrs, y=r["z"], name="z forebay [m]"), row=1, col=1)
    fig.add_trace(go.Scatter(x=hrs, y=r["H_net"], name="H_net [m]"), row=1, col=1)
    fig.add_trace(go.Scatter(x=hrs, y=r["Q_inflow"], name="Q_inflow [m3/s]"), row=2, col=1)
    fig.add_trace(go.Scatter(x=hrs, y=r["Q_turbine"], name="Q_turbine [m3/s]"), row=2, col=1)
    fig.add_trace(go.Scatter(x=hrs, y=r["Q_spill"], name="Q_spill [m3/s]"), row=2, col=1)
    fig.add_trace(go.Scatter(x=hrs, y=r["power_kw"], name="P [kW]"), row=3, col=1)
    fig.update_xaxes(title_text="time [h]", row=3, col=1)
    fig.update_layout(title="EC129 Run-of-River Hydropower -- F2a Headpond Transient",
                      height=850)

    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print(f"  wrote {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
