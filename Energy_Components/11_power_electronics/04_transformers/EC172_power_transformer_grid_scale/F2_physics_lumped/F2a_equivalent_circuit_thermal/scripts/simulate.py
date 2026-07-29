"""
EC172 -- Power Transformer (Grid-Scale) -- F2a
Optional Plotly simulation report. Plotly import is guarded so absence
does not crash. Produces simulation_report.html in the model folder.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel

_OUT = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")


def main():
    cm = ComponentModel()
    m = cm._model

    # 1) efficiency vs load at a few power factors
    plr = np.linspace(0.02, 1.3, 120)
    eta_unity = [m.efficiency(p, 1.0, 1.0, 75.0) for p in plr]
    eta_09 = [m.efficiency(p, 1.0, 0.9, 75.0) for p in plr]

    # 2) regulation vs load for lagging / unity / leading pf
    vr_lag = [m.voltage_regulation(p, 0.8) * 100 for p in plr]
    vr_unity = [m.voltage_regulation(p, 1.0) * 100 for p in plr]
    vr_lead = [m.voltage_regulation(p, 0.8, leading=True) * 100 for p in plr]

    # 3) daily loading transient -> hot-spot temperature
    def daily(t):  # t in minutes
        h = (t / 60.0) % 24.0
        return 0.6 + 0.5 * np.exp(-((h - 18.0) ** 2) / 8.0)  # evening peak
    r = m.simulate(daily, ambient_temperature=25.0, dt=5.0, duration=1440.0,
                   power_factor=0.9)

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:  # pragma: no cover
        print(f"[simulate] Plotly unavailable ({e}); skipping HTML report.")
        print(f"  Rated efficiency: {m.efficiency(1.0,1.0,1.0,75.0)*100:.3f}%")
        print(f"  Peak-eff load:    {m.max_efficiency_load():.3f} pu")
        print(f"  Daily peak hot-spot: {r['hotspot_temperature'].max():.1f} C")
        return

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Efficiency vs Load", "Voltage Regulation vs Load",
                        "Daily Hot-Spot Transient", "Loss Breakdown vs Load"),
    )
    fig.add_trace(go.Scatter(x=plr, y=eta_unity, name="eta @pf=1.0"), 1, 1)
    fig.add_trace(go.Scatter(x=plr, y=eta_09, name="eta @pf=0.9"), 1, 1)

    fig.add_trace(go.Scatter(x=plr, y=vr_lag, name="VR pf=0.8 lag"), 1, 2)
    fig.add_trace(go.Scatter(x=plr, y=vr_unity, name="VR pf=1.0"), 1, 2)
    fig.add_trace(go.Scatter(x=plr, y=vr_lead, name="VR pf=0.8 lead"), 1, 2)

    fig.add_trace(go.Scatter(x=r["t"] / 60.0, y=r["hotspot_temperature"],
                             name="hot-spot"), 2, 1)
    fig.add_trace(go.Scatter(x=r["t"] / 60.0, y=r["top_oil_temperature"],
                             name="top-oil"), 2, 1)

    pcore = [m.core_loss(1.0) / 1e3 for _ in plr]
    pcu = [m.copper_loss(p, 75.0) / 1e3 for p in plr]
    fig.add_trace(go.Scatter(x=plr, y=pcore, name="core loss [kW]"), 2, 2)
    fig.add_trace(go.Scatter(x=plr, y=pcu, name="copper loss [kW]"), 2, 2)

    fig.update_layout(title="EC172 Power Transformer F2a -- Equivalent Circuit + Thermal",
                      height=800)
    fig.write_html(_OUT)
    print(f"[simulate] wrote {_OUT}")


if __name__ == "__main__":
    main()
