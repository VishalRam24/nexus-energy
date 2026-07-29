"""
EC127 -- Gravity Energy Storage -- F2a Hoist Dynamics
Optional Plotly report. Plotly import is guarded so absence does not crash.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def build_report(out_html=None):
    cm = ComponentModel()
    m = cm._model

    charge = cm.predict({"mode": "charge", "v_target": 3.0, "dt": 2.0})
    discharge = cm.predict({"mode": "discharge", "v_target": 3.0, "dt": 2.0})
    eta_rt = cm.round_trip_efficiency(v_target=3.0, dt=2.0)

    # RTE vs commanded speed sweep
    speeds = np.linspace(0.5, m.v_max, 8)
    rte = [cm.round_trip_efficiency(v_target=v, dt=4.0) for v in speeds]

    print(f"Capacity: {m.energy_capacity_kwh():.0f} kWh")
    print(f"Charge E_in : {abs(charge['E_elec_kwh']):.1f} kWh")
    print(f"Discharge E_out: {abs(discharge['E_elec_kwh']):.1f} kWh")
    print(f"Round-trip efficiency: {eta_rt*100:.1f} %")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        print(f"[simulate] Plotly unavailable ({e}); skipping HTML report.")
        return None

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Height vs time (charge)", "Velocity vs time (charge)",
                        "Electrical power vs time", "RTE vs line speed"),
    )
    fig.add_trace(go.Scatter(x=charge["t"], y=charge["height"], name="height (charge)"), 1, 1)
    fig.add_trace(go.Scatter(x=charge["t"], y=charge["v"], name="v (charge)"), 1, 2)
    fig.add_trace(go.Scatter(x=charge["t"], y=charge["P_elec_kw"], name="P_elec charge"), 2, 1)
    fig.add_trace(go.Scatter(x=discharge["t"], y=discharge["P_elec_kw"], name="P_elec discharge"), 2, 1)
    fig.add_trace(go.Scatter(x=speeds, y=[100*e for e in rte], name="RTE %"), 2, 2)
    fig.update_layout(title=f"EC127 Gravity Storage F2a -- RTE={eta_rt*100:.1f}%", height=800)

    if out_html is None:
        out_html = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out_html)
    print(f"[simulate] Wrote {out_html}")
    return out_html


if __name__ == "__main__":
    build_report()
