"""
EC016 -- Hydrogen Compressor -- F2a Real-Gas Thermal
Optional Plotly simulation report. Plotly import is guarded so absence
does not break the build.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def build_report(out_html=None):
    cm = ComponentModel()

    # 1) transient warm-up
    r = cm.predict({"P_in_bar": 20.0, "P_out_bar": 900.0, "dt": 20.0, "duration_s": 3600.0})

    # 2) SEC / discharge T vs discharge pressure sweep
    P_outs = np.linspace(100.0, 1000.0, 25)
    sec = np.array([cm._model.sec_kwh_per_kg(20.0, P) for P in P_outs])
    Tdisc = np.array([cm._model.stage_profile(20.0, P)["T_discharge"][-1] for P in P_outs])

    # 3) Z vs pressure
    P_grid = np.linspace(20.0, 1000.0, 40)
    Z = np.array([cm._model.compressibility(298.15, P) for P in P_grid])

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:  # pragma: no cover
        print(f"[simulate] Plotly unavailable ({e}); skipping HTML report.")
        print(f"  T_metal {r['T_metal'][0]:.1f}->{r['T_metal'][-1]:.1f} K, "
              f"SEC={r['SEC_kWh_kg']:.3f} kWh/kg")
        return None

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Metal temperature transient", "Shaft power transient",
                        "SEC & final discharge T vs P_out", "H2 compressibility Z vs P"),
        specs=[[{}, {}], [{"secondary_y": True}, {}]],
    )
    fig.add_trace(go.Scatter(x=r["t"], y=r["T_metal"], name="T_metal [K]"), row=1, col=1)
    fig.add_trace(go.Scatter(x=r["t"], y=r["shaft_power_kW"], name="P_shaft [kW]"), row=1, col=2)
    fig.add_trace(go.Scatter(x=P_outs, y=sec, name="SEC [kWh/kg]"), row=2, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=P_outs, y=Tdisc, name="T_disc [K]"), row=2, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=P_grid, y=Z, name="Z [-]"), row=2, col=2)
    fig.update_layout(title="EC016 H2 Compressor — F2a Real-Gas Thermal", height=800)

    if out_html is None:
        out_html = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out_html)
    print(f"[simulate] Wrote {out_html}")
    return out_html


if __name__ == "__main__":
    build_report()
