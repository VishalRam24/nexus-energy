"""
EC141 -- Anaerobic Digester (Thermophilic) -- F2a
Optional Plotly simulation report (gracefully degrades if plotly absent).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run_report(out_html=None):
    cm = ComponentModel()
    r = cm.predict({"S_in_COD": 50.0, "Q_in": 6.667,
                    "T0_degC": 20.0, "duration_days": 90.0, "dt_days": 0.5})

    print("=== EC141 Thermophilic AD F2a — scenario summary ===")
    print(f"HRT            : {r['HRT_days']:.1f} days")
    print(f"Final T        : {r['temperature_degC'][-1]:.2f} C")
    print(f"Final CH4      : {r['Q_CH4_m3_day'][-1]:.1f} m3/day")
    print(f"Final biogas   : {r['Q_biogas_m3_day'][-1]:.1f} m3/day")
    print(f"Final energy   : {r['energy_kWh_day'][-1]:.1f} kWh/day")
    print(f"Heating demand : {r['heating_demand_W'][-1]/1000:.1f} kW")
    print(f"Final VFA      : {r['Sa_VFA'][-1]:.3f} kgCOD/m3")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        print(f"[plotly unavailable: {e}] — skipping HTML report.")
        return

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Substrate / VFA (kgCOD/m3)", "Biomass (kgCOD/m3)",
                        "Biogas & CH4 (m3/day)", "Temperature (C) & heat (kW)"),
        specs=[[{}, {}], [{}, {"secondary_y": True}]],
    )
    t = r["t"]
    fig.add_trace(go.Scatter(x=t, y=r["Xc"], name="Xc"), 1, 1)
    fig.add_trace(go.Scatter(x=t, y=r["Ss"], name="Ss"), 1, 1)
    fig.add_trace(go.Scatter(x=t, y=r["Sa_VFA"], name="VFA"), 1, 1)
    fig.add_trace(go.Scatter(x=t, y=r["Xaci"], name="acidogens"), 1, 2)
    fig.add_trace(go.Scatter(x=t, y=r["Xmet"], name="methanogens"), 1, 2)
    fig.add_trace(go.Scatter(x=t, y=r["Q_biogas_m3_day"], name="biogas"), 2, 1)
    fig.add_trace(go.Scatter(x=t, y=r["Q_CH4_m3_day"], name="CH4"), 2, 1)
    fig.add_trace(go.Scatter(x=t, y=r["temperature_degC"], name="T (C)"), 2, 2)
    fig.add_trace(go.Scatter(x=t, y=r["heating_demand_W"] / 1000.0,
                             name="heat (kW)"), 2, 2, secondary_y=True)
    fig.update_layout(title="EC141 Thermophilic AD — F2a ADM1 + Thermal",
                      height=700)
    if out_html is None:
        out_html = os.path.join(os.path.dirname(__file__), "..",
                                "simulation_report.html")
    fig.write_html(out_html)
    print(f"Report written: {out_html}")


if __name__ == "__main__":
    run_report()
