"""
EC093 — Adsorption Chiller — F2a — simulation + optional Plotly report.

Generates a cyclic-steady-state run and (if plotly is installed) an
interactive HTML report showing bed temperature, water uptake, cooling
power and a COP-vs-half-cycle map.  Plotly import is wrapped so its
absence never crashes the run.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run():
    cm = ComponentModel()
    r = cm.predict({"T_hot": 85, "T_cool": 30, "T_chilled": 14,
                    "t_half_cycle": 400, "n_cycles": 8})
    print("=== EC093 Adsorption Chiller — F2a cyclic steady state ===")
    print(f"thermal_COP        : {r['thermal_COP']:.3f}")
    print(f"cooling_power      : {r['cooling_power_kW']:.2f} kW")
    print(f"driving_heat       : {r['driving_heat_mean_kW']:.2f} kW")
    print(f"SCP                : {r['SCP_W_per_kg']:.1f} W/kg")
    print(f"uptake swing       : {r['dw_adsorbed']:.4f} kg/kg")

    # COP vs half-cycle map
    t_halves = [150, 250, 350, 450, 600, 800]
    cops, qcs = [], []
    for th in t_halves:
        rr = cm.predict({"t_half_cycle": th, "n_cycles": 6})
        cops.append(rr["thermal_COP"])
        qcs.append(rr["cooling_power_kW"])

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:  # plotly absent -> skip report
        print(f"[simulate] plotly unavailable ({e}); skipping HTML report.")
        return r

    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "Bed temperature (desorb / adsorb)", "Water uptake w(t)",
        "Cooling power Q_evap(t)", "COP & capacity vs half-cycle"))
    fig.add_trace(go.Scatter(x=r["t_des"], y=r["T_bed_des"] - 273.15,
                             name="desorbing bed"), 1, 1)
    fig.add_trace(go.Scatter(x=r["t_ads"], y=r["T_bed_ads"] - 273.15,
                             name="adsorbing bed"), 1, 1)
    fig.add_trace(go.Scatter(x=r["t_des"], y=r["w_des"], name="w desorb"), 1, 2)
    fig.add_trace(go.Scatter(x=r["t_ads"], y=r["w_ads"], name="w adsorb"), 1, 2)
    fig.add_trace(go.Scatter(x=r["t_ads"], y=r["Q_evap"] / 1000.0,
                             name="Q_evap [kW]"), 2, 1)
    fig.add_trace(go.Scatter(x=t_halves, y=cops, name="COP", mode="lines+markers"), 2, 2)
    fig.add_trace(go.Scatter(x=t_halves, y=qcs, name="Q_cool [kW]",
                             mode="lines+markers", yaxis="y2"), 2, 2)
    fig.update_layout(title="EC093 Adsorption Chiller — F2a Physics-Lumped",
                      height=750)
    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print(f"[simulate] wrote {os.path.abspath(out)}")
    return r


if __name__ == "__main__":
    run()
