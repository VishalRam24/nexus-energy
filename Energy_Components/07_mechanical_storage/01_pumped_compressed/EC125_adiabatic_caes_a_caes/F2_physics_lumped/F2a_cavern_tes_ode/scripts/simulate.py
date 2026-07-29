"""
EC125 — Adiabatic CAES (A-CAES) — F2a Physics-Lumped
Simulation scenarios + optional interactive Plotly report.
Plotly import is wrapped so its absence never crashes the run.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run_scenarios():
    cm = ComponentModel()
    m = cm._model

    # 1 h charge from empty, cold TES
    charge = cm.predict({"mode": "charge", "m_dot": 100.0, "duration_s": 3600.0,
                         "dt": 60.0, "soc0": 0.0, "T_tes0": m.T_tes_ambient})
    # 1 h discharge from full, hot TES
    discharge = cm.predict({"mode": "discharge", "m_dot": 100.0, "duration_s": 3600.0,
                            "dt": 60.0, "soc0": 1.0, "T_tes0": m.T_tes_design})
    # 24 h idle decay
    idle = cm.predict({"mode": "idle", "m_dot": 0.0, "duration_s": 24 * 3600.0,
                       "dt": 600.0, "soc0": 0.5, "T_tes0": m.T_tes_design})
    rte, ch, dis = m.round_trip_simulation(m_dot=100.0, charge_s=3600.0, dt=60.0)
    print(f"Charge: E_in   = {charge['E_elec_kwh']:.1f} kWh, TES {charge['T_tes'][-1]:.1f} K")
    print(f"Disch.: E_out  = {discharge['E_elec_kwh']:.1f} kWh")
    print(f"Round-trip RTE = {rte:.3f}   (diabatic ref {m.rte_diabatic_ref:.2f})")
    print(f"Fuel power     = {charge['fuel_power_kw'][-1]:.1f} kW (always 0)")
    return charge, discharge, idle


def build_report(out_html=None):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        print(f"[simulate] Plotly unavailable ({e}); skipping HTML report.")
        return None

    charge, discharge, idle = run_scenarios()
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=("Charge: SOC & TES temp", "Discharge: SOC & TES temp",
                                        "Idle: TES decay", "Cavern pressure (charge)"))
    fig.add_trace(go.Scatter(x=charge["t"] / 60, y=charge["soc"], name="SOC (chg)"), 1, 1)
    fig.add_trace(go.Scatter(x=charge["t"] / 60, y=charge["T_tes"], name="T_tes (chg)", yaxis="y2"), 1, 1)
    fig.add_trace(go.Scatter(x=discharge["t"] / 60, y=discharge["soc"], name="SOC (dis)"), 1, 2)
    fig.add_trace(go.Scatter(x=idle["t"] / 3600, y=idle["T_tes"], name="T_tes idle"), 2, 1)
    fig.add_trace(go.Scatter(x=charge["t"] / 60, y=charge["pressure"] / 1e5, name="p [bar]"), 2, 2)
    fig.update_layout(title="EC125 A-CAES F2a — Cavern + TES Coupled ODE", height=720)

    if out_html is None:
        out_html = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out_html)
    print(f"[simulate] Report written to {out_html}")
    return out_html


if __name__ == "__main__":
    build_report()
