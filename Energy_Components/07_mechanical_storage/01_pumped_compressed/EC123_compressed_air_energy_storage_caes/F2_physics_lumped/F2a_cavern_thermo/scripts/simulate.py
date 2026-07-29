"""
EC123 — CAES (Diabatic) — F2a Cavern Thermodynamics
Plotly simulation report (optional). Plotly import is guarded so its absence
does not crash the module.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _HAVE_PLOTLY = True
except Exception:
    _HAVE_PLOTLY = False


def run_report(out_html=None):
    cm = ComponentModel()
    m = cm._model

    # Charge from p_min to near p_max, then discharge back down.
    charge = m.simulate("charge", 100.0, m.T_rock, m.p_min, 60.0, 6 * 3600.0)
    discharge = m.simulate("discharge", 220.0, charge["temperature"][-1],
                           charge["pressure"][-1], 60.0, 3 * 3600.0)

    rte = m.round_trip_efficiency()
    el = m.electric_rte()
    hr = m.heat_rate()
    print(f"RTE (incl. fuel) = {rte:.3f} | electric RTE = {el:.3f} | heat rate = {hr:.0f} kJ/kWh_e")
    print(f"Charge: {charge['E_elec_J']/3.6e9:.1f} MWh in; "
          f"Discharge: {discharge['E_elec_J']/3.6e9:.1f} MWh out, "
          f"{discharge['m_fuel_kg']/1000:.1f} t gas")

    if not _HAVE_PLOTLY:
        print("plotly not installed — skipping HTML report.")
        return

    if out_html is None:
        out_html = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")

    t_c = charge["t"] / 3600.0
    t_d = charge["t"][-1] / 3600.0 + discharge["t"] / 3600.0

    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "Cavern Pressure [bar]", "Cavern Temperature [K]",
        "State of Charge [-]", "Electrical / Fuel Power [MW]"))

    fig.add_trace(go.Scatter(x=t_c, y=charge["pressure"]/1e5, name="charge P"), 1, 1)
    fig.add_trace(go.Scatter(x=t_d, y=discharge["pressure"]/1e5, name="discharge P"), 1, 1)
    fig.add_trace(go.Scatter(x=t_c, y=charge["temperature"], name="charge T"), 1, 2)
    fig.add_trace(go.Scatter(x=t_d, y=discharge["temperature"], name="discharge T"), 1, 2)
    fig.add_trace(go.Scatter(x=t_c, y=charge["soc"], name="charge SOC"), 2, 1)
    fig.add_trace(go.Scatter(x=t_d, y=discharge["soc"], name="discharge SOC"), 2, 1)
    fig.add_trace(go.Scatter(x=t_c, y=charge["P_elec"]/1e6, name="P_elec charge"), 2, 2)
    fig.add_trace(go.Scatter(x=t_d, y=discharge["P_elec"]/1e6, name="P_elec discharge"), 2, 2)
    fig.add_trace(go.Scatter(x=t_d, y=discharge["P_fuel"]/1e6, name="P_fuel"), 2, 2)

    fig.update_layout(title=f"EC123 Diabatic CAES F2a — RTE(fuel)={rte:.3f}, electric RTE={el:.3f}",
                      height=800, width=1100)
    fig.write_html(out_html)
    print(f"Report written: {out_html}")


if __name__ == "__main__":
    run_report()
