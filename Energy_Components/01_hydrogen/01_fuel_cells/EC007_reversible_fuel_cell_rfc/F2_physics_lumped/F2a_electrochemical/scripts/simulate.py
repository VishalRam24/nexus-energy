"""
EC007 -- RFC -- F2a Bidirectional Electrochemical
Optional Plotly report: bidirectional V-j polarization, a charge->discharge
thermal transient, and round-trip efficiency vs current density.
Plotly import is wrapped so absence does not crash.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def build_report(out_html=None):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        print(f"Plotly unavailable ({e}); skipping HTML report.")
        return None

    cm = ComponentModel()
    m = cm._model
    T, P_h2, P_o2 = 353.15, 1.0, 0.21

    # Bidirectional polarization curve
    j_fc = np.linspace(0.0, 1.45, 80)
    j_el = -np.linspace(0.0, 2.9, 80)
    V_fc = [m.cell_voltage(j, T, P_h2, P_o2) for j in j_fc]
    V_el = [m.cell_voltage(j, T, P_h2, P_o2) for j in j_el]

    # Round-trip efficiency vs |j|
    j_mag = np.linspace(0.1, 1.4, 40)
    rt = [m.round_trip_efficiency(j, T, P_h2, P_o2) for j in j_mag]

    # Charge then discharge thermal transient
    def cycle_j(t):
        return -0.8 if t < 150 else 0.8
    cyc = m.simulate(cycle_j, 333.15, P_h2, P_o2, 1.0, 300.0)

    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "Bidirectional polarization (V vs j)",
        "Round-trip voltaic efficiency vs |j|",
        "Charge -> Discharge thermal transient",
        "Cell voltage during cycle"))

    fig.add_trace(go.Scatter(x=j_fc, y=V_fc, name="FC (discharge)"), 1, 1)
    fig.add_trace(go.Scatter(x=j_el, y=V_el, name="EL (charge)"), 1, 1)
    fig.add_trace(go.Scatter(x=j_mag, y=rt, name="eta_rt"), 1, 2)
    fig.add_trace(go.Scatter(x=cyc["t"], y=cyc["temperature"], name="T [K]"), 2, 1)
    fig.add_trace(go.Scatter(x=cyc["t"], y=cyc["voltage"], name="V [V]"), 2, 2)

    fig.update_xaxes(title_text="j [A/cm2]", row=1, col=1)
    fig.update_yaxes(title_text="V [V]", row=1, col=1)
    fig.update_layout(title="EC007 RFC F2a -- Bidirectional Electrochemical + Thermal",
                      height=800, showlegend=True)

    if out_html is None:
        out_html = os.path.join(os.path.dirname(__file__), "..", "model_files",
                                "EC007_rfc_f2a_report.html")
    fig.write_html(out_html)
    print(f"Report written: {out_html}")
    return out_html


if __name__ == "__main__":
    build_report()
