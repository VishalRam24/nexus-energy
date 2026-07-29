"""
EC038 -- Fe-Cr Flow Battery -- F2a Physics-Lumped: simulation scenarios + Plotly report.
Plotly import is guarded so absence does not crash.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run_scenarios():
    cm = ComponentModel()
    scen = {}

    # Charge half-cycle
    scen["charge_50A"] = cm.predict(
        {"current_A": -50.0, "soc0": 0.2, "T0": 298.15, "dt": 30.0, "duration_s": 7200.0})
    # Discharge half-cycle
    scen["discharge_50A"] = cm.predict(
        {"current_A": 50.0, "soc0": 0.9, "T0": 298.15, "dt": 30.0, "duration_s": 7200.0})

    # Polarisation sweep at fixed SOC/T
    m = cm._model
    soc, T = 0.5, 308.15
    I_sweep = np.linspace(-150, 150, 121)
    V_sweep = np.array([m.terminal_voltage(I, soc, T) for I in I_sweep])
    P_sweep = V_sweep * I_sweep
    scen["polarisation"] = {"I": I_sweep, "V": V_sweep, "P": P_sweep}
    return scen


def build_report(scen, out_html):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        print(f"[simulate] Plotly unavailable ({e}); skipping HTML report.")
        return None

    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "Charge: SOC & Voltage", "Discharge: SOC & Voltage",
        "Stack polarisation (V-I)", "Temperature & H2 parasitic (charge)"))

    c = scen["charge_50A"]
    fig.add_trace(go.Scatter(x=c["t"]/3600, y=c["soc"], name="SOC (chg)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=c["t"]/3600, y=c["voltage"], name="V (chg)", yaxis="y2"), row=1, col=1)

    d = scen["discharge_50A"]
    fig.add_trace(go.Scatter(x=d["t"]/3600, y=d["soc"], name="SOC (dis)"), row=1, col=2)
    fig.add_trace(go.Scatter(x=d["t"]/3600, y=d["voltage"], name="V (dis)"), row=1, col=2)

    p = scen["polarisation"]
    fig.add_trace(go.Scatter(x=p["I"], y=p["V"], name="V-I"), row=2, col=1)

    fig.add_trace(go.Scatter(x=c["t"]/3600, y=c["temperature"], name="T (chg)"), row=2, col=2)
    fig.add_trace(go.Scatter(x=c["t"]/3600, y=c["I_H2"], name="I_H2"), row=2, col=2)

    fig.update_layout(title="EC038 Fe-Cr Flow Battery -- F2a Physics-Lumped", height=800)
    fig.write_html(out_html)
    print(f"[simulate] Report written: {out_html}")
    return out_html


if __name__ == "__main__":
    scen = run_scenarios()
    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    build_report(scen, out)
    c = scen["charge_50A"]
    print(f"Charge: SOC {c['soc'][0]:.3f}->{c['soc'][-1]:.3f}, "
          f"T {c['temperature'][0]:.1f}->{c['temperature'][-1]:.1f} K")
