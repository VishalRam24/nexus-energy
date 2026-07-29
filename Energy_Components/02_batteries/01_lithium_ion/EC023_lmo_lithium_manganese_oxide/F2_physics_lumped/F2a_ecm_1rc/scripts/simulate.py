"""
EC023 -- LMO Battery -- F2a Thevenin ECM
Optional Plotly simulation report. Plotly import is wrapped so its absence
does not crash. Run: python3 scripts/simulate.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run_scenarios():
    cm = ComponentModel()

    # Scenario A: constant 1C (3 A) discharge from full
    a = cm.predict({"current_A": 3.0, "soc0": 1.0, "T0": 298.15,
                    "dt": 2.0, "duration_s": 3000.0})

    # Scenario B: dynamic drive cycle (charge / rest / discharge pulses)
    def drive(t):
        if t < 300:
            return 6.0          # discharge pulse
        if t < 600:
            return 0.0          # rest
        if t < 900:
            return -6.0         # charge pulse
        return 2.0              # mild discharge
    b = cm.predict({"current_A": drive, "soc0": 0.6, "T0": 298.15,
                    "dt": 1.0, "duration_s": 1500.0})

    # OCV(SOC) curve
    socs = np.linspace(0, 1, 200)
    ocv = cm._model.ocv(socs)
    return a, b, socs, ocv


def main():
    a, b, socs, ocv = run_scenarios()
    print(f"Scenario A (1C discharge): SOC {a['soc'][0]:.2f}->{a['soc'][-1]:.2f}, "
          f"V {a['voltage'][0]:.3f}->{a['voltage'][-1]:.3f} V, "
          f"T_peak {a['temperature'].max():.2f} K")
    print(f"Scenario B (drive cycle):  T_peak {b['temperature'].max():.2f} K, "
          f"Q_peak {b['heat_generation'].max():.3f} W")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        print(f"[simulate] Plotly unavailable ({e}); skipping HTML report.")
        return

    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "OCV vs SOC (spinel dual-plateau)", "1C discharge: voltage & SOC",
        "Drive cycle: temperature", "Drive cycle: heat generation"))
    fig.add_trace(go.Scatter(x=socs, y=ocv, name="OCV"), row=1, col=1)
    fig.add_trace(go.Scatter(x=a["t"], y=a["voltage"], name="V_term"), row=1, col=2)
    fig.add_trace(go.Scatter(x=a["t"], y=a["soc"], name="SOC", yaxis="y2"), row=1, col=2)
    fig.add_trace(go.Scatter(x=b["t"], y=b["temperature"], name="T [K]"), row=2, col=1)
    fig.add_trace(go.Scatter(x=b["t"], y=b["heat_generation"], name="Q [W]"), row=2, col=2)
    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print(f"[simulate] Report written to {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
