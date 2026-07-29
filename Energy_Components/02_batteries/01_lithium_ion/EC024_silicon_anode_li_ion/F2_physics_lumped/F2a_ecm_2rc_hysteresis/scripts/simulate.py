"""
EC024 -- Silicon-Anode Li-ion (Si/NMC) -- F2a Thevenin ECM
Optional Plotly simulation report. Plotly import is guarded; absence of
plotly does not crash the script (prints a notice instead).
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run_scenarios():
    cm = ComponentModel()
    # 1C discharge then rest, full-window charge for hysteresis loop
    dis = cm.predict({"current_A": 3.5, "soc0": 0.95, "dt": 5.0, "duration_s": 3000.0})
    chg = cm.predict({"current_A": -3.5, "soc0": 0.05, "dt": 5.0, "duration_s": 3000.0})
    return dis, chg


def main():
    dis, chg = run_scenarios()
    print(f"Discharge: SOC {dis['soc'][0]:.2f}->{dis['soc'][-1]:.2f}, "
          f"V_end={dis['voltage'][-1]:.3f} V, T_end={dis['temperature'][-1]:.2f} K")
    print(f"Charge:    SOC {chg['soc'][0]:.2f}->{chg['soc'][-1]:.2f}, "
          f"V_end={chg['voltage'][-1]:.3f} V, h_end={chg['hysteresis'][-1]:.2f}")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        print(f"[simulate] plotly not available ({e}); skipping HTML report.")
        return

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("OCV hysteresis loop (V vs SOC)",
                        "Terminal voltage vs time",
                        "Temperature vs time",
                        "Si swelling strain vs SOC"),
    )
    fig.add_trace(go.Scatter(x=dis["soc"], y=dis["ocv"], name="discharge OCV"), 1, 1)
    fig.add_trace(go.Scatter(x=chg["soc"], y=chg["ocv"], name="charge OCV"), 1, 1)
    fig.add_trace(go.Scatter(x=dis["t"], y=dis["voltage"], name="V discharge"), 1, 2)
    fig.add_trace(go.Scatter(x=dis["t"], y=dis["temperature"], name="T"), 2, 1)
    fig.add_trace(go.Scatter(x=dis["soc"], y=dis["swelling_strain"]*100, name="strain %"), 2, 2)

    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print(f"[simulate] wrote {out}")


if __name__ == "__main__":
    main()
