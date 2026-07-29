"""
EC035 -- NaS Battery -- F2a Thevenin ECM: Plotly simulation report.
Optional. Plotly import is guarded so its absence never crashes the build.
Run: python3 scripts/simulate.py  ->  ../simulation_report.html
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run_scenarios():
    cm = ComponentModel()
    # 1 h discharge at 0.4C (40 A) from full
    dis = cm.predict({"current_A": 40.0, "soc0": 0.95, "T0_K": 593.15,
                      "dt": 10.0, "duration_s": 3600.0})
    # idle hold by heater from a low start
    idle = cm.predict({"current_A": 0.0, "soc0": 0.5, "T0_K": 576.0,
                       "dt": 30.0, "duration_s": 7200.0})
    # R(T) sweep across the operating window
    m = cm._model
    Tsweep = np.linspace(m.T_op_min, m.T_op_max, 100)
    R0 = m.R0(Tsweep)
    socs = np.linspace(0, 1, 200)
    ocv = m.ocv(socs)
    return dis, idle, Tsweep, R0, socs, ocv


def build_report(path):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:  # pragma: no cover
        print(f"[simulate] Plotly unavailable ({e}); skipping HTML report.")
        return False

    dis, idle, Tsweep, R0, socs, ocv = run_scenarios()
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Discharge: terminal voltage", "Discharge: SOC",
            "Discharge: cell temperature", "Idle: heater holds T in band",
            "OCV(SOC) — NaS two-plateau", "Beta-alumina R0(T) Arrhenius",
        ),
    )
    fig.add_trace(go.Scatter(x=dis["t"] / 60, y=dis["voltage"], name="V_term"), 1, 1)
    fig.add_trace(go.Scatter(x=dis["t"] / 60, y=dis["soc"], name="SOC"), 1, 2)
    fig.add_trace(go.Scatter(x=dis["t"] / 60, y=dis["temperature"] - 273.15, name="T (C)"), 2, 1)
    fig.add_trace(go.Scatter(x=idle["t"] / 60, y=idle["temperature"] - 273.15, name="idle T (C)"), 2, 2)
    fig.add_trace(go.Scatter(x=socs, y=ocv, name="OCV"), 3, 1)
    fig.add_trace(go.Scatter(x=Tsweep - 273.15, y=R0 * 1e3, name="R0 (mOhm)"), 3, 2)
    fig.update_layout(height=1100, width=1200,
                      title_text="EC035 NaS Battery — F2a Thevenin ECM (physics-lumped)")
    fig.write_html(path)
    print(f"[simulate] Report written: {path}")
    return True


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    build_report(os.path.abspath(out))
