"""
EC190 -- LNG Regasification Terminal -- F2a Physics-Lumped
Optional Plotly simulation report. Plotly import is guarded so absence
never crashes the build/test pipeline.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run_scenarios():
    cm = ComponentModel()

    # Scenario A: cold start to steady state
    rA = cm.predict({"sendout_rate_ton_per_h": 500.0, "T_metal0_K": 160.0,
                     "duration_s": 7200.0, "dt": 20.0})

    # Scenario B: diurnal demand ramp (send-out tracks demand)
    def demand(t):
        # 300 t/h baseline, +400 t/h peak around mid-window
        return 300.0 + 400.0 * max(0.0, np.sin(np.pi * t / 7200.0))
    rB = cm.predict({"sendout_rate_ton_per_h": demand, "T_metal0_K": 280.0,
                     "duration_s": 7200.0, "dt": 30.0})

    # Scenario C: cold-energy potential vs send-out rate
    rates = np.linspace(50, 2000, 25)
    cold = np.array([cm._model.cold_exergy_W(r, 288.15) for r in rates]) / 1e6
    heat = np.array([cm._model.process_heat_demand_W(r) for r in rates]) / 1e6
    return rA, rB, (rates, cold, heat)


def build_report(path=None):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:  # plotly absent -> skip gracefully
        print(f"[simulate] plotly unavailable ({e}); skipping HTML report.")
        return None

    rA, rB, (rates, cold, heat) = run_scenarios()
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Vaporizer metal T (cold start)",
                        "Heat flows (cold start)",
                        "Send-out tracking diurnal demand",
                        "Cold-energy potential vs send-out"),
    )
    fig.add_trace(go.Scatter(x=rA["t"]/3600, y=rA["T_metal"], name="T_metal"), 1, 1)
    fig.add_trace(go.Scatter(x=rA["t"]/3600, y=rA["Q_source_W"]/1e6, name="Q_source"), 1, 2)
    fig.add_trace(go.Scatter(x=rA["t"]/3600, y=rA["Q_process_W"]/1e6, name="Q_process"), 1, 2)
    fig.add_trace(go.Scatter(x=rB["t"]/3600, y=rB["sendout_kg_s"], name="sendout kg/s"), 2, 1)
    fig.add_trace(go.Scatter(x=rates, y=heat, name="regas heat MW"), 2, 2)
    fig.add_trace(go.Scatter(x=rates, y=cold, name="cold exergy MW"), 2, 2)

    if path is None:
        path = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(path)
    print(f"[simulate] report written to {path}")
    return path


if __name__ == "__main__":
    build_report()
