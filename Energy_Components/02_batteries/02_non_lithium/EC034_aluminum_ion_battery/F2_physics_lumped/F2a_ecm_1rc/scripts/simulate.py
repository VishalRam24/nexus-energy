"""
EC034 -- Aluminum-Ion Battery -- F2a Thevenin ECM
Optional Plotly simulation report. Plotly import is guarded so its absence
never crashes the build/test path.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run_scenarios():
    cm = ComponentModel()
    scenarios = {}

    # 1C discharge from full
    scenarios["discharge_1C"] = cm.predict(
        {"current_A": 1.0, "soc0": 0.98, "dt": 5.0, "duration_s": 3000.0})
    # High-rate 10C discharge (Al-ion fast capability)
    scenarios["discharge_10C"] = cm.predict(
        {"current_A": 10.0, "soc0": 0.98, "dt": 1.0, "duration_s": 300.0})
    # Charge
    scenarios["charge_2C"] = cm.predict(
        {"current_A": -2.0, "soc0": 0.2, "dt": 5.0, "duration_s": 1500.0})

    return scenarios


def build_report(out_html=None):
    if out_html is None:
        out_html = os.path.join(os.path.dirname(__file__), "..",
                                "simulation_report.html")
    scenarios = run_scenarios()
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:  # plotly not installed -> skip gracefully
        print(f"[simulate] Plotly unavailable ({e}); skipping HTML report.")
        for name, r in scenarios.items():
            print(f"  {name}: SOC {r['soc'][0]:.3f}->{r['soc'][-1]:.3f}, "
                  f"V {r['voltage'][0]:.3f}->{r['voltage'][-1]:.3f} V, "
                  f"T {r['temperature'][0]:.2f}->{r['temperature'][-1]:.2f} K")
        return None

    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "Voltage vs time", "SOC vs time", "Temperature vs time", "OCV vs SOC"))
    for name, r in scenarios.items():
        fig.add_trace(go.Scatter(x=r["t"], y=r["voltage"], name=f"V {name}"), 1, 1)
        fig.add_trace(go.Scatter(x=r["t"], y=r["soc"], name=f"SOC {name}"), 1, 2)
        fig.add_trace(go.Scatter(x=r["t"], y=r["temperature"], name=f"T {name}"), 2, 1)
        fig.add_trace(go.Scatter(x=r["soc"], y=r["ocv"], name=f"OCV {name}"), 2, 2)
    fig.update_layout(title="EC034 Aluminum-Ion Battery -- F2a Thevenin ECM",
                      height=800)
    fig.write_html(out_html)
    print(f"[simulate] Wrote {out_html}")
    return out_html


if __name__ == "__main__":
    build_report()
