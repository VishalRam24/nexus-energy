"""
EC206 -- CO2 Mineralization F2a -- simulation scenarios + optional Plotly report.
Plotly import is wrapped so its absence does not crash the script.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run_scenarios():
    cm = ComponentModel()
    # Batch carbonation at three temperatures
    temps = [398.15, 458.15, 503.15]      # 125, 185, 230 C
    runs = {}
    for T in temps:
        runs[T] = cm.predict({"T0_K": T, "P_CO2_atm": 115.0,
                              "duration_s": 7200.0, "dt": 60.0})
    # Particle-size sweep at reference T/P
    sizes = [1.85e-5, 3.7e-5, 7.4e-5]      # 18, 37, 74 um
    size_runs = {R0: cm.predict({"particle_radius_m": R0, "duration_s": 7200.0,
                                 "dt": 60.0}) for R0 in sizes}
    return runs, size_runs


def make_report(path=None):
    runs, size_runs = run_scenarios()
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "..",
                            "simulation_report.html")
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception:
        print("Plotly not available -- skipping HTML report.")
        for T, r in runs.items():
            print(f"T0={T-273.15:.0f}C  Xf={r['final_conversion']:.3f}  "
                  f"CO2={r['co2_stored_kg']:.1f} kg  Tpeak={r['peak_temperature_K']:.1f} K")
        return None

    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "Conversion vs time (T sweep)", "Slurry temperature (T sweep)",
        "Conversion vs time (particle size)", "Cumulative CO2 bound (T sweep)"))
    for T, r in runs.items():
        lbl = f"{T-273.15:.0f} C"
        fig.add_trace(go.Scatter(x=r["t"]/3600, y=r["conversion"], name=lbl), 1, 1)
        fig.add_trace(go.Scatter(x=r["t"]/3600, y=r["temperature"], name=lbl,
                                 showlegend=False), 1, 2)
        fig.add_trace(go.Scatter(x=r["t"]/3600, y=r["co2_bound_kg"], name=lbl,
                                 showlegend=False), 2, 2)
    for R0, r in size_runs.items():
        fig.add_trace(go.Scatter(x=r["t"]/3600, y=r["conversion"],
                                 name=f"{R0*1e6:.0f} um"), 2, 1)
    fig.update_xaxes(title_text="time [h]")
    fig.update_layout(title="EC206 CO2 Mineralization F2a -- Carbonation Kinetics",
                      height=800)
    fig.write_html(path)
    print(f"Report written: {path}")
    return path


if __name__ == "__main__":
    make_report()
