"""
EC066 -- Offshore Floating Wind Turbine -- F2a Aero-Hydro
Optional Plotly simulation report. Plotly import is guarded; absence won't crash.
Run: python3 scripts/simulate.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run_scenarios():
    cm = ComponentModel()
    scenarios = {}

    # Steady below-rated wind, calm vs rough sea (shows aero-hydro coupling)
    scenarios["calm_9ms"] = cm.predict(
        {"wind_speed_ms": 9.0, "duration_s": 200.0, "dt": 0.1,
         "wave_height_m": 0.0})
    scenarios["rough_9ms"] = cm.predict(
        {"wind_speed_ms": 9.0, "duration_s": 200.0, "dt": 0.1,
         "wave_height_m": 5.0, "wave_period_s": 9.0})
    # Above-rated wind (power capping + larger thrust/pitch)
    scenarios["rated_18ms"] = cm.predict(
        {"wind_speed_ms": 18.0, "duration_s": 200.0, "dt": 0.1,
         "wave_height_m": 4.0, "wave_period_s": 11.0})
    return scenarios


def build_report(scenarios, out_html):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        print(f"[simulate] Plotly unavailable ({e}); skipping HTML report.")
        return False

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=("Rotor speed", "Electrical power",
                        "Platform surge", "Platform pitch",
                        "Relative wind at hub (V_rel)", "Power coefficient Cp"))

    for name, r in scenarios.items():
        t = r["t"]
        fig.add_trace(go.Scatter(x=t, y=r["rotor_speed"], name=f"{name} Omega"),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=t, y=r["power_elec"] / 1e6, name=f"{name} P"),
                      row=1, col=2)
        fig.add_trace(go.Scatter(x=t, y=r["surge"], name=f"{name} surge"),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=t, y=r["pitch_deg"], name=f"{name} pitch"),
                      row=2, col=2)
        fig.add_trace(go.Scatter(x=t, y=r["V_rel"], name=f"{name} V_rel"),
                      row=3, col=1)
        fig.add_trace(go.Scatter(x=t, y=r["cp"], name=f"{name} Cp"),
                      row=3, col=2)

    fig.update_layout(height=1000, width=1200,
                      title_text="EC066 Floating Wind F2a — Aero-Hydro Coupled Response")
    fig.write_html(out_html)
    print(f"[simulate] Report written to {out_html}")
    return True


if __name__ == "__main__":
    sc = run_scenarios()
    for name, r in sc.items():
        print(f"{name:12s}  P_mean={r['power_elec_mean_MW']:.2f} MW  "
              f"Cp_max={r['cp_max']:.3f}  surge_pk={r['surge_peak_m']:.2f} m  "
              f"pitch_pk={r['pitch_peak_deg']:.2f} deg")
    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    build_report(sc, out)
