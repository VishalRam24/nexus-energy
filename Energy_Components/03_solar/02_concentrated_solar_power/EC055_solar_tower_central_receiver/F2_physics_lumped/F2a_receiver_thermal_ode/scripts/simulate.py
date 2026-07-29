"""
EC055 -- Solar Tower / Central Receiver CSP -- F2a Physics-Lumped
Optional Plotly report: diurnal receiver dynamics. Plotly import is guarded so
absence does not crash. Run: python3 scripts/simulate.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run_diurnal():
    cm = ComponentModel()
    dur = 24 * 3600.0
    dt = 600.0
    n = int(dur / dt) + 1
    t = np.linspace(0, dur, n)
    hour = t / 3600.0
    dni = np.where((hour > 6) & (hour < 18),
                   950.0 * np.maximum(0.0, np.sin(np.pi * (hour - 6) / 12.0)), 0.0)
    zen = np.clip(np.abs(hour - 12) * 7.0, 0.0, 85.0)
    r = cm.predict({"dni": dni, "solar_zenith": zen, "T_amb_C": 25.0,
                    "wind_speed": 5.0, "mdot_salt": 150.0, "T0_C": 290.0,
                    "dt": dt, "duration_s": dur})
    r["hour"] = hour
    return r


def main():
    r = run_diurnal()
    print("Diurnal central-receiver simulation:")
    print(f"  peak T_receiver  : {np.max(r['T_receiver_C']):.1f} degC")
    print(f"  peak Q_to_PB     : {np.max(r['Q_thermal_to_PB_MWth']):.1f} MWth")
    print(f"  peak P_electric  : {np.max(r['P_electric_MWe']):.1f} MWe")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:  # pragma: no cover
        print(f"[plotly unavailable: {e}] skipping HTML report.")
        return

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=("Receiver temperature", "Power flows",
                        "Efficiencies"))
    h = r["hour"]
    fig.add_trace(go.Scatter(x=h, y=r["T_receiver_C"], name="T_receiver [C]"), 1, 1)
    fig.add_trace(go.Scatter(x=h, y=r["Q_field_W"] / 1e6, name="Q_field [MW]"), 2, 1)
    fig.add_trace(go.Scatter(x=h, y=r["Q_thermal_to_PB_MWth"], name="Q_to_PB [MWth]"), 2, 1)
    fig.add_trace(go.Scatter(x=h, y=r["Q_loss_W"] / 1e6, name="Q_loss [MW]"), 2, 1)
    fig.add_trace(go.Scatter(x=h, y=r["P_electric_MWe"], name="P_elec [MWe]"), 2, 1)
    fig.add_trace(go.Scatter(x=h, y=r["field_efficiency"], name="eta_field"), 3, 1)
    fig.add_trace(go.Scatter(x=h, y=r["receiver_efficiency"], name="eta_receiver"), 3, 1)
    fig.add_trace(go.Scatter(x=h, y=r["overall_efficiency"], name="eta_overall"), 3, 1)
    fig.update_xaxes(title_text="hour of day", row=3, col=1)
    fig.update_layout(title="EC055 Solar Tower F2a -- Lumped Receiver Dynamics",
                      height=900)
    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
