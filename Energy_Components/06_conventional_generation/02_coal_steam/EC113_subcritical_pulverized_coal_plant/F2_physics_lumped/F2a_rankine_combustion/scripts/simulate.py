"""
EC113 -- Subcritical Pulverized Coal Plant -- F2a Physics-Lumped
Optional Plotly simulation report.  Plotly import is guarded so its absence
does not crash the model/test pipeline.  Run: python3 scripts/simulate.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _HAVE_PLOTLY = True
except Exception:
    _HAVE_PLOTLY = False


def run():
    cm = ComponentModel()

    # Scenario 1: cold-start drum warm-up to full load
    warm = cm.predict({"part_load_ratio": 1.0, "T0_drum_K": 300.0,
                       "dt": 20.0, "duration_s": 12000.0})

    # Scenario 2: 50% -> 100% load step
    step = cm.predict({"part_load_ratio": lambda t: 0.5 if t < 4000 else 1.0,
                       "T0_drum_K": float(warm["T_sat_drum"][0]),
                       "dt": 20.0, "duration_s": 9000.0})

    # Scenario 3: net efficiency vs part-load
    plrs = np.linspace(0.30, 1.0, 25)
    eta = np.array([cm.steady_state(p)["eta_net"] for p in plrs])
    ci = np.array([cm._model.co2_intensity_g_per_kwh(p) for p in plrs])

    ss = cm.steady_state(1.0)
    print(f"Design point: {ss['power_net_mw']:.0f} MW_e | "
          f"eta_net={ss['eta_net']*100:.1f}% | boiler={ss['eta_boiler']*100:.1f}% | "
          f"cycle={ss['eta_cycle']*100:.1f}% | Carnot={ss['eta_carnot']*100:.1f}%")
    print(f"Coal={ss['coal_rate_kgs']:.1f} kg/s | steam={ss['steam_rate_kgs']:.0f} kg/s | "
          f"CO2={ss['co2_intensity_g_per_kwh']:.0f} g/kWh")

    if not _HAVE_PLOTLY:
        print("[simulate] plotly not installed -- skipping HTML report.")
        return

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Drum warm-up (cold start)", "50%->100% load step",
                        "Net efficiency vs part load", "CO2 intensity vs part load"))

    fig.add_trace(go.Scatter(x=warm["t"] / 60, y=warm["T_drum"] - 273.15,
                             name="T_drum"), row=1, col=1)
    fig.add_trace(go.Scatter(x=warm["t"] / 60,
                             y=warm["T_sat_drum"] - 273.15,
                             name="T_sat", line=dict(dash="dash")), row=1, col=1)

    fig.add_trace(go.Scatter(x=step["t"] / 60, y=step["coal_rate_kgs"],
                             name="coal kg/s"), row=1, col=2)
    fig.add_trace(go.Scatter(x=step["t"] / 60, y=step["steam_rate_kgs"],
                             name="steam kg/s"), row=1, col=2)

    fig.add_trace(go.Scatter(x=plrs, y=eta * 100, name="eta_net %"), row=2, col=1)
    fig.add_trace(go.Scatter(x=plrs, y=ci, name="CO2 g/kWh"), row=2, col=2)

    fig.update_layout(height=750, width=1100,
                      title_text="EC113 Subcritical PC Plant -- F2a Physics-Lumped")
    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print(f"[simulate] wrote {os.path.abspath(out)}")


if __name__ == "__main__":
    run()
