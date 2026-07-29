"""
EC118 -- SMR F2a Point Kinetics + Lumped Thermal -- simulation scenarios.

Generates an interactive Plotly report (simulation_report.html) showing:
  1. Reactivity step -> self-stabilizing power & temperature transient.
  2. Deep load-following 100% -> 60% -> 100% with natural-circulation flow.

Plotly is optional: the import is guarded so absence does not crash the script.
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
    m = ComponentModel()

    # Scenario 1: +100 pcm step reactivity insertion (feedback arrests it)
    step = m.predict({"rho_step": 0.001, "dt": 0.5, "duration_s": 600.0,
                      "t_insert": 1.0})

    # Scenario 2: deep load-following 100% -> 60% -> 100%
    def sched(t):
        if t < 2000: return 1.0
        if t < 5000: return 0.6
        return 1.0
    lf = m.predict({"mode": "load_follow", "power_schedule": sched,
                    "duration_s": 8000.0, "dt": 10.0})

    print("Scenario 1 (+100 pcm step):")
    print(f"  peak power frac = {step['n'].max():.3f}, settled = {step['n'][-1]:.3f}")
    print(f"  T_f {step['T_f'][0]:.1f} -> {step['T_f'][-1]:.1f} K")
    print("Scenario 2 (load-following 100/60/100%):")
    print(f"  max tracking error = "
          f"{np.abs(lf['power_fraction']-lf['power_demand']).max():.3f}")

    if not _HAVE_PLOTLY:
        print("[plotly not installed -- skipping HTML report]")
        return

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Step +100 pcm: power self-stabilizes",
                        "Step +100 pcm: fuel & coolant temperature",
                        "Load-following: power tracks demand",
                        "Load-following: natural-circulation flow"))

    fig.add_trace(go.Scatter(x=step["t"], y=step["n"], name="power frac"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=step["t"], y=step["T_f"], name="T_fuel [K]"),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=step["t"], y=step["T_m"], name="T_coolant [K]"),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=lf["t"], y=lf["power_demand"], name="demand",
                             line=dict(dash="dash")), row=2, col=1)
    fig.add_trace(go.Scatter(x=lf["t"], y=lf["power_fraction"], name="actual"),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=lf["t"], y=lf["flow_fraction"], name="flow frac"),
                  row=2, col=2)

    fig.update_layout(height=720, width=1100,
                      title_text="EC118 SMR F2a -- Point Kinetics + Lumped Thermal")
    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print(f"Wrote {os.path.abspath(out)}")


if __name__ == "__main__":
    run()
