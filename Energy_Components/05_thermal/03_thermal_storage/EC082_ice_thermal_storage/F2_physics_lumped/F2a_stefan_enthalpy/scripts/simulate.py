"""
EC082 -- Ice Thermal Storage -- F2a Stefan-Problem Enthalpy Model
Plotly HTML simulation report (charge -> discharge daily cycle).
Plotly import is wrapped so absence does not crash.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _HAVE_PLOTLY = True
except ImportError:
    _HAVE_PLOTLY = False

OUTPUT = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")


def run():
    m = ComponentModel()
    # Daily cycle: 8 h ice build at -6 C, idle 4 h, then discharge at 10 C.
    def brine(t):
        if t < 8 * 3600:
            return -6.0
        elif t < 12 * 3600:
            return 0.0
        else:
            return 10.0
    r = m.predict({"T_brine_C": brine, "T_amb_C": 22.0, "ice_fraction0": 0.0,
                   "dt": 300.0, "duration_s": 24 * 3600.0})
    t_h = r["t"] / 3600.0

    print("EC082 Ice TES F2a daily cycle:")
    print(f"  peak ice fraction : {r['ice_fraction'].max():.3f}")
    print(f"  end ice fraction  : {r['ice_fraction'][-1]:.3f}")
    print(f"  UA range          : {r['UA_eff_W_per_K'].min():.0f} .. "
          f"{r['UA_eff_W_per_K'].max():.0f} W/K")
    cooling = np.trapezoid(r["cooling_power_W"], r["t"]) / 3.6e6
    print(f"  cooling delivered : {cooling:.1f} kWh_th")

    if not _HAVE_PLOTLY:
        print("plotly not installed -- skipping HTML report.")
        return

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Ice fraction (SOC)", "Storage temperature [C]",
                        "Effective coil UA [W/K]", "Heat flows [kW]"),
    )
    fig.add_trace(go.Scatter(x=t_h, y=r["ice_fraction"], name="ice fraction"), 1, 1)
    fig.add_trace(go.Scatter(x=t_h, y=r["temperature_C"], name="T storage"), 1, 2)
    fig.add_trace(go.Scatter(x=t_h, y=r["UA_eff_W_per_K"], name="UA_eff"), 2, 1)
    fig.add_trace(go.Scatter(x=t_h, y=r["q_coil_W"] / 1e3, name="q_coil"), 2, 2)
    fig.add_trace(go.Scatter(x=t_h, y=r["q_loss_W"] / 1e3, name="q_loss"), 2, 2)
    fig.update_xaxes(title_text="time [h]")
    fig.update_layout(title="EC082 Ice TES F2a -- Stefan Enthalpy Daily Cycle",
                      height=720, width=1000)
    fig.write_html(OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    run()
