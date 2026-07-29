"""
EC131 -- Tidal Barrage -- F2a Physics-Lumped Basin ODE
Optional Plotly simulation report. Plotly import is guarded so absence does
not crash. Generates simulation_report.html in the model folder.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run(n_cycles=2, flood_gen=False):
    cm = ComponentModel()
    r = cm.predict({"n_cycles": n_cycles, "flood_gen": flood_gen, "n_eval": 3000})

    print(f"Energy/cycle      : {r['energy_per_cycle_MWh']:.1f} MWh")
    print(f"Theoretical/cycle : {r['theoretical_energy_per_cycle_MWh']:.1f} MWh")
    print(f"Average power     : {r['avg_power_MW']:.1f} MW")
    print(f"Peak power        : {r['peak_power_MW']:.1f} MW")
    print(f"Volume in / out   : {r['volume_in_m3']:.3e} / {r['volume_out_m3']:.3e} m3")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:  # pragma: no cover
        print(f"[simulate] Plotly unavailable ({e}); skipping HTML report.")
        return r

    t_h = r["t"] / 3600.0
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=("Sea vs Basin level", "Head & Flow", "Electrical power"),
    )
    fig.add_trace(go.Scatter(x=t_h, y=r["z_sea"], name="Sea level [m]"), 1, 1)
    fig.add_trace(go.Scatter(x=t_h, y=r["z_basin"], name="Basin level [m]"), 1, 1)
    fig.add_trace(go.Scatter(x=t_h, y=r["head"], name="Head [m]"), 2, 1)
    fig.add_trace(go.Scatter(x=t_h, y=r["flow"] / 1e3, name="Flow [10^3 m3/s]"), 2, 1)
    fig.add_trace(go.Scatter(x=t_h, y=r["power"] / 1e6, name="Power [MW]"), 3, 1)
    fig.update_xaxes(title_text="Time [h]", row=3, col=1)

    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(os.path.abspath(out))
    print(f"[simulate] Report written to {os.path.abspath(out)}")
    return r


if __name__ == "__main__":
    run()
