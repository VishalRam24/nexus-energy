"""
EC155 -- Geothermal District Heating -- F2a Lumped Network Thermal Transient
Simulation scenarios + optional Plotly HTML report.  Plotly is optional; if it
is not installed the script still runs and prints a text summary.
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


def diurnal_load(t):
    """Cold-morning / mild-afternoon district heat demand over a day (kW)."""
    hod = (t / 3600.0) % 24.0
    base = 5500.0
    morning = 3500.0 * np.exp(-0.5 * ((hod - 7.0) / 2.0) ** 2)
    evening = 2500.0 * np.exp(-0.5 * ((hod - 19.0) / 2.5) ** 2)
    return base + morning + evening


def main():
    cm = ComponentModel()
    r = cm.predict({
        "Q_load_kW": diurnal_load,
        "T_supply0": 70.0,
        "T_return0": 40.0,
        "dt": 300.0,
        "duration_s": 2 * 86400.0,
        "boiler_on": True,
    })

    t_h = r["t"] / 3600.0
    print(f"Geothermal DH 2-day simulation: success={r['success']}")
    print(f"  T_supply range : {r['T_supply'].min():.1f} - {r['T_supply'].max():.1f} C")
    print(f"  T_return range : {r['T_return'].min():.1f} - {r['T_return'].max():.1f} C")
    print(f"  Q_geo mean     : {r['Q_geo_kW'].mean():.0f} kW")
    print(f"  Q_boiler peak  : {r['Q_boiler_kW'].max():.0f} kW")
    print(f"  Q_cascade mean : {r['Q_cascade_kW'].mean():.0f} kW")

    if not _HAVE_PLOTLY:
        print("Plotly not installed -- skipping HTML report.")
        return

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Network temperatures", "Heat flows"))
    fig.add_trace(go.Scatter(x=t_h, y=r["T_supply"], name="T_supply"), 1, 1)
    fig.add_trace(go.Scatter(x=t_h, y=r["T_return"], name="T_return"), 1, 1)
    fig.add_trace(go.Scatter(x=t_h, y=r["T_reinject"], name="T_reinject"), 1, 1)
    fig.add_trace(go.Scatter(x=t_h, y=r["Q_load_kW"], name="Q_load"), 2, 1)
    fig.add_trace(go.Scatter(x=t_h, y=r["Q_geo_kW"], name="Q_geo"), 2, 1)
    fig.add_trace(go.Scatter(x=t_h, y=r["Q_boiler_kW"], name="Q_boiler"), 2, 1)
    fig.add_trace(go.Scatter(x=t_h, y=r["Q_cascade_kW"], name="Q_cascade"), 2, 1)
    fig.update_xaxes(title_text="time (h)", row=2, col=1)
    fig.update_yaxes(title_text="degC", row=1, col=1)
    fig.update_yaxes(title_text="kW", row=2, col=1)
    fig.update_layout(title="EC155 Geothermal District Heating -- F2a Network Transient")

    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
