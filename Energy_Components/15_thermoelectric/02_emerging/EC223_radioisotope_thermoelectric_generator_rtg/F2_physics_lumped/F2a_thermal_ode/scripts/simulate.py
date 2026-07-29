"""
EC223 -- RTG F2a -- optional Plotly mission-lifetime report.
Plotly import is guarded so absence does not crash.
Run:  python3 scripts/simulate.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def main():
    cm = ComponentModel()
    r = cm.predict({"mission_years": 100.0, "n_points": 400})

    print("RTG mission summary (GPHS-RTG class, Pu-238):")
    for yr in [0, 14, 50, 100]:
        i = min(int(yr / 100.0 * (len(r["t_years"]) - 1)), len(r["t_years"]) - 1)
        print(f"  t={r['t_years'][i]:6.1f} yr  "
              f"Q_decay={r['Q_decay_W'][i]:6.0f} W  "
              f"T_hot={r['T_hot_K'][i]:6.0f} K  "
              f"P_e={r['P_electric_W'][i]:6.1f} W  "
              f"eta={r['eta_module'][i]*100:4.2f}%  "
              f"frac={r['power_fraction'][i]*100:5.1f}%")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception:
        print("\n[plotly not available -- skipping HTML report]")
        return

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Decay heat & electrical power",
                        "Hot-side temperature",
                        "Conversion efficiency vs bounds",
                        "Power fraction (of BOL)"))
    t = r["t_years"]
    fig.add_trace(go.Scatter(x=t, y=r["Q_decay_W"], name="Q_decay (W)"), 1, 1)
    fig.add_trace(go.Scatter(x=t, y=r["P_electric_W"], name="P_electric (W)", yaxis="y2"), 1, 1)
    fig.add_trace(go.Scatter(x=t, y=r["T_hot_K"], name="T_hot (K)"), 1, 2)
    fig.add_trace(go.Scatter(x=t, y=r["eta_module"] * 100, name="eta_module (%)"), 2, 1)
    fig.add_trace(go.Scatter(x=t, y=r["eta_zt_max"] * 100, name="eta_ZT_max (%)"), 2, 1)
    fig.add_trace(go.Scatter(x=t, y=r["eta_carnot"] * 100, name="eta_Carnot (%)"), 2, 1)
    fig.add_trace(go.Scatter(x=t, y=r["power_fraction"] * 100, name="power fraction (%)"), 2, 2)
    fig.update_layout(title="EC223 RTG F2a -- Mission Lifetime", height=720)

    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print(f"\nReport written to {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
