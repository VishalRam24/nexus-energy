"""
EC146 -- Torrefaction Reactor -- F2a
Optional Plotly report: mass/energy yield maps + transient profiles.
Plotly import is guarded so absence does not crash.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run_report(out_html="simulation_report.html"):
    cm = ComponentModel()

    # Transient at a nominal deep-torrefaction point
    r = cm.predict({"T_set_degC": 290, "residence_time_min": 45,
                    "T0_degC": 25, "dt_s": 5})

    # Yield maps over (T, t)
    Ts = np.arange(230, 311, 10)
    ts = np.arange(10, 121, 10)
    Ym = np.zeros((len(ts), len(Ts)))
    Ye = np.zeros_like(Ym)
    for i, t in enumerate(ts):
        for j, T in enumerate(Ts):
            rr = cm.predict({"T_set_degC": float(T), "residence_time_min": float(t),
                             "T0_degC": float(T)})
            Ym[i, j] = rr["mass_yield_final"]
            Ye[i, j] = rr["energy_yield_final"]

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        print(f"[simulate] Plotly unavailable ({e}); skipping HTML report.")
        print(f"  Sample: mass_yield={r['mass_yield_final']:.3f} "
              f"energy_yield={r['energy_yield_final']:.3f}")
        return

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Mass yield map", "Energy yield map",
                        "Transient yields (290 C, 45 min)", "Reactor temperature"),
        specs=[[{"type": "heatmap"}, {"type": "heatmap"}],
               [{"type": "xy"}, {"type": "xy"}]],
    )
    fig.add_trace(go.Heatmap(z=Ym, x=Ts, y=ts, colorscale="Viridis",
                             colorbar=dict(title="Ym", x=0.46)), row=1, col=1)
    fig.add_trace(go.Heatmap(z=Ye, x=Ts, y=ts, colorscale="Plasma",
                             colorbar=dict(title="Ye", x=1.0)), row=1, col=2)
    fig.add_trace(go.Scatter(x=r["t"], y=r["mass_yield"], name="mass yield"),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=r["t"], y=r["energy_yield"], name="energy yield"),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=r["t"], y=r["temperature_degC"], name="T [C]"),
                  row=2, col=2)
    fig.update_xaxes(title_text="T_set [C]", row=1, col=1)
    fig.update_yaxes(title_text="time [min]", row=1, col=1)
    fig.update_xaxes(title_text="T_set [C]", row=1, col=2)
    fig.update_xaxes(title_text="time [min]", row=2, col=1)
    fig.update_xaxes(title_text="time [min]", row=2, col=2)
    fig.update_layout(title_text="EC146 Torrefaction F2a — Kinetics + Reactor ODE",
                      height=800, width=1100)

    out_path = os.path.join(os.path.dirname(__file__), "..", out_html)
    fig.write_html(out_path)
    print(f"[simulate] Report written to {out_path}")


if __name__ == "__main__":
    run_report()
