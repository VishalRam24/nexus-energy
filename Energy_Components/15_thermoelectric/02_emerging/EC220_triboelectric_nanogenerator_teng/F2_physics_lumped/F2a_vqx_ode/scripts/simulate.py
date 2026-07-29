"""
EC220 -- TENG -- F2a Physics-Lumped V-Q-x Model
Optional Plotly simulation report. Plotly import wrapped in try/except so its
absence does not crash. Generates simulation_report.html with:
  * charge / voltage / current / power waveforms over a few cycles
  * average power vs load resistance (with optimum)
  * average power vs frequency
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def main():
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:  # pragma: no cover
        print(f"Plotly unavailable ({e}); skipping HTML report.")
        return

    cm = ComponentModel()
    r = cm.predict({"frequency_hz": 3.0, "R_load_ohm": 1e7, "n_cycles": 5,
                    "sweep_load": True, "sweep_freq": True})

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "Gap x(t) [mm]", "Transferred charge Q(t) [nC]",
            "Terminal voltage V(t) [V]", "Instantaneous power P(t) [uW]",
            "Avg power vs load R [uW]", "Avg power vs frequency [uW]",
        ),
    )
    t = r["t"]
    fig.add_trace(go.Scatter(x=t, y=r["gap"] * 1e3, name="x"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=r["charge"] * 1e9, name="Q"), row=1, col=2)
    fig.add_trace(go.Scatter(x=t, y=r["voltage"], name="V"), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=r["power"] * 1e6, name="P"), row=2, col=2)

    fig.add_trace(go.Scatter(x=r["sweep_R_load"], y=r["sweep_power_vs_load"] * 1e6,
                             name="P vs R", mode="lines+markers"), row=3, col=1)
    fig.add_vline(x=r["R_optimal_ohm"], line_dash="dash", row=3, col=1)
    fig.update_xaxes(type="log", row=3, col=1)

    fig.add_trace(go.Scatter(x=r["sweep_frequency"], y=r["sweep_power_vs_freq"] * 1e6,
                             name="P vs f", mode="lines+markers"), row=3, col=2)
    fig.update_xaxes(type="log", row=3, col=2)

    fig.update_layout(
        title_text=(f"EC220 TENG F2a -- V-Q-x ODE | optimal load "
                    f"{r['R_optimal_ohm']:.2e} ohm, "
                    f"P_avg {r['power_avg']*1e6:.3f} uW"),
        height=900, showlegend=False,
    )
    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
