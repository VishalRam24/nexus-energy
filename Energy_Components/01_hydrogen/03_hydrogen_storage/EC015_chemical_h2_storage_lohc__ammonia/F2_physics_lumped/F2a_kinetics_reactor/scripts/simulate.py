"""
EC015 -- Chemical H2 Storage (LOHC / Ammonia) -- F2a Lumped Kinetics Reactor
Optional Plotly simulation report. Plotly import is guarded so absence
does not crash. Produces simulation_report.html in the model folder.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel

_HTML = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")


def main():
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:  # pragma: no cover
        print(f"[simulate] plotly unavailable ({e}); skipping HTML report.")
        return

    cm = ComponentModel()
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Conversion X(t)", "H2 release rate [kg/s]",
                        "Reactor temperature [K]", "Heat duties [W]"),
    )

    for mode, color in [("lohc", "#1f77b4"), ("ammonia", "#d62728")]:
        r = cm.predict({"mode": mode, "dt": 20.0, "duration_s": 6000.0,
                        "T0": cm._model._mode(mode)["T_set"]})
        t = r["t"] / 60.0  # minutes
        fig.add_trace(go.Scatter(x=t, y=r["conversion"], name=f"{mode} X",
                                 line=dict(color=color)), row=1, col=1)
        fig.add_trace(go.Scatter(x=t, y=r["h2_rate_kg_s"], name=f"{mode} rate",
                                 line=dict(color=color)), row=1, col=2)
        fig.add_trace(go.Scatter(x=t, y=r["temperature"], name=f"{mode} T",
                                 line=dict(color=color)), row=2, col=1)
        fig.add_trace(go.Scatter(x=t, y=r["q_rxn_W"], name=f"{mode} Q_rxn",
                                 line=dict(color=color, dash="dot")), row=2, col=2)
        fig.add_trace(go.Scatter(x=t, y=r["q_heat_W"], name=f"{mode} Q_heat",
                                 line=dict(color=color)), row=2, col=2)

    fig.update_xaxes(title_text="time [min]")
    fig.update_layout(title="EC015 F2a -- LOHC vs Ammonia H2-release reactor",
                      height=800, template="plotly_white")
    fig.write_html(_HTML)
    print(f"[simulate] wrote {_HTML}")


if __name__ == "__main__":
    main()
