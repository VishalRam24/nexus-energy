"""
EC178 -- SRM F2a -- simulation scenarios + optional Plotly HTML report.
Plotly import is wrapped so absence does not crash; this file need not be run.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run_scenarios():
    cm = ComponentModel()
    base = cm.predict({"T_load": 2.0, "duration_s": 0.05, "dt": 2e-5})
    print(f"Base: T_avg={base['T_avg']:.3f} N.m ripple={base['torque_ripple']:.3f} "
          f"eff={base['efficiency']:.3f}")
    loads = [1.0, 2.0, 4.0, 8.0]
    runs = {TL: cm.predict({"T_load": TL, "duration_s": 0.05, "dt": 2e-5}) for TL in loads}
    return base, runs


def build_report(out_html=None):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:  # pragma: no cover
        print(f"[simulate] Plotly unavailable ({e}); skipping HTML report.")
        return None

    base, runs = run_scenarios()
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Instantaneous torque", "Phase currents",
                        "Speed", "Efficiency vs load torque"),
    )
    t_ms = base["t"] * 1e3
    fig.add_trace(go.Scatter(x=t_ms, y=base["torque"], name="T_e"), row=1, col=1)
    for ph in range(base["phase_currents"].shape[0]):
        fig.add_trace(go.Scatter(x=t_ms, y=base["phase_currents"][ph],
                                 name=f"i_{ph+1}"), row=1, col=2)
    fig.add_trace(go.Scatter(x=t_ms, y=base["speed_rpm"], name="speed"), row=2, col=1)
    fig.add_trace(go.Scatter(x=list(runs.keys()),
                             y=[r["efficiency"] for r in runs.values()],
                             mode="lines+markers", name="eff"), row=2, col=2)
    fig.update_layout(title="EC178 SRM F2a — Reluctance / Co-energy Model")

    if out_html is None:
        out_html = os.path.join(os.path.dirname(__file__), "..",
                                "simulation_report.html")
    fig.write_html(out_html)
    print(f"[simulate] Report written to {out_html}")
    return out_html


if __name__ == "__main__":
    build_report()
