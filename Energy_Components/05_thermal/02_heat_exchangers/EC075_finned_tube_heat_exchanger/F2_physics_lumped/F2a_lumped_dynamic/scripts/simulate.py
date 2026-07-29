"""
EC075 -- Finned-Tube HX -- F2a Physics-Lumped -- Simulation / Plotly report.

Generates an interactive HTML report of the transient approach to steady state
and the e-NTU cross-check. Plotly import is guarded so its absence never crashes.
Run: python3 scripts/simulate.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run_report(out_html=None):
    cm = ComponentModel()

    # Transient from cold start.
    r = cm.predict({"T_h_in": 80.0, "T_c_in": 20.0, "m_dot_hot": 1.0,
                    "m_dot_cold": 2.0, "duration_s": 600.0, "n_out": 200, "T0": 20.0})
    ss = r["steady_state"]
    ref = r["entu_reference"]
    print(f"Transient SS: Q={ss['Q_kw']:.2f} kW eps={ss['effectiveness']:.4f}  "
          f"| e-NTU: Q={ref['Q_kw']:.2f} kW eps={ref['effectiveness']:.4f}")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        print(f"[simulate] Plotly unavailable ({e}); skipping HTML report.")
        return None

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Outlet temperatures vs time",
                                        "Heat duty approach to e-NTU"))
    fig.add_trace(go.Scatter(x=r["t"], y=r["T_h_out"], name="Hot (water) outlet"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=r["t"], y=r["T_c_out"], name="Cold (air) outlet"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=r["t"], y=r["T_wall_mean"], name="Wall mean",
                             line=dict(dash="dot")), row=1, col=1)

    fig.add_trace(go.Scatter(x=r["t"], y=r["Q_kw"], name="Transient duty [kW]"),
                  row=1, col=2)
    fig.add_hline(y=ref["Q_kw"], line_dash="dash", line_color="red",
                  annotation_text="e-NTU steady duty", row=1, col=2)

    fig.update_xaxes(title_text="time [s]", row=1, col=1)
    fig.update_xaxes(title_text="time [s]", row=1, col=2)
    fig.update_yaxes(title_text="T [degC]", row=1, col=1)
    fig.update_yaxes(title_text="Q [kW]", row=1, col=2)
    fig.update_layout(title_text="EC075 Finned-Tube HX -- F2a Lumped Transient",
                      template="plotly_white")

    if out_html is None:
        out_html = os.path.join(os.path.dirname(__file__), "..",
                                "simulation_report.html")
    fig.write_html(out_html)
    print(f"[simulate] Report written to {os.path.abspath(out_html)}")
    return out_html


if __name__ == "__main__":
    run_report()
