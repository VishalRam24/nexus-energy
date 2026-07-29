"""
EC163 -- Single-Phase DC-AC Inverter -- F2a Averaged SPWM + LC Filter
Optional Plotly report. Plotly import is guarded so absence does not crash.
Run: python3 scripts/simulate.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def build_report(out_html=None):
    cm = ComponentModel()
    r = cm.predict({"m_a": 0.85, "duration_s": 0.14})

    # Efficiency / THD sweep over modulation index
    m_a_sweep = np.linspace(0.1, 1.0, 19)
    eta_sweep, thd_sweep, vrms_sweep = [], [], []
    for ma in m_a_sweep:
        op = cm.predict({"m_a": float(ma), "duration_s": 0.12})
        eta_sweep.append(op["efficiency"] * 100.0)
        thd_sweep.append(op["thd_postfilter"] * 100.0)
        vrms_sweep.append(op["v_out_rms"])

    print(f"Operating point m_a=0.85: V_out_rms={r['v_out_rms']:.1f} V, "
          f"P_out={r['p_out_w']:.0f} W, eta={r['efficiency']*100:.2f}%, "
          f"THD_post={r['thd_postfilter']*100:.3f}%")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:  # pragma: no cover
        print(f"[simulate] Plotly unavailable ({e}); skipping HTML report.")
        return None

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Averaged bridge vs filtered output voltage",
            "Filter inductor current i_L",
            "Efficiency vs modulation index",
            "Output THD vs modulation index",
        ),
    )
    t_ms = r["t"] * 1e3
    fig.add_trace(go.Scatter(x=t_ms, y=r["v_inv"], name="v_inv (pre-filter)",
                             line=dict(color="lightgray")), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_ms, y=r["v_out"], name="v_out (filtered)",
                             line=dict(color="crimson")), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_ms, y=r["i_L"], name="i_L",
                             line=dict(color="royalblue")), row=1, col=2)
    fig.add_trace(go.Scatter(x=m_a_sweep, y=eta_sweep, name="eta %",
                             line=dict(color="green")), row=2, col=1)
    fig.add_trace(go.Scatter(x=m_a_sweep, y=thd_sweep, name="THD %",
                             line=dict(color="darkorange")), row=2, col=2)
    fig.update_layout(title="EC163 F2a — Averaged SPWM Single-Phase Inverter",
                      height=760, showlegend=True)

    if out_html is None:
        out_html = os.path.join(os.path.dirname(__file__), "..",
                                "simulation_report.html")
    fig.write_html(out_html)
    print(f"[simulate] Report written to {out_html}")
    return out_html


if __name__ == "__main__":
    build_report()
