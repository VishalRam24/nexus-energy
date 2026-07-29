"""
EC166 -- AC-DC Rectifier (Diode Bridge) -- F2a Averaged Cap-Filter
Plotly simulation report (optional). Plotly import is wrapped so its absence
does not crash the build.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run_report(out_html=None):
    cm = ComponentModel()
    base = cm.predict({"v_ac_rms": 400.0, "R_load": 8.0, "duration_s": 0.12})

    # ripple vs C sweep
    Cs = [5e-4, 1e-3, 4.7e-3, 1e-2, 2e-2]
    ripples = []
    for C in Cs:
        cm_c = ComponentModel(params={"C_out": {"value": C, "unit": "F"}})
        r = cm_c.predict({"v_ac_rms": 400.0, "R_load": 8.0, "duration_s": 0.14})
        ripples.append(r["v_ripple_pp"])

    print("=== EC166 F2a summary ===")
    print(f"V_dc_mean={base['v_dc_mean']:.1f} V (ideal {base['v_dc_ideal']:.1f} V)")
    print(f"ripple_pp={base['v_ripple_pp']:.2f} V, eff={base['efficiency']:.3f}, "
          f"PF={base['power_factor']:.3f}")
    for C, rp in zip(Cs, ripples):
        print(f"  C={C*1e6:.0f} uF -> ripple {rp:.2f} V")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        print(f"[simulate] Plotly unavailable ({e}); skipping HTML.")
        return

    fig = make_subplots(rows=3, cols=1, subplot_titles=(
        "DC-bus (capacitor) voltage & rectified envelope",
        "Pulsed diode charging current vs smooth load current",
        "Output ripple vs filter capacitance",
    ))
    t_ms = base["t"] * 1e3
    fig.add_trace(go.Scatter(x=t_ms, y=base["v_rect"], name="v_rect (envelope)",
                             line=dict(color="lightgray")), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_ms, y=base["v_dc"], name="v_dc (cap)",
                             line=dict(color="firebrick")), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_ms, y=base["i_diode"], name="i_diode (pulsed)",
                             line=dict(color="royalblue")), row=2, col=1)
    fig.add_trace(go.Scatter(x=t_ms, y=base["i_load"], name="i_load (smooth)",
                             line=dict(color="green")), row=2, col=1)
    fig.add_trace(go.Scatter(x=[c * 1e6 for c in Cs], y=ripples, name="ripple_pp",
                             mode="lines+markers", line=dict(color="purple")), row=3, col=1)
    fig.update_xaxes(title_text="time [ms]", row=1, col=1)
    fig.update_xaxes(title_text="time [ms]", row=2, col=1)
    fig.update_xaxes(title_text="C_out [uF]", type="log", row=3, col=1)
    fig.update_yaxes(title_text="V", row=1, col=1)
    fig.update_yaxes(title_text="A", row=2, col=1)
    fig.update_yaxes(title_text="ripple_pp [V]", row=3, col=1)
    fig.update_layout(height=900, title_text="EC166 Diode Bridge Rectifier — F2a Physics-Lumped")

    if out_html is None:
        out_html = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out_html)
    print(f"[simulate] Report written to {out_html}")


if __name__ == "__main__":
    run_report()
