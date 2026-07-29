"""
EC167 -- Boost-PFC F2a -- simulation report (optional Plotly).
Generates an interactive HTML report of waveforms + KPIs. Plotly import is
guarded so absence does not crash. Run: python3 scripts/simulate.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run():
    cm = ComponentModel()
    res = cm.predict({"P_load": 3000.0, "duration_s": 0.16, "n_points": 6000})
    wf = res["waveforms"]
    print(f"PF={res['power_factor']:.4f}  THD={res['thd_current']*100:.2f}%  "
          f"eta={res['efficiency']*100:.2f}%  V_dc={res['v_dc_mean']:.1f} V  "
          f"V_peak={res['V_peak']:.1f} V  P_loss={res['p_loss_w']:.1f} W")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:  # plotly absent -> skip plot, not a failure
        print(f"[simulate] Plotly unavailable ({e}); skipping HTML report.")
        return

    t = wf["t"]
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=("Line voltage & current (PFC: in-phase)",
                        "Boost inductor current i_L",
                        "Regulated DC-link voltage v_dc vs line peak"))
    fig.add_trace(go.Scatter(x=t, y=wf["v_line"], name="v_line [V]"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=wf["i_line"] * 10.0,
                             name="i_line x10 [A]"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=wf["i_L"], name="i_L [A]"), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=wf["v_dc"], name="v_dc [V]"), row=3, col=1)
    fig.add_hline(y=res["V_peak"], line_dash="dash", row=3, col=1,
                  annotation_text="V_peak")
    fig.update_layout(
        height=850,
        title_text=(f"EC167 Boost-PFC F2a | PF={res['power_factor']:.3f} "
                    f"THD={res['thd_current']*100:.1f}% eta={res['efficiency']*100:.1f}%"))
    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print(f"[simulate] wrote {os.path.abspath(out)}")


if __name__ == "__main__":
    run()
