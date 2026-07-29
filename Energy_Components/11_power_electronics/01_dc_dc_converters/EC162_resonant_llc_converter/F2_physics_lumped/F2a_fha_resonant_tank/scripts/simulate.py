"""
EC162 -- Resonant LLC Converter -- F2a Physics-Lumped (FHA)
Optional Plotly report: gain curves M(fn,Q), ZVS boundary, efficiency vs load,
and the output-filter start-up transient. Plotly import is guarded.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def build_report(out_html=None):
    cm = ComponentModel()
    m = cm._model

    fn = np.linspace(0.4, 2.0, 400)
    loads = [0.05, 0.072, 0.2, 1.0]  # rated ~0.072 ohm (12V/2kW)
    gain_curves = {f"R={r}Ω (Q={m.quality_factor(r):.2f})":
                   [m.gain_from_load(f, r) for f in fn] for r in loads}

    r_rated = m.V_out_nom ** 2 / m.P_rated
    eff = [m.efficiency(f, r_rated) for f in fn]
    zvs = [1 if m.is_zvs(f, r_rated) else 0 for f in fn]

    sim = m.simulate(1.0, r_rated, dt=2e-6, duration_s=3e-3)

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:  # plotly absent -> print summary only
        print("Plotly unavailable, text summary:")
        print(f"  f_r={m.f_r/1e3:.1f} kHz  k={m.k:.2f}  Z0={m.Z_0:.2f} ohm")
        print(f"  unity gain @ fn=1: M={m.gain_from_load(1.0, r_rated):.4f}")
        print(f"  eta@rated={m.efficiency(1.0, r_rated)*100:.2f}%  ZVS={m.is_zvs(1.05, r_rated)}")
        print(f"  V_out_ss={sim['v_out_ss']:.3f} V  V_out(final)={sim['v_out'][-1]:.3f} V")
        return None

    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "DC Gain M(fn, Q, k)", "Efficiency vs fn (rated load)",
        "ZVS region (1=ZVS)", "Output-filter start-up transient"))
    for label, g in gain_curves.items():
        fig.add_trace(go.Scatter(x=fn, y=g, name=label), row=1, col=1)
    fig.add_hline(y=1.0, line_dash="dot", row=1, col=1)
    fig.add_trace(go.Scatter(x=fn, y=eff, name="efficiency"), row=1, col=2)
    fig.add_trace(go.Scatter(x=fn, y=zvs, name="ZVS"), row=2, col=1)
    fig.add_trace(go.Scatter(x=sim["t"]*1e3, y=sim["v_out"], name="V_out(t)"), row=2, col=2)
    fig.add_hline(y=sim["v_out_ss"], line_dash="dot", row=2, col=2)
    fig.update_layout(title="EC162 LLC F2a (FHA) — physics-lumped report", height=800)

    if out_html is None:
        out_html = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out_html)
    print(f"Report written: {out_html}")
    return out_html


if __name__ == "__main__":
    build_report()
