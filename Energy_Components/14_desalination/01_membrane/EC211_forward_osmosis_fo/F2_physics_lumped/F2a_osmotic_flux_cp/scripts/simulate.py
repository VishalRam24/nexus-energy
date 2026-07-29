"""
EC211 -- Forward Osmosis (FO) -- F2a Physics-Lumped
Optional Plotly report. Plotly import is guarded so absence won't crash.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def build_report(duration_s=14400.0, out_html=None):
    cm = ComponentModel()
    r = cm.predict({"duration_s": duration_s, "n_points": 300})
    t_h = r["t"] / 3600.0

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:  # plotly absent -> print summary, don't crash
        print(f"[simulate] Plotly unavailable ({e}); text summary:")
        print(f"  Jw: {r['Jw_LMH'][0]:.2f} -> {r['Jw_LMH'][-1]:.2f} LMH")
        print(f"  c_draw: {r['c_draw_mol_m3'][0]:.0f} -> {r['c_draw_mol_m3'][-1]:.0f} mol/m3")
        print(f"  permeate: {r['permeate_m3'][-1]*1000:.1f} L; "
              f"SEC_regen={r['SEC_regen_kWh_m3']:.2f} kWh/m3")
        return None

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Water flux Jw (declines as draw dilutes)",
                        "Draw concentration & osmotic pressure",
                        "Reverse salt flux Js",
                        "Cumulative permeate produced"),
    )
    fig.add_trace(go.Scatter(x=t_h, y=r["Jw_LMH"], name="Jw [LMH]"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_h, y=r["c_draw_mol_m3"], name="c_draw [mol/m3]"), row=1, col=2)
    fig.add_trace(go.Scatter(x=t_h, y=r["pi_draw_bar"], name="pi_draw [bar]"), row=1, col=2)
    fig.add_trace(go.Scatter(x=t_h, y=r["Js_gMH"], name="Js [g/m2/h]"), row=2, col=1)
    fig.add_trace(go.Scatter(x=t_h, y=r["permeate_m3"], name="permeate [m3]"), row=2, col=2)
    fig.update_xaxes(title_text="time [h]")
    fig.update_layout(title="EC211 Forward Osmosis F2a — Osmotic Flux + Concentration Polarization",
                      height=720, showlegend=True)

    if out_html is None:
        out_html = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out_html)
    print(f"[simulate] report written to {out_html}")
    return out_html


if __name__ == "__main__":
    build_report()
