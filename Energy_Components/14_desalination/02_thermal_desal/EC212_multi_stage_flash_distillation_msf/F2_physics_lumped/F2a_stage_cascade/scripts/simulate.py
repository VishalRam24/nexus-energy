"""
EC212 -- MSF F2a Stage-Cascade -- simulation report.
Generates an interactive Plotly HTML report (optional; plotly import is guarded).
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run():
    cm = ComponentModel()

    # 1) design-point transient
    r = cm.predict({"duration_s": 2000.0, "n_eval": 200})

    # 2) GOR vs top brine temperature sweep
    TBT = np.linspace(90, 120, 16)
    gor = [cm._model.steady_state(T_top=float(t))["GOR"] for t in TBT]

    # 3) distillate per stage at design point
    stage_idx = np.arange(1, len(r["distillate_stage"]) + 1)

    print("=== EC212 MSF F2a simulation summary ===")
    print(f"GOR design        : {r['GOR']:.2f}")
    print(f"Distillate        : {r['D_total']:.1f} kg/s ({r['D_total']*3.6:.0f} m3/h)")
    print(f"Steam             : {r['M_steam']:.1f} kg/s")
    print(f"Recovery          : {r['recovery']*100:.1f} %")
    print(f"GOR @90C / @120C  : {gor[0]:.2f} / {gor[-1]:.2f}")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        print(f"[plotly unavailable: {e}] skipping HTML report.")
        return

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Stage temperature cascade (transient start-up)",
            "Steady temperature cascade",
            "GOR vs top brine temperature",
            "Distillate produced per stage",
        ),
    )

    # transient: a few stages over time
    t = r["t"]
    for s in [0, len(r["T_target"]) // 2, len(r["T_target"]) - 1]:
        fig.add_trace(go.Scatter(x=t, y=r["T_stages"][:, s], name=f"stage {s+1}"),
                      row=1, col=1)
    fig.add_trace(go.Scatter(x=stage_idx, y=r["T_target"], name="cascade",
                             mode="lines+markers"), row=1, col=2)
    fig.add_trace(go.Scatter(x=TBT, y=gor, name="GOR(TBT)",
                             mode="lines+markers"), row=2, col=1)
    fig.add_trace(go.Bar(x=stage_idx, y=r["distillate_stage"], name="D_stage"),
                  row=2, col=2)

    fig.update_xaxes(title_text="time [s]", row=1, col=1)
    fig.update_yaxes(title_text="T [degC]", row=1, col=1)
    fig.update_xaxes(title_text="stage", row=1, col=2)
    fig.update_yaxes(title_text="T [degC]", row=1, col=2)
    fig.update_xaxes(title_text="TBT [degC]", row=2, col=1)
    fig.update_yaxes(title_text="GOR [-]", row=2, col=1)
    fig.update_xaxes(title_text="stage", row=2, col=2)
    fig.update_yaxes(title_text="distillate [kg/s]", row=2, col=2)
    fig.update_layout(title_text="EC212 MSF F2a Stage-Cascade physics-lumped model",
                      height=800, showlegend=True)

    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print(f"Report written: {os.path.abspath(out)}")


if __name__ == "__main__":
    run()
