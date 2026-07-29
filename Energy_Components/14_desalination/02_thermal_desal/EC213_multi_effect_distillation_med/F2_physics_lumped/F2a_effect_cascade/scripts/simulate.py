"""
EC213 -- MED F2a -- simulation scenarios + optional Plotly HTML report.
Plotly import is guarded so its absence does not crash the model.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run_scenarios():
    cm = ComponentModel()

    # 1) GOR vs number of effects
    Ns = list(range(2, 17))
    gors = [cm.predict({"N_effects": N})["GOR"] for N in Ns]

    # 2) Steady cascade at design point
    base = cm.predict({})

    # 3) Start-up transient
    trans = cm.predict({"transient": True, "T0_C": 40.0,
                        "dt": 20.0, "duration_s": 7200.0})["transient"]

    return Ns, gors, base, trans


def build_report(out_html=None):
    Ns, gors, base, trans = run_scenarios()
    if out_html is None:
        out_html = os.path.join(os.path.dirname(__file__), "..",
                                "simulation_report.html")
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:  # plotly optional
        print(f"[simulate] plotly unavailable ({e}); printing summary instead.")
        print(f"GOR(N): {list(zip(Ns, [round(g,2) for g in gors]))}")
        print(f"Design GOR={base['GOR']:.2f}, distillate={base['distillate_total_m3_h']:.1f} m3/h")
        return None

    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "GOR vs number of effects", "Temperature cascade (design)",
        "Distillate per effect", "Start-up transient: effect temperatures"))

    fig.add_trace(go.Scatter(x=Ns, y=gors, mode="lines+markers", name="GOR"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=Ns, y=Ns, mode="lines", name="GOR=N (ideal)",
                             line=dict(dash="dash")), row=1, col=1)

    eff = list(range(1, base["N_effects"] + 1))
    fig.add_trace(go.Scatter(x=eff, y=base["T_effect"], mode="lines+markers",
                             name="T_effect"), row=1, col=2)
    fig.add_trace(go.Bar(x=eff, y=base["D_effect"], name="D_effect"), row=2, col=1)

    t = trans["t"] / 60.0
    for i in range(trans["N_effects"]):
        fig.add_trace(go.Scatter(x=t, y=trans["T_effect"][i], mode="lines",
                                 name=f"eff {i+1}", showlegend=False), row=2, col=2)

    fig.update_layout(height=800, title_text="EC213 MED F2a -- Physics-Lumped Effect Cascade")
    fig.write_html(out_html)
    print(f"[simulate] report written to {out_html}")
    return out_html


if __name__ == "__main__":
    build_report()
