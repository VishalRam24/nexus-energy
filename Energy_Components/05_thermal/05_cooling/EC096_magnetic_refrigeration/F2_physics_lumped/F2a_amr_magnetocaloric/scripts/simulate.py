"""
EC096 -- Magnetic Refrigeration -- F2a AMR
Simulation scenarios + optional interactive Plotly report.

Generates: (1) Gd Delta T_ad(T) sweep showing the Curie-temperature peak,
(2) magnetic entropy curves at 0 and B_max, (3) COP & cooling power vs span.
Plotly is optional; absence does not crash.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from model import MU0
from predict import ComponentModel


def run_scenarios():
    cm = ComponentModel()
    mat = cm._model.mat
    H = cm._model.H_max

    # 1) Delta T_ad(T) sweep
    Ts = np.arange(255, 330, 2.5)
    dTad = np.array([mat.delta_T_ad(T, H, 0.0) for T in Ts])

    # 2) magnetic entropy at 0 and B_max
    S0 = np.array([mat.magnetic_entropy(T, 0.0) for T in Ts])
    SH = np.array([mat.magnetic_entropy(T, H) for T in Ts])

    # 3) COP & Q_cold vs reservoir span (centred on T_C)
    spans = np.arange(2.0, 16.0, 2.0)
    COPs, Qcs, Carnots = [], [], []
    for s in spans:
        r = cm.predict({"T_cold_K": 293.0 - s / 2, "T_hot_K": 293.0 + s / 2,
                        "n_cycles": 20})
        COPs.append(r["COP"]); Qcs.append(r["Q_cold_W"]); Carnots.append(r["COP_Carnot"])

    return dict(Ts=Ts, dTad=dTad, S0=S0, SH=SH, spans=spans,
                COPs=np.array(COPs), Qcs=np.array(Qcs), Carnots=np.array(Carnots))


def make_report(data, out_html=None):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        print(f"[simulate] Plotly unavailable ({e}); skipping HTML report.")
        return None

    fig = make_subplots(rows=1, cols=3, subplot_titles=(
        "Gd Delta T_ad vs T (peak at Curie T_C)",
        "Magnetic entropy: B=0 vs B_max",
        "COP and cooling power vs span"))

    fig.add_trace(go.Scatter(x=data["Ts"], y=data["dTad"], name="ΔT_ad (1.5 T)"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=data["Ts"], y=data["S0"], name="S_mag(B=0)"), row=1, col=2)
    fig.add_trace(go.Scatter(x=data["Ts"], y=data["SH"], name="S_mag(B_max)"), row=1, col=2)
    fig.add_trace(go.Scatter(x=data["spans"], y=data["COPs"], name="COP"), row=1, col=3)
    fig.add_trace(go.Scatter(x=data["spans"], y=data["Carnots"], name="COP_Carnot",
                             line=dict(dash="dash")), row=1, col=3)

    fig.update_xaxes(title_text="T [K]", row=1, col=1)
    fig.update_yaxes(title_text="ΔT_ad [K]", row=1, col=1)
    fig.update_xaxes(title_text="T [K]", row=1, col=2)
    fig.update_yaxes(title_text="S_mag [J/(kg·K)]", row=1, col=2)
    fig.update_xaxes(title_text="Span [K]", row=1, col=3)
    fig.update_yaxes(title_text="COP [-]", row=1, col=3)
    fig.update_layout(title_text="EC096 Magnetic Refrigeration F2a — AMR / Gd magnetocaloric")

    if out_html is None:
        out_html = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out_html)
    print(f"[simulate] Report written to {out_html}")
    return out_html


if __name__ == "__main__":
    data = run_scenarios()
    print(f"Peak Delta T_ad = {data['dTad'].max():.2f} K at "
          f"T = {data['Ts'][np.argmax(data['dTad'])]:.0f} K")
    make_report(data)
