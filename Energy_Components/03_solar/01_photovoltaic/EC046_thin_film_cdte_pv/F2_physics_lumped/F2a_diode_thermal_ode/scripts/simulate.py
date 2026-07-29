"""
EC046 -- Thin-Film CdTe PV -- F2a Physics-Lumped
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
    m = cm._model

    # 1) I-V / P-V curves at several irradiances (25 C cell)
    iv_sets = {}
    for G in [200.0, 500.0, 800.0, 1000.0]:
        iv = m.iv_curve(G, 25.0, n_points=200)
        iv_sets[G] = iv

    # 2) MPP vs temperature (fixed 1000 W/m2)
    temps = np.linspace(-10, 75, 30)
    pmp_T = np.array([m.mpp(1000.0, float(t))["p_mp"] for t in temps])

    # 3) Efficiency vs irradiance (low-light behaviour, 25 C)
    Gs = np.linspace(50, 1100, 40)
    eff_G = np.array([m.efficiency(float(g), 25.0) for g in Gs])

    # 4) Dynamic thermal step: cloud passing (G drop then recovery)
    def G_step(t):
        return 950.0 if (t < 600 or t > 1200) else 250.0
    dyn = m.simulate(G_step, 28.0, T_cell0_c=28.0, wind=1.5,
                     duration_s=2400.0, dt=20.0)

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:  # plotly absent -> print summary only
        print(f"[simulate] plotly unavailable ({e}); printing summary instead.")
        print(f"  STC Pmp           : {m.mpp(1000.0,25.0)['p_mp']:.1f} W")
        print(f"  Pmp @ 65 C        : {m.mpp(1000.0,65.0)['p_mp']:.1f} W")
        print(f"  eff @1000/25      : {m.efficiency(1000.0,25.0)*100:.2f}%")
        print(f"  eff @200/25       : {m.efficiency(200.0,25.0)*100:.2f}%")
        print(f"  dyn T_cell final  : {dyn['temperature'][-1]:.1f} C")
        return None

    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "I-V curves (25 C)", "P-V curves (25 C)",
        "Pmp vs cell temperature (1000 W/m2)",
        "Dynamic cell temperature (cloud transient)"))

    for G, iv in iv_sets.items():
        fig.add_trace(go.Scatter(x=iv["V"], y=iv["I"], name=f"{int(G)} W/m2"),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=iv["V"], y=iv["P"], name=f"{int(G)} W/m2",
                                 showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=temps, y=pmp_T, name="Pmp(T)"), row=2, col=1)
    fig.add_trace(go.Scatter(x=dyn["t"], y=dyn["temperature"], name="T_cell"),
                  row=2, col=2)

    fig.update_xaxes(title_text="Voltage [V]", row=1, col=1)
    fig.update_yaxes(title_text="Current [A]", row=1, col=1)
    fig.update_xaxes(title_text="Voltage [V]", row=1, col=2)
    fig.update_yaxes(title_text="Power [W]", row=1, col=2)
    fig.update_xaxes(title_text="Cell temp [C]", row=2, col=1)
    fig.update_yaxes(title_text="Pmp [W]", row=2, col=1)
    fig.update_xaxes(title_text="Time [s]", row=2, col=2)
    fig.update_yaxes(title_text="T_cell [C]", row=2, col=2)
    fig.update_layout(title="EC046 CdTe PV -- F2a Physics-Lumped Report", height=800)

    if out_html is None:
        out_html = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out_html)
    print(f"[simulate] wrote {out_html}")
    return out_html


if __name__ == "__main__":
    build_report()
