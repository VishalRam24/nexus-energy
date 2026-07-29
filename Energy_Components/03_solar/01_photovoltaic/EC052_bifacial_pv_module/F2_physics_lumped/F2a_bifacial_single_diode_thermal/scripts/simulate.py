"""
EC052 -- Bifacial PV Module -- F2a Physics-Lumped
Optional Plotly simulation report. Plotly import is guarded so absence of the
library does not crash the rest of the toolchain.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def build_report(out_html=None):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:  # plotly not installed -- skip gracefully
        print(f"[simulate] Plotly unavailable ({e}); skipping HTML report.")
        return None

    cm = ComponentModel()
    m = cm._model

    # 1. P-V and I-V at STC
    V, I, P = m.iv_curve(1000.0, 25.0, n=300)

    # 2. Bifacial gain vs albedo
    albedos = np.linspace(0.0, 0.9, 30)
    gains = [m.bifacial_gain(800.0, 25.0, albedo=a) * 100.0 for a in albedos]

    # 3. Thermal transient under a step of sun
    def G_step(t):
        return 0.0 if t < 600 else 900.0
    r = m.simulate(G_step, T_amb_C=25.0, v_wind=1.0, albedo=0.4,
                   T_cell0_C=25.0, dt=60.0, duration_s=7200.0)

    # 4. Efficiency vs irradiance
    Gs = np.linspace(50.0, 1100.0, 30)
    effs = [m.efficiency(G, 25.0, albedo=0.3) * 100.0 for G in Gs]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("I-V and P-V @ STC", "Bifacial gain vs albedo",
                        "Cell temperature transient (step sun)",
                        "Efficiency vs front irradiance"))

    fig.add_trace(go.Scatter(x=V, y=I, name="I-V", line=dict(color="blue")),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=V, y=P / 10.0, name="P-V /10",
                             line=dict(color="red")), row=1, col=1)
    fig.add_trace(go.Scatter(x=albedos, y=gains, name="bifacial gain %",
                             line=dict(color="green")), row=1, col=2)
    fig.add_trace(go.Scatter(x=r["t"] / 60.0, y=r["temperature_C"],
                             name="T_cell [C]", line=dict(color="orange")),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=Gs, y=effs, name="efficiency %",
                             line=dict(color="purple")), row=2, col=2)

    fig.update_xaxes(title_text="Voltage [V]", row=1, col=1)
    fig.update_xaxes(title_text="Albedo [-]", row=1, col=2)
    fig.update_xaxes(title_text="Time [min]", row=2, col=1)
    fig.update_xaxes(title_text="G_front [W/m2]", row=2, col=2)
    fig.update_layout(title="EC052 Bifacial PV -- F2a Physics-Lumped",
                      height=800, showlegend=True)

    if out_html is None:
        out_html = os.path.join(os.path.dirname(__file__), "..",
                                "simulation_report.html")
    fig.write_html(out_html)
    print(f"[simulate] Report written to {out_html}")
    return out_html


if __name__ == "__main__":
    build_report()
