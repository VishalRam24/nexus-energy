"""
EC032 -- Zinc-Air Battery -- F2a Air-Cathode Electrochemical
Optional Plotly report: polarization curve (showing air-electrode limiting
current) + a constant-current discharge plateau with thermal/SOC traces.
Plotly import is guarded so its absence does not crash the script.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def build_report(out_html=None):
    cm = ComponentModel()
    m = cm._model

    # 1) Steady polarization curve (air vs pure O2) -- shows limiting current
    j = np.linspace(0.0, 0.30, 200)
    V_air = m.polarization_curve(j, 298.15, 0.21)
    V_o2 = m.polarization_curve(j, 298.15, 1.0)

    # 2) Constant-current discharge plateau
    r = cm.predict({"current_density_A_cm2": 0.05, "dt": 10.0, "duration_s": 7200.0})

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        print(f"[simulate] Plotly unavailable ({e}); printing summary instead.")
        print(f"  Plateau V ~ {r['voltage'][len(r['voltage'])//2]:.3f} V")
        print(f"  Limiting current (air) j_L = {m.limiting_current(298.15,0.21):.3f} A/cm2")
        return None

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Polarization V-j (limiting current)",
                        "Power density",
                        "Discharge plateau (V vs SOC)",
                        "Temperature during discharge"),
    )
    fig.add_trace(go.Scatter(x=j, y=V_air, name="air (pO2=0.21)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=j, y=V_o2, name="pure O2"), row=1, col=1)
    fig.add_trace(go.Scatter(x=j, y=j * V_air, name="P air"), row=1, col=2)
    fig.add_trace(go.Scatter(x=r["soc"], y=r["voltage"], name="V vs SOC"), row=2, col=1)
    fig.add_trace(go.Scatter(x=r["t"] / 60.0, y=r["temperature"], name="T [K]"), row=2, col=2)
    fig.update_xaxes(title_text="j [A/cm2]", row=1, col=1)
    fig.update_yaxes(title_text="V [V]", row=1, col=1)
    fig.update_xaxes(title_text="SOC", row=2, col=1)
    fig.update_xaxes(title_text="t [min]", row=2, col=2)
    fig.update_layout(title="EC032 Zn-Air F2a -- Air-Cathode Electrochemical", height=800)

    if out_html is None:
        out_html = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out_html)
    print(f"[simulate] wrote {out_html}")
    return out_html


if __name__ == "__main__":
    build_report()
