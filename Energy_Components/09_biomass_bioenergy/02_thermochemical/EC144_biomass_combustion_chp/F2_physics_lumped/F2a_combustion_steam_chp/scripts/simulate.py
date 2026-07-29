"""
EC144 -- Biomass Combustion CHP -- F2a
Optional Plotly simulation report. Plotly import is guarded so absence
does not crash. Run: python3 scripts/simulate.py
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run_report():
    cm = ComponentModel()

    # 1. Efficiency vs moisture (full load)
    M = np.linspace(0.0, 0.55, 30)
    eta_el, eta_th, eta_tot = [], [], []
    for m in M:
        r = cm._model.predict_steady(1.0, float(m))
        eta_el.append(r["eta_electrical"])
        eta_th.append(r["eta_thermal"])
        eta_tot.append(r["eta_total_chp"])

    # 2. Boiler thermal transient (cold start ramp)
    tr = cm.predict({"PLR": 1.0, "moisture_fraction": 0.2,
                     "T0_K": 288.15, "duration_s": 7200.0, "dt": 30.0})

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        print(f"[simulate] Plotly unavailable ({e}); printing summary instead.")
        print(f"  eta_total dry={eta_tot[0]:.3f}  wet={eta_tot[-1]:.3f}")
        print(f"  Boiler T: {tr['T_boiler_K'][0]:.1f} -> {tr['T_boiler_K'][-1]:.1f} K")
        return

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("CHP efficiency vs moisture",
                                        "Lumped boiler thermal transient"))
    fig.add_trace(go.Scatter(x=M, y=eta_el, name="electrical"), row=1, col=1)
    fig.add_trace(go.Scatter(x=M, y=eta_th, name="thermal"), row=1, col=1)
    fig.add_trace(go.Scatter(x=M, y=eta_tot, name="total CHP"), row=1, col=1)
    fig.add_trace(go.Scatter(x=tr["t"] / 60.0, y=tr["T_boiler_K"],
                             name="T_boiler"), row=1, col=2)
    fig.update_xaxes(title_text="moisture (wet basis)", row=1, col=1)
    fig.update_xaxes(title_text="time (min)", row=1, col=2)
    fig.update_yaxes(title_text="efficiency", row=1, col=1)
    fig.update_yaxes(title_text="T_boiler (K)", row=1, col=2)
    fig.update_layout(title="EC144 Biomass Combustion CHP — F2a")

    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print(f"[simulate] wrote {out}")


if __name__ == "__main__":
    run_report()
