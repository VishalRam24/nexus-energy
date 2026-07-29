"""
EC218 -- Thermionic Converter -- F2a Physics-Lumped Emitter-Thermal
Optional Plotly simulation report. Plotly import is guarded so absence of the
library does not crash the model.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run_report(out_html=None):
    cm = ComponentModel()

    # Scenario 1: emitter warm-up transient under fixed external heat
    sim = cm.predict({"Q_external_w": 80.0, "T_emitter0_K": 1400.0,
                      "T_collector_K": 900.0, "dt": 0.02, "duration_s": 30.0})

    # Scenario 2: steady-state I-V / efficiency sweep over emitter temperature
    m = cm._model
    import numpy as np
    T_grid = np.linspace(1400, 2100, 60)
    Jv, Pv, Ev, Cv = [], [], [], []
    for T in T_grid:
        op = m.operating_point(T, 900.0)
        Jv.append(op["J_net_Acm2"])
        Pv.append(op["power_density_w_cm2"])
        Ev.append(op["efficiency"])
        Cv.append(op["carnot_efficiency"])

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:  # noqa: BLE001
        print(f"[simulate] Plotly unavailable ({e}); skipping HTML report.")
        print(f"  Warm-up final T_emitter = {sim['T_emitter'][-1]:.1f} K, "
              f"eta = {sim['efficiency'][-1]:.3f}")
        return None

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Emitter warm-up T(t)", "Electrical power P(t)",
                        "Power density vs T_emitter", "Efficiency vs Carnot"),
    )
    fig.add_trace(go.Scatter(x=sim["t"], y=sim["T_emitter"], name="T_emitter"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=sim["t"], y=sim["power_w"], name="P [W]"),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=T_grid, y=Pv, name="P density [W/cm^2]"),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=T_grid, y=Ev, name="efficiency"), row=2, col=2)
    fig.add_trace(go.Scatter(x=T_grid, y=Cv, name="Carnot", line=dict(dash="dash")),
                  row=2, col=2)
    fig.update_layout(title="EC218 Thermionic Converter F2a — Physics-Lumped",
                      height=720)

    if out_html is None:
        out_html = os.path.join(os.path.dirname(__file__), "..",
                                "simulation_report.html")
    fig.write_html(out_html)
    print(f"[simulate] Report written to {out_html}")
    return out_html


if __name__ == "__main__":
    run_report()
