"""
EC095 — Thermoelectric Cooler (Peltier) — F2a Physics-Lumped Transient
Optional Plotly report (transient pull-down + COP vs current curves).
Plotly import is guarded so absence does not crash.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def build_report(out_html=None):
    cm = ComponentModel()
    m = cm._model

    # --- Scenario 1: cold-start pull-down ---
    sim = cm.predict({"current_A": 4.0, "T_ambient_C": 25.0, "T_load_C": 5.0,
                      "Q_load_W": 20.0, "dt": 2.0, "duration_s": 900.0})

    # --- Scenario 2: COP & Q_c vs current at fixed plate temps ---
    Tc, Th = 283.15, 308.15
    I = np.linspace(0.1, 6.0, 80)
    Qc = m.cooling_power(I, Tc, Th)
    cop = m.cop(I, Tc, Th)
    I_qc = float(m.optimum_current_qc(Tc))
    I_cop = float(m.optimum_current_cop(Tc, Th))

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        print(f"[simulate] Plotly unavailable ({e}); printing summary only.")
        print(f"  Pull-down: T_cold {sim['T_cold_C'][0]:.1f} -> "
              f"{sim['T_cold_C'][-1]:.1f} C, COP_ss={sim['cop'][-1]:.3f}")
        print(f"  I_optQc={I_qc:.2f} A, I_optCOP={I_cop:.2f} A")
        return None

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Transient plate temperatures", "Transient COP",
                        "Cooling power Q_c vs current", "COP vs current"),
    )
    fig.add_trace(go.Scatter(x=sim["t"], y=sim["T_cold_C"], name="T_cold"), 1, 1)
    fig.add_trace(go.Scatter(x=sim["t"], y=sim["T_hot_C"], name="T_hot"), 1, 1)
    fig.add_trace(go.Scatter(x=sim["t"], y=sim["cop"], name="COP(t)"), 1, 2)
    fig.add_trace(go.Scatter(x=I, y=Qc, name="Q_c"), 2, 1)
    fig.add_vline(x=I_qc, line_dash="dash", row=2, col=1)
    fig.add_trace(go.Scatter(x=I, y=cop, name="COP"), 2, 2)
    fig.add_vline(x=I_cop, line_dash="dash", row=2, col=2)
    fig.update_layout(title="EC095 Peltier TEC — F2a Physics-Lumped Transient",
                      height=720)

    if out_html is None:
        out_html = os.path.join(os.path.dirname(__file__), "..",
                                "simulation_report.html")
    fig.write_html(out_html)
    print(f"[simulate] wrote {out_html}")
    return out_html


if __name__ == "__main__":
    build_report()
