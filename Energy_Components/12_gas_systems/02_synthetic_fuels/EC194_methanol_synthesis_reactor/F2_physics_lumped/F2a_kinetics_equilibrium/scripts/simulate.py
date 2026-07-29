"""
EC194 -- Methanol Synthesis Reactor -- F2a Kinetics + Equilibrium
Optional Plotly simulation report. Plotly import is wrapped so its absence
does not crash anything else.

Generates:
  1. Dynamic startup: T(t), per-pass conversion X_C(t), MeOH dry fraction.
  2. Steady-state conversion vs temperature (kinetic vs equilibrium).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _HAVE_PLOTLY = True
except Exception:
    _HAVE_PLOTLY = False


def run_report(out_html=None):
    cm = ComponentModel()
    m = cm._model

    # --- 1. Dynamic startup ---
    r = cm.predict({"duration_s": 800.0, "dt": 2.0})

    # --- 2. conversion vs T sweep ---
    T_range, X_kin, X_eq = m.conversion_vs_temperature(
        T_range=np.linspace(483.15, 563.15, 20))

    print(f"Startup: T_final={r['T'][-1]:.1f} K, X_C={r['X_C'][-1]:.3f}, "
          f"X_eq={r['X_eq_final']:.3f}, runaway={r['thermal_runaway']}")
    print(f"Sweep: X_kin in [{X_kin.min():.3f}, {X_kin.max():.3f}], "
          f"X_eq in [{X_eq.min():.3f}, {X_eq.max():.3f}]")

    if not _HAVE_PLOTLY:
        print("[simulate] Plotly not available -- skipping HTML report.")
        return

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Reactor temperature (startup)",
            "Per-pass carbon conversion (startup)",
            "MeOH dry mole fraction (startup)",
            "Steady-state conversion vs T",
        ),
    )
    fig.add_trace(go.Scatter(x=r["t"], y=r["T"], name="T [K]"), row=1, col=1)
    fig.add_trace(go.Scatter(x=r["t"], y=r["X_C"], name="X_C"), row=1, col=2)
    fig.add_trace(go.Scatter(x=r["t"], y=r["y_MeOH_dry"], name="y_MeOH dry"), row=2, col=1)
    fig.add_trace(go.Scatter(x=T_range, y=X_kin, name="X kinetic"), row=2, col=2)
    fig.add_trace(go.Scatter(x=T_range, y=X_eq, name="X equilibrium",
                             line=dict(dash="dash")), row=2, col=2)
    fig.update_layout(title="EC194 Methanol Synthesis Reactor -- F2a Kinetics+Equilibrium",
                      height=750, showlegend=True)

    if out_html is None:
        out_html = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out_html)
    print(f"[simulate] Report written to {out_html}")


if __name__ == "__main__":
    run_report()
