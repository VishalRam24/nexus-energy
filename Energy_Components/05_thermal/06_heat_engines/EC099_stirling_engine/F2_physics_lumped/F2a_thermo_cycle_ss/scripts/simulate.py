"""
EC099 -- Stirling Engine -- F2a Physics-Lumped
Simulation + optional Plotly HTML report. Plotly import is guarded so its
absence does not crash the script.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run():
    cm = ComponentModel()

    # 1) Warm-up transient
    warm = cm.predict({"T_h0": 300.0, "dt": 5.0, "duration_s": 1200.0})

    # 2) Steady-state efficiency vs hot-end temperature sweep
    m = cm._model
    Th_sweep = np.linspace(650.0, 1100.0, 40)
    eta = np.array([m.cycle_efficiency(Th, 323.15) for Th in Th_sweep])
    eta_c = np.array([m.carnot_efficiency(Th, 323.15) for Th in Th_sweep])

    # 3) Power vs mean pressure
    p_sweep = np.linspace(1.0e6, 1.2e7, 40)
    P_ind = []
    p0 = m.p_mean
    for p in p_sweep:
        m.p_mean = p
        P_ind.append(m.indicated_power(923.15, 323.15, 1500.0))
    m.p_mean = p0
    P_ind = np.array(P_ind)

    print(f"Warm-up final T_h    : {warm['T_h'][-1]:.1f} K")
    print(f"Warm-up final brake P: {warm['brake_power'][-1]:.1f} W")
    print(f"Design eta @923 K    : {m.cycle_efficiency(923.15, 323.15)*100:.2f} % "
          f"(Carnot {m.carnot_efficiency(923.15, 323.15)*100:.2f} %)")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception:
        print("[simulate] plotly not available -- skipping HTML report.")
        return

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Warm-up: hot-end temperature", "Warm-up: brake power",
                        "Efficiency vs T_h (with Carnot cap)",
                        "Indicated power vs mean pressure"),
    )
    fig.add_trace(go.Scatter(x=warm["t"], y=warm["T_h"], name="T_h"), 1, 1)
    fig.add_trace(go.Scatter(x=warm["t"], y=warm["brake_power"], name="P_brake"), 1, 2)
    fig.add_trace(go.Scatter(x=Th_sweep, y=eta*100, name="cycle eta"), 2, 1)
    fig.add_trace(go.Scatter(x=Th_sweep, y=eta_c*100, name="Carnot", line=dict(dash="dash")), 2, 1)
    fig.add_trace(go.Scatter(x=p_sweep/1e5, y=np.array(P_ind)/1e3, name="P_ind"), 2, 2)
    fig.update_layout(title="EC099 Stirling Engine -- F2a Physics-Lumped", height=800)

    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print(f"[simulate] wrote {out}")


if __name__ == "__main__":
    run()
