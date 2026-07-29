"""
EC004 -- PAFC F2a -- simulation report.
Generates an interactive Plotly HTML report. Plotly is optional: if it is not
installed, the script still computes and prints a numeric summary.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel


def run_scenarios():
    cm = ComponentModel()
    m = cm._model

    # 1) Polarization curve at nominal 180 C
    T_nom = 453.15
    j_curve = np.linspace(0.01, 0.69, 60)
    V_curve = np.array([m.cell_voltage(j, T_nom, 1.0, 0.21) for j in j_curve])
    P_curve = j_curve * V_curve

    # 2) Cold-start thermal transient
    trans = m.simulate(0.3, 423.15, 1.0, 0.21, 2.0, 1200.0)

    # 3) CO sensitivity at nominal point
    co_levels = [0.0, 0.005, 0.01, 0.02]
    V_co = [m.cell_voltage(0.3, T_nom, 1.0, 0.21, x_CO=x) for x in co_levels]

    return cm, j_curve, V_curve, P_curve, trans, co_levels, V_co


def main():
    cm, j_curve, V_curve, P_curve, trans, co_levels, V_co = run_scenarios()

    print("PAFC F2a simulation summary")
    print(f"  Peak power density : {P_curve.max():.4f} W/cm2 "
          f"at j={j_curve[np.argmax(P_curve)]:.3f} A/cm2")
    print(f"  OCV-ish V(j=0.01)  : {V_curve[0]:.4f} V")
    print(f"  Cold-start T_final : {trans['temperature'][-1]:.2f} K "
          f"(from 423.15 K)")
    print("  CO sensitivity (V at j=0.3):")
    for x, V in zip(co_levels, V_co):
        print(f"    x_CO={x:.3f} -> V={V:.4f} V")

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:  # pragma: no cover
        print(f"  [plotly not available: {e}] skipping HTML report.")
        return

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Polarization V-j", "Power density",
                        "Cold-start thermal transient", "CO sensitivity"),
    )
    fig.add_trace(go.Scatter(x=j_curve, y=V_curve, name="V_cell"), 1, 1)
    fig.add_trace(go.Scatter(x=j_curve, y=P_curve, name="P"), 1, 2)
    fig.add_trace(go.Scatter(x=trans["t"], y=trans["temperature"], name="T"),
                  2, 1)
    fig.add_trace(go.Bar(x=[str(x) for x in co_levels], y=V_co, name="V vs CO"),
                  2, 2)
    fig.update_layout(title="EC004 PAFC F2a — Physics-Lumped Report",
                      height=750, showlegend=False)

    out = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")
    fig.write_html(out)
    print(f"  Report written: {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
