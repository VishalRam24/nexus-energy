"""EC075 — Finned-Tube Heat Exchanger — F1b — Simulation Scenarios + HTML Report"""

import json
import numpy as np
from pathlib import Path

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

import sys
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


def generate_report(output_path=None):
    model = ComponentModel()

    BASE = {"T_h_in": 70.0, "T_c_in": 20.0, "m_dot_hot": 2.0, "m_dot_cold": 5.0}

    # Scenario 1: Fouling resistance sweep
    Rf_vals = np.linspace(0, 0.005, 50)
    Q_fouled, eps_fouled, U_fouled_arr, eps_red = [], [], [], []
    for Rf in Rf_vals:
        r = model.predict({**BASE, "fouling_resistance_tube": Rf, "fouling_resistance_air": Rf})
        Q_fouled.append(float(r["Q_kw"]))
        eps_fouled.append(float(r["effectiveness"]))
        U_fouled_arr.append(float(r["U_fouled"]))
        eps_red.append(float(r["effectiveness_reduction"]) * 100)

    # Scenario 2: Part-load (flow rate sweep)
    m_dots = np.linspace(0.2, 8.0, 50)
    Q_partload, U_eff_arr = [], []
    for m in m_dots:
        r = model.predict({**BASE, "m_dot_hot": m})
        Q_partload.append(float(r["Q_kw"]))
        U_eff_arr.append(float(r["U_effective_clean"]))

    # Scenario 3: Hot-inlet temperature sweep
    T_h_range = np.linspace(40, 100, 50)
    Q_temp = []
    for T in T_h_range:
        r = model.predict({**BASE, "T_h_in": T})
        Q_temp.append(float(r["Q_kw"]))

    # Scenario 4: Combined fouling + part-load
    m_grid = np.array([0.5, 1.0, 2.0, 4.0])
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    if not HAS_PLOTLY:
        print("Plotly not installed.")
        return

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Heat Duty vs Fouling Resistance",
            "Part-Load: U_effective & Q vs Flow Rate",
            "Heat Duty vs Hot Inlet Temperature",
            "Effectiveness Reduction vs Fouling",
        ],
    )

    fig.add_trace(go.Scatter(x=Rf_vals * 1e4, y=Q_fouled, mode="lines",
                              name="Q (kW)", line=dict(color="#1f77b4")), row=1, col=1)

    fig.add_trace(go.Scatter(x=m_dots, y=Q_partload, mode="lines",
                              name="Q (kW)", line=dict(color="#1f77b4")), row=1, col=2)
    fig.add_trace(go.Scatter(x=m_dots, y=U_eff_arr, mode="lines",
                              name="U_eff (W/m2K)", yaxis="y3",
                              line=dict(color="#ff7f0e", dash="dash")), row=1, col=2)

    fig.add_trace(go.Scatter(x=T_h_range, y=Q_temp, mode="lines",
                              name="Q (kW)", line=dict(color="#1f77b4")), row=2, col=1)

    fig.add_trace(go.Scatter(x=Rf_vals * 1e4, y=eps_red, mode="lines",
                              name="eps reduction (%)", line=dict(color="#d62728")), row=2, col=2)

    fig.update_xaxes(title_text="Fouling Resistance (×10⁻⁴ m²K/W)", row=1, col=1)
    fig.update_xaxes(title_text="Hot Flow Rate (kg/s)", row=1, col=2)
    fig.update_xaxes(title_text="Hot Inlet Temp (°C)", row=2, col=1)
    fig.update_xaxes(title_text="Fouling Resistance (×10⁻⁴ m²K/W)", row=2, col=2)
    fig.update_yaxes(title_text="Q (kW)", row=1, col=1)
    fig.update_yaxes(title_text="Q (kW)", row=1, col=2)
    fig.update_yaxes(title_text="Q (kW)", row=2, col=1)
    fig.update_yaxes(title_text="Effectiveness Reduction (%)", row=2, col=2)

    fig.update_layout(
        title_text="EC075 Finned-Tube HX — F1b: Fouling + Property Corrections",
        height=700,
    )

    if output_path is None:
        output_path = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(output_path))
    print(f"Report saved: {output_path}")


if __name__ == "__main__":
    generate_report()
