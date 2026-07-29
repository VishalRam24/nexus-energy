"""EC076 — Regenerative Heat Exchanger — F1b — Simulation Scenarios + HTML Report"""

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
    BASE = {"T_h_in": 250.0, "T_c_in": 20.0, "m_dot_hot": 3.0, "m_dot_cold": 3.0}

    # Scenario 1: Fouling sweep
    Rf_vals = np.linspace(0, 0.005, 50)
    Q_arr, eps_arr, U_arr = [], [], []
    for Rf in Rf_vals:
        r = model.predict({**BASE, "fouling_resistance_hot": Rf, "fouling_resistance_cold": Rf})
        Q_arr.append(float(r["Q_kw"]))
        eps_arr.append(float(r["effectiveness"]))
        U_arr.append(float(r["U_fouled"]))

    # Scenario 2: Carryover sweep
    co_vals = np.linspace(0, 0.10, 50)
    Q_co, eps_co = [], []
    for co in co_vals:
        r = model.predict({**BASE, "carryover_leakage": co})
        Q_co.append(float(r["Q_kw"]))
        eps_co.append(float(r["effectiveness"]))

    # Scenario 3: Cr* sweep
    crs_vals = np.linspace(1.1, 20.0, 50)
    eps_crs = []
    for crs in crs_vals:
        r = model.predict({**BASE, "Cr_star": crs})
        eps_crs.append(float(r["effectiveness"]))

    # Scenario 4: Temperature range
    T_h_range = np.linspace(80, 500, 50)
    Q_temp = []
    for T in T_h_range:
        r = model.predict({**BASE, "T_h_in": T})
        Q_temp.append(float(r["Q_kw"]))

    if not HAS_PLOTLY:
        print("Plotly not installed.")
        return

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Q & Effectiveness vs Fouling",
            "Effect of Carryover/Leakage",
            "Effectiveness vs Matrix Cr*",
            "Heat Duty vs Hot Inlet Temperature",
        ],
    )

    fig.add_trace(go.Scatter(x=Rf_vals * 1e4, y=Q_arr, mode="lines",
                              name="Q (kW)", line=dict(color="#1f77b4")), row=1, col=1)
    fig.add_trace(go.Scatter(x=Rf_vals * 1e4, y=[e * 100 for e in eps_arr],
                              mode="lines", name="eps (%)",
                              line=dict(color="#ff7f0e", dash="dash")), row=1, col=1)

    fig.add_trace(go.Scatter(x=co_vals * 100, y=Q_co, mode="lines",
                              name="Q (kW)", line=dict(color="#2ca02c")), row=1, col=2)
    fig.add_trace(go.Scatter(x=co_vals * 100, y=[e * 100 for e in eps_co],
                              mode="lines", name="eps (%)",
                              line=dict(color="#d62728", dash="dash")), row=1, col=2)

    fig.add_trace(go.Scatter(x=crs_vals, y=[e * 100 for e in eps_crs],
                              mode="lines", name="eps (%)", line=dict(color="#1f77b4")), row=2, col=1)

    fig.add_trace(go.Scatter(x=T_h_range, y=Q_temp, mode="lines",
                              name="Q (kW)", line=dict(color="#1f77b4")), row=2, col=2)

    fig.update_xaxes(title_text="Fouling Resistance (×10⁻⁴ m²K/W)", row=1, col=1)
    fig.update_xaxes(title_text="Carryover/Leakage (%)", row=1, col=2)
    fig.update_xaxes(title_text="Matrix Capacity Ratio Cr*", row=2, col=1)
    fig.update_xaxes(title_text="Hot Inlet Temp (°C)", row=2, col=2)
    fig.update_yaxes(title_text="Q (kW) / eps (%)", row=1, col=1)
    fig.update_yaxes(title_text="Q (kW) / eps (%)", row=1, col=2)
    fig.update_yaxes(title_text="Effectiveness (%)", row=2, col=1)
    fig.update_yaxes(title_text="Q (kW)", row=2, col=2)

    fig.update_layout(
        title_text="EC076 Regenerative HX — F1b: Fouling + Carryover + Cr* Correction",
        height=700,
    )

    if output_path is None:
        output_path = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(output_path))
    print(f"Report saved: {output_path}")


if __name__ == "__main__":
    generate_report()
