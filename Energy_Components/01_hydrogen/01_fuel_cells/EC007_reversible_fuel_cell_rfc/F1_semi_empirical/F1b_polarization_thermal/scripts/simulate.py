"""
EC007 -- RFC -- F1b Polarization-Thermal -- Simulation & HTML Report
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("plotly required: pip install plotly")
    sys.exit(1)


def generate_report():
    model = ComponentModel()

    temps = [313.15, 333.15, 353.15, 363.15]
    labels = ["40C", "60C", "80C", "90C"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    j_fc = np.linspace(0.01, 1.9, 200)
    j_el = np.linspace(0.01, 2.8, 200)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "FC Mode: V-j Curves at Different Temperatures",
            "Electrolyser Mode: V-j Curves",
            "Membrane Resistance vs Temperature",
            "Efficiency vs Current Density",
        ],
        vertical_spacing=0.13,
        horizontal_spacing=0.10,
    )

    for T, lbl, clr in zip(temps, labels, colors):
        # FC V-j
        V_fc = [float(model.predict({"current_density": float(j), "temperature": T, "mode": "fc"})["cell_voltage_V"]) for j in j_fc]
        fig.add_trace(go.Scatter(x=j_fc, y=V_fc, name=f"FC {lbl}", line=dict(color=clr)), row=1, col=1)
        # EL V-j
        V_el = [float(model.predict({"current_density": float(j), "temperature": T, "mode": "electrolyser"})["cell_voltage_V"]) for j in j_el]
        fig.add_trace(go.Scatter(x=j_el, y=V_el, name=f"EL {lbl}", line=dict(color=clr, dash="dash")), row=1, col=2)

    # Membrane R vs T
    Ts = np.linspace(313.15, 363.15, 100)
    R_mem = [float(model._model.membrane_resistance(T)) for T in Ts]
    fig.add_trace(go.Scatter(x=Ts - 273.15, y=R_mem, name="R_mem", line=dict(color="#9467bd")), row=2, col=1)

    # Efficiency vs j at 353 K
    eta_fc = [float(model.predict({"current_density": float(j), "temperature": 353.15, "mode": "fc"})["efficiency"]) for j in j_fc]
    eta_el = [float(model.predict({"current_density": float(j), "temperature": 353.15, "mode": "electrolyser"})["efficiency"]) for j in j_el]
    fig.add_trace(go.Scatter(x=j_fc, y=eta_fc, name="FC η (353K)", line=dict(color="#2ca02c")), row=2, col=2)
    fig.add_trace(go.Scatter(x=j_el, y=eta_el, name="EL η (353K)", line=dict(color="#d62728", dash="dash")), row=2, col=2)

    fig.update_layout(title_text="EC007 RFC F1b Polarization-Thermal Simulation Report", height=700)
    fig.update_xaxes(title_text="j (A/cm2)", row=1, col=1)
    fig.update_yaxes(title_text="V_cell (V)", row=1, col=1)
    fig.update_xaxes(title_text="j (A/cm2)", row=1, col=2)
    fig.update_yaxes(title_text="V_cell (V)", row=1, col=2)
    fig.update_xaxes(title_text="T (degC)", row=2, col=1)
    fig.update_yaxes(title_text="R_mem (ohm cm2)", row=2, col=1)
    fig.update_xaxes(title_text="j (A/cm2)", row=2, col=2)
    fig.update_yaxes(title_text="Efficiency", row=2, col=2)

    out_path = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out_path))
    print(f"Report saved to {out_path}")


if __name__ == "__main__":
    generate_report()
