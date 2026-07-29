"""EC009 -- AEL -- F2a Electrochemical -- Simulation & HTML Report"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "V-I Polarization Curve (T=353K, 30% KOH)",
            "Bubble Coverage vs Current Density",
            "Efficiency vs Current Density",
            "Temperature Sensitivity",
        ],
        vertical_spacing=0.14,
    )

    j_range = np.linspace(100, 5000, 100)

    # Panel 1: Polarization curve
    V_cell = [float(model._model.cell_voltage(j, 353.0, 30.0)) for j in j_range]
    fig.add_trace(go.Scatter(x=j_range, y=V_cell, name="V_cell", line=dict(width=2)), row=1, col=1)

    # Panel 2: Bubble coverage
    theta = [float(model._model.bubble_coverage(j)) for j in j_range]
    fig.add_trace(go.Scatter(x=j_range, y=theta, name="theta", line=dict(width=2, color="orange")), row=1, col=2)

    # Panel 3: Efficiency
    eff = [float(model._model.efficiency(j, 353.0, 30.0)) for j in j_range]
    fig.add_trace(go.Scatter(x=j_range, y=eff, name="eta", line=dict(width=2, color="green")), row=2, col=1)

    # Panel 4: Temperature sensitivity
    for T in [323, 333, 343, 353, 363]:
        V = [float(model._model.cell_voltage(j, T, 30.0)) for j in j_range]
        fig.add_trace(go.Scatter(x=j_range, y=V, name=f"T={T}K"), row=2, col=2)

    for r in [1, 2]:
        for c in [1, 2]:
            fig.update_xaxes(title_text="Current Density (A/m2)", row=r, col=c)
    fig.update_yaxes(title_text="Cell Voltage (V)", row=1, col=1)
    fig.update_yaxes(title_text="Bubble Coverage", row=1, col=2)
    fig.update_yaxes(title_text="Efficiency", row=2, col=1)
    fig.update_yaxes(title_text="Cell Voltage (V)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} -- {info['name']} -- {info['fidelity']} Electrochemical Model",
        height=800, template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
