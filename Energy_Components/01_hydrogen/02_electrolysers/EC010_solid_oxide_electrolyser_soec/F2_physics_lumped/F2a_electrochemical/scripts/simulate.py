"""EC010 -- SOEC -- F2a Electrochemical -- Simulation & HTML Report"""
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
            "V-I Curve at Different Temperatures",
            "Efficiency vs Current Density",
            "Steam Utilization Effect on Voltage",
            "Thermal Mode Boundary",
        ],
        vertical_spacing=0.14,
    )

    j_range = np.linspace(0.01, 1.5, 100)

    # Panel 1: Temperature sensitivity
    for T in [923, 1023, 1073, 1123, 1173]:
        V = [float(model._model.cell_voltage(j, T, 0.5)) for j in j_range]
        fig.add_trace(go.Scatter(x=j_range, y=V, name=f"T={T}K"), row=1, col=1)
    fig.add_hline(y=model._model.V_tn, line_dash="dash", line_color="red",
                  annotation_text="V_thermoneutral", row=1, col=1)

    # Panel 2: Efficiency
    for T in [923, 1073, 1173]:
        eff = [float(model._model.efficiency(j, T, 0.5)) for j in j_range]
        fig.add_trace(go.Scatter(x=j_range, y=eff, name=f"eta T={T}K"), row=1, col=2)

    # Panel 3: Steam utilization
    for U in [0.2, 0.4, 0.6, 0.8]:
        V = [float(model._model.cell_voltage(j, 1073.15, U)) for j in j_range]
        fig.add_trace(go.Scatter(x=j_range, y=V, name=f"U={U}"), row=2, col=1)

    # Panel 4: Thermal mode
    for T in [923, 1023, 1073, 1123]:
        modes = [float(model._model.thermal_mode(j, T, 0.5)) for j in j_range]
        fig.add_trace(go.Scatter(x=j_range, y=modes, name=f"mode T={T}K"), row=2, col=2)

    for r in [1, 2]:
        for c in [1, 2]:
            fig.update_xaxes(title_text="Current Density (A/cm2)", row=r, col=c)

    fig.update_layout(
        title=f"{info['ec_id']} -- {info['name']} -- {info['fidelity']} Electrochemical Model",
        height=800, template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
