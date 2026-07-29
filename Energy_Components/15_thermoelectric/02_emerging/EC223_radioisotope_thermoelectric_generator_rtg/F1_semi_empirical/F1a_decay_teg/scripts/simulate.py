"""EC223 — RTG — F1a — Simulation & HTML Report"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()
    t_half = model.params["unit"]["t_half_years"]["value"]
    design_life = model.params["unit"]["design_life_years"]["value"]

    fig = make_subplots(rows=2, cols=2,
        subplot_titles=[
            "Thermal & Electric Power vs Time",
            "Hot-Side Temperature vs Time",
            "TEG Efficiency vs Time",
            "Power Fraction vs Time (normalized)",
        ],
        vertical_spacing=0.14)

    t = np.linspace(0.0, 100.0, 500)
    r = model.predict({"t_years": t})

    # Plot 1: Thermal and electric power
    fig.add_trace(go.Scatter(x=t, y=r["P_thermal_W"],
        name="P_thermal", line=dict(color="firebrick")), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=r["P_electric_W"],
        name="P_electric", line=dict(color="steelblue")), row=1, col=1)
    fig.add_vline(x=design_life, line_dash="dash", line_color="gray",
        annotation_text=f"Design Life {design_life:.0f}y", row=1, col=1)

    # Plot 2: Hot-side temperature
    fig.add_trace(go.Scatter(x=t, y=r["T_hot_K"] - 273.15,
        name="T_hot (°C)", line=dict(color="orange")), row=1, col=2)

    # Plot 3: TEG efficiency
    fig.add_trace(go.Scatter(x=t, y=r["eta_teg"] * 100.0,
        name="eta_TEG (%)", line=dict(color="green")), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=r["eta_carnot"] * 100.0,
        name="eta_Carnot (%)", line=dict(color="gray", dash="dash")), row=2, col=1)

    # Plot 4: Power fraction
    fig.add_trace(go.Scatter(x=t, y=r["power_fraction"] * 100.0,
        name="P_el fraction (%)", line=dict(color="purple")), row=2, col=2)
    fig.add_vline(x=t_half, line_dash="dot", line_color="red",
        annotation_text=f"t_half={t_half}y", row=2, col=2)

    fig.update_xaxes(title_text="Time (years)", row=1, col=1)
    fig.update_xaxes(title_text="Time (years)", row=1, col=2)
    fig.update_xaxes(title_text="Time (years)", row=2, col=1)
    fig.update_xaxes(title_text="Time (years)", row=2, col=2)
    fig.update_yaxes(title_text="Power (W)", row=1, col=1)
    fig.update_yaxes(title_text="Temperature (°C)", row=1, col=2)
    fig.update_yaxes(title_text="Efficiency (%)", row=2, col=1)
    fig.update_yaxes(title_text="Power fraction (%)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}",
        height=850, template="plotly_white")

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
