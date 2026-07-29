"""EC104 — Gas Engine CHP — F1a — Simulation & HTML Report"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()

    fig = make_subplots(rows=2, cols=2,
        subplot_titles=[
            "Efficiencies vs Part-Load Ratio",
            "Power Outputs vs Part-Load Ratio",
            "Fuel Input & Total Output vs PLR",
            "Power Split: Electrical vs Thermal",
        ],
        vertical_spacing=0.14)

    plr = np.linspace(0.5, 1.0, 100)
    r = model.predict({"part_load_ratio": plr})

    # Plot 1: Efficiencies vs PLR
    fig.add_trace(go.Scatter(x=plr, y=r["eta_electrical"]*100,
        name="eta_el (%)", line=dict(color="steelblue")), row=1, col=1)
    fig.add_trace(go.Scatter(x=plr, y=r["eta_thermal"]*100,
        name="eta_th (%)", line=dict(color="firebrick")), row=1, col=1)
    fig.add_trace(go.Scatter(x=plr, y=r["eta_total"]*100,
        name="eta_total (%)", line=dict(color="green", dash="dash")), row=1, col=1)

    # Plot 2: Power outputs
    fig.add_trace(go.Scatter(x=plr, y=r["electrical_power_kw"],
        name="P_el (kW)", line=dict(color="steelblue")), row=1, col=2)
    fig.add_trace(go.Scatter(x=plr, y=r["thermal_power_kw"],
        name="Q_th (kW)", line=dict(color="firebrick")), row=1, col=2)

    # Plot 3: Fuel input and useful output
    fig.add_trace(go.Scatter(x=plr, y=r["fuel_input_kw"],
        name="Fuel input (kW)", line=dict(color="gray")), row=2, col=1)
    useful = r["electrical_power_kw"] + r["thermal_power_kw"]
    fig.add_trace(go.Scatter(x=plr, y=useful,
        name="P_el + Q_th (kW)", line=dict(color="purple", dash="dot")), row=2, col=1)

    # Plot 4: Stacked bar / area — power split
    fig.add_trace(go.Scatter(
        x=plr, y=r["electrical_power_kw"],
        fill="tozeroy", name="P_el", fillcolor="rgba(70,130,180,0.4)",
        line=dict(color="steelblue")), row=2, col=2)
    fig.add_trace(go.Scatter(
        x=plr, y=r["electrical_power_kw"] + r["thermal_power_kw"],
        fill="tonexty", name="Q_th (stacked)", fillcolor="rgba(178,34,34,0.3)",
        line=dict(color="firebrick")), row=2, col=2)

    fig.update_xaxes(title_text="Part-Load Ratio (-)", row=1, col=1)
    fig.update_xaxes(title_text="Part-Load Ratio (-)", row=1, col=2)
    fig.update_xaxes(title_text="Part-Load Ratio (-)", row=2, col=1)
    fig.update_xaxes(title_text="Part-Load Ratio (-)", row=2, col=2)
    fig.update_yaxes(title_text="Efficiency (%)", row=1, col=1)
    fig.update_yaxes(title_text="Power (kW)", row=1, col=2)
    fig.update_yaxes(title_text="Power (kW)", row=2, col=1)
    fig.update_yaxes(title_text="Power (kW)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}",
        height=850, template="plotly_white")

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
