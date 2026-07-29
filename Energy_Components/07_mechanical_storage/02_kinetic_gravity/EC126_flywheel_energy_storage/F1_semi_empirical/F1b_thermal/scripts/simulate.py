"""EC126 — Flywheel Energy Storage — F1b Thermal — Simulation & HTML Report"""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
from model import FlywheelF1b
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()

    with open(Path(__file__).parent.parent / "data" / "parameters.json") as f:
        params = json.load(f)
    m = FlywheelF1b(params)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Standby Losses vs SOC (Windage + Bearing)",
            "Self-Discharge Rate vs SOC at Different Temperatures",
            "Efficiency vs Power Command at SOC=0.5",
            "Speed and Energy vs SOC",
        ],
        vertical_spacing=0.14,
    )

    soc_arr = np.linspace(0.01, 1.0, 200)
    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]

    # Panel 1: Loss breakdown vs SOC
    P_windage = m.windage_loss(soc_arr, 25.0)
    P_bearing = m.bearing_loss(soc_arr)
    P_total = P_windage + P_bearing

    fig.add_trace(go.Scatter(
        x=soc_arr, y=P_windage * 1000,
        name="Windage (W)", fill="tozeroy", fillcolor="rgba(99,110,250,0.15)",
        line=dict(color="#636EFA", width=2),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=soc_arr, y=P_total * 1000,
        name="Total (W)", fill="tonexty", fillcolor="rgba(239,85,59,0.15)",
        line=dict(color="#EF553B", width=2),
    ), row=1, col=1)

    # Panel 2: Self-discharge rate vs SOC at different temperatures
    for i, T_amb in enumerate([-10, 0, 25, 40, 55]):
        sd = m.self_discharge_rate(soc_arr, T_amb)
        fig.add_trace(go.Scatter(
            x=soc_arr, y=sd * 100,
            name=f"T_amb={T_amb}C", line=dict(color=colors[i], width=2),
        ), row=1, col=2)

    # Panel 3: Efficiency vs power command
    P_range = np.linspace(-100, 100, 200)
    # Remove near-zero region to avoid division issues
    P_range = P_range[np.abs(P_range) > 2.0]
    for i, soc in enumerate([0.1, 0.25, 0.5, 0.75, 1.0]):
        eta = m.efficiency(soc, P_range, 25.0)
        fig.add_trace(go.Scatter(
            x=P_range, y=np.clip(eta * 100, 0, 100),
            name=f"SOC={soc:.2f}", line=dict(color=colors[i], width=2),
        ), row=2, col=1)

    # Panel 4: Speed and energy vs SOC
    speed = m.speed_rpm(soc_arr)
    energy = m.energy_stored(soc_arr)

    fig.add_trace(go.Scatter(
        x=soc_arr, y=speed,
        name="Speed (rpm)", line=dict(color="#636EFA", width=2.5),
    ), row=2, col=2)
    fig.add_trace(go.Scatter(
        x=soc_arr, y=energy,
        name="Energy (kWh)", line=dict(color="#00CC96", width=2.5, dash="dash"),
    ), row=2, col=2)

    fig.update_xaxes(title_text="SOC", row=1, col=1)
    fig.update_xaxes(title_text="SOC", row=1, col=2)
    fig.update_xaxes(title_text="Power Command (kW)", row=2, col=1)
    fig.update_xaxes(title_text="SOC", row=2, col=2)
    fig.update_yaxes(title_text="Loss (W)", row=1, col=1)
    fig.update_yaxes(title_text="Self-Discharge Rate (%/h)", row=1, col=2)
    fig.update_yaxes(title_text="Efficiency (%)", row=2, col=1)
    fig.update_yaxes(title_text="Speed (rpm) / Energy (kWh)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}",
        height=850,
        template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
