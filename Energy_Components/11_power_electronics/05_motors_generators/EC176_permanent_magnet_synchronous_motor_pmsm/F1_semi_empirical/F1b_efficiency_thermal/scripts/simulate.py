"""EC176 — PMSM — F1b Thermal — Simulation & HTML Report"""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
from model import PMSMF1b
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Efficiency vs Torque at Different Magnet Temperatures",
            "Back-EMF vs Speed at Different Magnet Temperatures",
            "Loss Breakdown at 3000 rpm (T_mag=80C vs 150C)",
            "PM Flux & Derating vs Magnet Temperature",
        ],
        vertical_spacing=0.14,
    )

    T_arr = np.linspace(1, 25, 100)
    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]

    # Panel 1: Efficiency vs Torque at different magnet temps
    for i, T_mag in enumerate([25, 60, 80, 120, 150]):
        r = model.predict({"torque": T_arr, "speed_rpm": 3000.0,
                           "magnet_temperature": T_mag})
        fig.add_trace(go.Scatter(
            x=T_arr, y=r["efficiency"] * 100,
            name=f"T_mag={T_mag}C", line=dict(color=colors[i], width=2),
        ), row=1, col=1)

    # Panel 2: Back-EMF vs speed
    omega_arr = np.linspace(100, 12000, 200)
    for i, T_mag in enumerate([25, 80, 120, 150]):
        r = model.predict({"torque": 10.0, "speed_rpm": omega_arr,
                           "magnet_temperature": T_mag})
        fig.add_trace(go.Scatter(
            x=omega_arr, y=r["back_emf_V"],
            name=f"EMF T_mag={T_mag}C", line=dict(color=colors[i], width=2),
        ), row=1, col=2)

    # Panel 3: Loss breakdown comparison
    with open(Path(__file__).parent.parent / "data" / "parameters.json") as f:
        params = json.load(f)
    m = PMSMF1b(params)
    T_range = np.linspace(1, 25, 100)

    for T_mag, ls in [(80, "solid"), (150, "dash")]:
        losses = m.losses(T_range, 3000.0, T_mag)
        label_suffix = f" ({T_mag}C)"
        fig.add_trace(go.Scatter(
            x=T_range, y=losses["p_copper_w"],
            name=f"Copper{label_suffix}", line=dict(dash=ls, width=2),
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=T_range, y=np.full_like(T_range, float(losses["p_iron_w"])),
            name=f"Iron{label_suffix}", line=dict(dash=ls, width=1.5),
        ), row=2, col=1)

    # Panel 4: Flux and derating vs magnet temperature
    T_mag_range = np.linspace(20, 180, 200)
    flux = m.flux(T_mag_range)
    derate = m.derating_factor(T_mag_range, 25.0)

    fig.add_trace(go.Scatter(
        x=T_mag_range, y=flux * 1000,
        name="PM Flux (mWb)", line=dict(color="#636EFA", width=2.5),
    ), row=2, col=2)
    fig.add_trace(go.Scatter(
        x=T_mag_range, y=derate * 100,
        name="Derating (%)", line=dict(color="#EF553B", width=2.5, dash="dash"),
    ), row=2, col=2)
    fig.add_vline(x=150, line_dash="dot", line_color="red",
                  annotation_text="Demag threshold", row=2, col=2)

    fig.update_xaxes(title_text="Torque (Nm)", row=1, col=1)
    fig.update_xaxes(title_text="Speed (rpm)", row=1, col=2)
    fig.update_xaxes(title_text="Torque (Nm)", row=2, col=1)
    fig.update_xaxes(title_text="Magnet Temperature (C)", row=2, col=2)
    fig.update_yaxes(title_text="Efficiency (%)", row=1, col=1)
    fig.update_yaxes(title_text="Back-EMF (V)", row=1, col=2)
    fig.update_yaxes(title_text="Loss (W)", row=2, col=1)
    fig.update_yaxes(title_text="Flux (mWb) / Derating (%)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} Thermal Model",
        height=850,
        template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
