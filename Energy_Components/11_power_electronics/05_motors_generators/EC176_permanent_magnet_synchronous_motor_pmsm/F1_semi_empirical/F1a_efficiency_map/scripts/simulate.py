"""EC176 — PMSM — F1a Efficiency Map — Simulation & HTML Report"""
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
            "Efficiency Map (Torque vs Speed Heatmap)",
            "Efficiency vs Torque at Constant Speed",
            "Loss Breakdown at Rated Speed (3000 rpm)",
            "Output Power vs Speed at Constant Torque",
        ],
        vertical_spacing=0.14,
    )

    T_arr = np.linspace(1, 200, 80)
    omega_arr = np.linspace(100, 12000, 80)

    # Plot 1: Efficiency heatmap
    TT, OO = np.meshgrid(T_arr, omega_arr)
    r = model.predict({"torque": TT.ravel(), "speed_rpm": OO.ravel()})
    eta_map = r["efficiency"].reshape(80, 80)
    fig.add_trace(
        go.Heatmap(
            x=T_arr, y=omega_arr, z=eta_map * 100,
            colorscale="RdYlGn", colorbar=dict(title="Efficiency (%)"),
            zmin=50, zmax=100,
            name="Efficiency Map",
        ),
        row=1, col=1,
    )

    # Plot 2: Efficiency vs Torque at constant speeds
    for omega in [500, 1500, 3000, 6000, 9000, 12000]:
        r2 = model.predict({"torque": T_arr, "speed_rpm": omega})
        fig.add_trace(
            go.Scatter(x=T_arr, y=r2["efficiency"] * 100, name=f"{omega} rpm"),
            row=1, col=2,
        )

    # Plot 3: Loss breakdown at rated speed 3000 rpm
    from model import PMSMF1a
    import json
    with open(Path(__file__).parent.parent / "data" / "parameters.json") as f:
        params = json.load(f)
    m = PMSMF1a(params)
    T_range = np.linspace(1, 200, 100)
    losses = m.losses(T_range, 3000.0)
    fig.add_trace(
        go.Scatter(x=T_range, y=losses["p_copper_w"], name="Copper Loss", fill="tozeroy"),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=T_range, y=losses["p_iron_w"] * np.ones_like(T_range),
                   name="Iron Loss (const at const speed)", line=dict(dash="dash")),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=T_range, y=losses["p_mech_w"] * np.ones_like(T_range),
                   name="Mech Loss (const at const speed)", line=dict(dash="dot")),
        row=2, col=1,
    )

    # Plot 4: P_out vs speed at constant torques
    omega_range = np.linspace(100, 12000, 200)
    for T in [40, 80, 120, 160, 200]:
        r4 = model.predict({"torque": T, "speed_rpm": omega_range})
        fig.add_trace(
            go.Scatter(x=omega_range, y=r4["output_power_kw"],
                       name=f"T={T}Nm", showlegend=True),
            row=2, col=2,
        )

    fig.update_xaxes(title_text="Torque (Nm)", row=1, col=1)
    fig.update_xaxes(title_text="Torque (Nm)", row=1, col=2)
    fig.update_xaxes(title_text="Torque (Nm)", row=2, col=1)
    fig.update_xaxes(title_text="Speed (rpm)", row=2, col=2)
    fig.update_yaxes(title_text="Speed (rpm)", row=1, col=1)
    fig.update_yaxes(title_text="Efficiency (%)", row=1, col=2)
    fig.update_yaxes(title_text="Loss (W)", row=2, col=1)
    fig.update_yaxes(title_text="Output Power (kW)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} Efficiency Map",
        height=850,
        template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
