"""EC168 — MPPT Controller — F1a Tracking Efficiency — Simulation & HTML Report"""
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
            "Tracking Efficiency vs Irradiance",
            "Output Power vs Irradiance",
            "Power Loss vs Irradiance",
            "Efficiency vs Irradiance (Log Scale)",
        ],
        vertical_spacing=0.14,
    )

    G = np.linspace(0, 1200, 500)

    # Different P_rated scenarios to show effect
    for p_scale, label in [(5000, "P_mpp=5kW@1kW/m2"), (10000, "P_mpp=10kW@1kW/m2"), (12000, "P_mpp=12kW@1kW/m2")]:
        p_in = G / 1000.0 * p_scale
        r = model.predict({"irradiance": G, "p_mpp_input": p_in})

        # Plot 1: eta vs G
        fig.add_trace(
            go.Scatter(x=G, y=r["tracking_efficiency"] * 100, name=f"eta {label}"),
            row=1, col=1,
        )
        # Plot 2: P_out vs G
        fig.add_trace(
            go.Scatter(x=G, y=r["p_output"] / 1000.0, name=f"P_out {label}", showlegend=False),
            row=1, col=2,
        )
        # Plot 3: P_loss vs G
        fig.add_trace(
            go.Scatter(x=G, y=r["power_loss"], name=f"Loss {label}", showlegend=False),
            row=2, col=1,
        )

    # Plot 4: Log-scale efficiency for low-irradiance detail
    G_log = np.logspace(0, 3, 300)  # 1 to 1000 W/m2
    p_in_log = G_log / 1000.0 * 10000.0
    r_log = model.predict({"irradiance": G_log, "p_mpp_input": p_in_log})
    fig.add_trace(
        go.Scatter(x=G_log, y=r_log["tracking_efficiency"] * 100, name="eta (log scale)", showlegend=False),
        row=2, col=2,
    )

    # Add reference lines for eta_max
    fig.add_hline(y=99.0, line_dash="dash", line_color="red", annotation_text="eta_max=99%", row=1, col=1)
    fig.add_hline(y=98.0, line_dash="dot", line_color="orange", annotation_text="98%", row=1, col=1)

    fig.update_xaxes(title_text="Irradiance (W/m2)", row=1, col=1)
    fig.update_xaxes(title_text="Irradiance (W/m2)", row=1, col=2)
    fig.update_xaxes(title_text="Irradiance (W/m2)", row=2, col=1)
    fig.update_xaxes(title_text="Irradiance (W/m2, log)", type="log", row=2, col=2)
    fig.update_yaxes(title_text="Tracking Efficiency (%)", row=1, col=1)
    fig.update_yaxes(title_text="Output Power (kW)", row=1, col=2)
    fig.update_yaxes(title_text="Power Loss (W)", row=2, col=1)
    fig.update_yaxes(title_text="Tracking Efficiency (%)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} Tracking Efficiency",
        height=800,
        template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
