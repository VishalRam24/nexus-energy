"""EC221 — MHD Generator — F1a — Simulation & HTML Report"""
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
            "Power vs Load Factor K (various B fields)",
            "Power vs Plasma Velocity",
            "MHD Efficiency vs K (eta_mhd = K*(1-K))",
            "Power Map (u vs B at K=0.5)",
        ],
        vertical_spacing=0.14)

    K_range = np.linspace(0.0, 1.0, 100)
    u_range = np.linspace(100.0, 2000.0, 100)

    # Plot 1: Power vs K for various B
    for B in [2.0, 5.0, 7.0]:
        r = model.predict({"sigma": 10.0, "u": 800.0, "B": B, "K": K_range})
        fig.add_trace(go.Scatter(x=K_range, y=r["power_w"]/1e6,
            name=f"B={B}T"), row=1, col=1)

    # Plot 2: Power vs velocity
    for B in [3.0, 5.0, 7.0]:
        r = model.predict({"sigma": 10.0, "u": u_range, "B": B, "K": 0.5})
        fig.add_trace(go.Scatter(x=u_range, y=r["power_w"]/1e6,
            name=f"B={B}T P(u)"), row=1, col=2)

    # Plot 3: eta_mhd vs K
    r3 = model.predict({"sigma": 10.0, "u": 800.0, "B": 5.0, "K": K_range})
    fig.add_trace(go.Scatter(x=K_range, y=r3["eta_mhd"]*100.0,
        name="eta_mhd", line=dict(color="firebrick")), row=2, col=1)
    # Theoretical max line
    fig.add_trace(go.Scatter(x=[0.5, 0.5], y=[0, 25],
        name="K=0.5 max", line=dict(color="gray", dash="dash")), row=2, col=1)

    # Plot 4: Power heatmap at K=0.5
    u_grid = np.linspace(200.0, 1500.0, 30)
    B_grid = np.linspace(1.0, 8.0, 30)
    P_map = np.zeros((30, 30))
    for i, B in enumerate(B_grid):
        r4 = model.predict({"sigma": 10.0, "u": u_grid, "B": B, "K": 0.5})
        P_map[i, :] = r4["power_w"] / 1e6
    fig.add_trace(go.Heatmap(
        x=u_grid, y=B_grid, z=P_map,
        colorscale="Plasma", colorbar=dict(title="Power (MW)"),
        name="Power MW"), row=2, col=2)

    fig.update_xaxes(title_text="Load Factor K", row=1, col=1)
    fig.update_xaxes(title_text="Plasma Velocity (m/s)", row=1, col=2)
    fig.update_xaxes(title_text="Load Factor K", row=2, col=1)
    fig.update_xaxes(title_text="Plasma Velocity (m/s)", row=2, col=2)
    fig.update_yaxes(title_text="Power (MW)", row=1, col=1)
    fig.update_yaxes(title_text="Power (MW)", row=1, col=2)
    fig.update_yaxes(title_text="eta_mhd (%)", row=2, col=1)
    fig.update_yaxes(title_text="B (T)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}",
        height=850, template="plotly_white")

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
