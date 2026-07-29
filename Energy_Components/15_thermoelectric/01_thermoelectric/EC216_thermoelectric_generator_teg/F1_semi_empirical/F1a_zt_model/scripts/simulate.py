"""EC216 — Thermoelectric Generator (TEG) — F1a — Simulation & HTML Report"""
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
            "Efficiency vs T_hot (various T_cold)",
            "Power Output vs T_hot",
            "Efficiency vs Carnot Efficiency",
            "Power Map (T_hot vs T_cold)",
        ],
        vertical_spacing=0.14)

    T_h = np.linspace(50.0, 300.0, 100)

    # Plot 1: Efficiency vs T_hot for various T_cold
    for T_c in [0.0, 10.0, 25.0, 40.0]:
        r = model.predict({"T_hot": T_h, "T_cold": T_c})
        fig.add_trace(go.Scatter(x=T_h, y=r["efficiency"]*100.0,
            name=f"T_cold={T_c}C"), row=1, col=1)

    # Plot 2: Power vs T_hot
    for T_c in [10.0, 25.0, 40.0]:
        r = model.predict({"T_hot": T_h, "T_cold": T_c})
        fig.add_trace(go.Scatter(x=T_h, y=r["power_w"],
            name=f"T_cold={T_c}C P"), row=1, col=2)

    # Plot 3: TEG efficiency vs Carnot efficiency
    T_c_ref = 25.0
    r3 = model.predict({"T_hot": T_h, "T_cold": T_c_ref})
    eta_carnot = (1.0 - (T_c_ref + 273.15) / (T_h + 273.15)) * 100.0
    fig.add_trace(go.Scatter(x=T_h, y=r3["efficiency"]*100.0,
        name="TEG eta", line=dict(color="firebrick")), row=2, col=1)
    fig.add_trace(go.Scatter(x=T_h, y=eta_carnot,
        name="Carnot eta", line=dict(color="steelblue", dash="dash")), row=2, col=1)

    # Plot 4: Power heatmap
    T_h_grid = np.linspace(50.0, 300.0, 50)
    T_c_grid = np.linspace(0.0, 50.0, 50)
    P_map = np.zeros((50, 50))
    for i, T_c in enumerate(T_c_grid):
        r4 = model.predict({"T_hot": T_h_grid, "T_cold": T_c})
        P_map[i, :] = r4["power_w"]
    fig.add_trace(go.Heatmap(
        x=T_h_grid, y=T_c_grid, z=P_map,
        colorscale="Plasma", colorbar=dict(title="Power (W)"),
        name="Power W"), row=2, col=2)

    fig.update_xaxes(title_text="T_hot (degC)", row=1, col=1)
    fig.update_xaxes(title_text="T_hot (degC)", row=1, col=2)
    fig.update_xaxes(title_text="T_hot (degC)", row=2, col=1)
    fig.update_xaxes(title_text="T_hot (degC)", row=2, col=2)
    fig.update_yaxes(title_text="Efficiency (%)", row=1, col=1)
    fig.update_yaxes(title_text="Power (W)", row=1, col=2)
    fig.update_yaxes(title_text="Efficiency (%)", row=2, col=1)
    fig.update_yaxes(title_text="T_cold (degC)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}",
        height=850, template="plotly_white")

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
