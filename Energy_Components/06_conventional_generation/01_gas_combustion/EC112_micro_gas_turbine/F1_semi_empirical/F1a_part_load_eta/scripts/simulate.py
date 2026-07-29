"""EC112 — Micro Gas Turbine — F1a — Simulation & HTML Report"""
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
            "Efficiency vs PLR (parametric T_amb)",
            "Power Output vs PLR",
            "Heat Rate vs PLR",
            "Efficiency Map: PLR vs T_amb",
        ],
        vertical_spacing=0.14)

    plr = np.linspace(0.3, 1.0, 100)

    for T in [-10.0, 0.0, 15.0, 30.0, 45.0]:
        r = model.predict({"part_load_ratio": plr, "ambient_temp_c": T})
        fig.add_trace(go.Scatter(x=plr, y=r["eta_electrical"]*100,
            name=f"η T={T:.0f}°C"), row=1, col=1)

    r = model.predict({"part_load_ratio": plr, "ambient_temp_c": 15.0})
    fig.add_trace(go.Scatter(x=plr, y=r["electrical_power_kw"],
        name="P_el (kW)", line=dict(color="steelblue")), row=1, col=2)

    fig.add_trace(go.Scatter(x=plr, y=r["heat_rate_kjkwh"],
        name="HR (kJ/kWh)", line=dict(color="darkorange")), row=2, col=1)

    PLR_grid = np.linspace(0.3, 1.0, 50)
    T_grid = np.linspace(-20.0, 50.0, 50)
    eta_map = np.zeros((len(T_grid), len(PLR_grid)))
    for i, T in enumerate(T_grid):
        rg = model.predict({"part_load_ratio": PLR_grid, "ambient_temp_c": T})
        eta_map[i, :] = rg["eta_electrical"] * 100.0

    fig.add_trace(go.Heatmap(x=PLR_grid, y=T_grid, z=eta_map,
        colorscale="Viridis", colorbar=dict(title="η (%)")), row=2, col=2)

    fig.update_xaxes(title_text="Part-Load Ratio (-)", row=1, col=1)
    fig.update_xaxes(title_text="Part-Load Ratio (-)", row=1, col=2)
    fig.update_xaxes(title_text="Part-Load Ratio (-)", row=2, col=1)
    fig.update_xaxes(title_text="Part-Load Ratio (-)", row=2, col=2)
    fig.update_yaxes(title_text="η_el (%)", row=1, col=1)
    fig.update_yaxes(title_text="Power (kW)", row=1, col=2)
    fig.update_yaxes(title_text="Heat Rate (kJ/kWh)", row=2, col=1)
    fig.update_yaxes(title_text="T_amb (°C)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}",
        height=820, template="plotly_white")

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
