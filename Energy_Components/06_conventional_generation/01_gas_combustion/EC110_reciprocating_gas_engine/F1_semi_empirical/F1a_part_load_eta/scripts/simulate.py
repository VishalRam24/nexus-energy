"""EC110 — Reciprocating Gas Engine — F1a — Simulation & HTML Report"""
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
            "Electrical Efficiency vs PLR (parametric T_amb)",
            "Electrical Power vs PLR",
            "Specific Fuel Consumption vs PLR",
            "Gas Volume Flow vs PLR",
        ],
        vertical_spacing=0.14)

    plr = np.linspace(0.5, 1.0, 100)

    for T in [-10.0, 15.0, 25.0, 40.0]:
        r = model.predict({"part_load_ratio": plr, "ambient_temp_c": T})
        fig.add_trace(go.Scatter(x=plr, y=r["eta_electrical"]*100,
            name=f"η T={T:.0f}°C"), row=1, col=1)

    r = model.predict({"part_load_ratio": plr, "ambient_temp_c": 25.0})
    fig.add_trace(go.Scatter(x=plr, y=r["electrical_power_kw"],
        name="P_el (kW)", line=dict(color="steelblue")), row=1, col=2)

    fig.add_trace(go.Scatter(x=plr, y=r["sfc_gkwh"],
        name="SFC (g/kWh)", line=dict(color="darkorange")), row=2, col=1)

    fig.add_trace(go.Scatter(x=plr, y=r["gas_volume_flow_m3h"],
        name="Gas flow (m³/h)", line=dict(color="green")), row=2, col=2)

    fig.update_xaxes(title_text="Part-Load Ratio (-)", row=1, col=1)
    fig.update_xaxes(title_text="Part-Load Ratio (-)", row=1, col=2)
    fig.update_xaxes(title_text="Part-Load Ratio (-)", row=2, col=1)
    fig.update_xaxes(title_text="Part-Load Ratio (-)", row=2, col=2)
    fig.update_yaxes(title_text="η_el (%)", row=1, col=1)
    fig.update_yaxes(title_text="Power (kW)", row=1, col=2)
    fig.update_yaxes(title_text="SFC (g/kWh)", row=2, col=1)
    fig.update_yaxes(title_text="Gas Flow (m³/h)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}",
        height=820, template="plotly_white")

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
