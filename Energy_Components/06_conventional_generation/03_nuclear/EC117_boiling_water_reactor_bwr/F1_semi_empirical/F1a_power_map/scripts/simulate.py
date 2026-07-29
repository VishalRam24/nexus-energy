"""EC117 — BWR — F1a Steady-State Power Map — Simulation & HTML Report"""
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
            "Thermal & Electric Power vs PLR",
            "Net Efficiency vs PLR",
            "Steam Mass Flow vs PLR",
            "Power Conversion: Pe vs Pth",
        ],
        vertical_spacing=0.14)

    PLR = np.linspace(0.6, 1.0, 200)
    r = model.predict({"part_load_ratio": PLR})

    fig.add_trace(go.Scatter(x=PLR*100, y=r["thermal_power_mw"],
        name="P_thermal (MW_th)", line=dict(color="red", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=PLR*100, y=r["electric_power_mw"],
        name="P_electric (MW_e)", line=dict(color="blue", width=2)), row=1, col=1)
    fig.add_hline(y=1100, row=1, col=1, line_dash="dash", line_color="gray",
                  annotation_text="1100 MW_e rated")
    fig.add_hline(y=3300, row=1, col=1, line_dash="dash", line_color="salmon",
                  annotation_text="3300 MW_th rated")

    fig.add_trace(go.Scatter(x=PLR*100, y=r["efficiency"]*100,
        name="η_net", line=dict(color="green", width=2)), row=1, col=2)
    fig.add_hline(y=33.5, row=1, col=2, line_dash="dash", line_color="gray",
                  annotation_text="33.5% direct-cycle")

    fig.add_trace(go.Scatter(x=PLR*100, y=r["steam_mass_flow_kgs"],
        name="Steam flow (kg/s)", line=dict(color="purple", width=2)), row=2, col=1)
    fig.add_hline(y=1900, row=2, col=1, line_dash="dash", line_color="gray",
                  annotation_text="1900 kg/s nominal")

    fig.add_trace(go.Scatter(x=r["thermal_power_mw"], y=r["electric_power_mw"],
        name="Pe vs Pth", line=dict(color="darkorange", width=2)), row=2, col=2)

    fig.update_xaxes(title_text="PLR (%)", row=1, col=1)
    fig.update_xaxes(title_text="PLR (%)", row=1, col=2)
    fig.update_xaxes(title_text="PLR (%)", row=2, col=1)
    fig.update_xaxes(title_text="P_thermal (MW_th)", row=2, col=2)
    fig.update_yaxes(title_text="Power (MW)", row=1, col=1)
    fig.update_yaxes(title_text="Efficiency (%)", row=1, col=2)
    fig.update_yaxes(title_text="Steam (kg/s)", row=2, col=1)
    fig.update_yaxes(title_text="P_electric (MW_e)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}",
        height=820, template="plotly_white")

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
