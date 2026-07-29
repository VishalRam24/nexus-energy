"""EC118 — SMR — F1a Steady-State Power Map — Simulation & HTML Report"""
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
            "Coolant Outlet Temperature vs PLR",
            "Power Map: P_e vs PLR & Coolant Flow",
        ],
        vertical_spacing=0.14, horizontal_spacing=0.12)

    PLR = np.linspace(0.2, 1.0, 200)
    r = model.predict({"part_load_ratio": PLR})

    fig.add_trace(go.Scatter(x=PLR*100, y=r["thermal_power_mw"],
        name="P_thermal", line=dict(color="red", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=PLR*100, y=r["electric_power_mw"],
        name="P_electric", line=dict(color="blue", width=2)), row=1, col=1)
    fig.add_hline(y=180, row=1, col=1, line_dash="dash", line_color="gray",
                  annotation_text="180 MW_e rated")

    fig.add_trace(go.Scatter(x=PLR*100, y=r["efficiency"]*100,
        name="η_net", line=dict(color="green", width=2)), row=1, col=2)
    fig.add_hline(y=34.0, row=1, col=2, line_dash="dash", line_color="gray",
                  annotation_text="34% Rankine cycle")

    for m_dot in [800, 1300, 1900]:
        rt = model.predict({"part_load_ratio": PLR, "coolant_flow_kgs": float(m_dot)})
        fig.add_trace(go.Scatter(x=PLR*100, y=rt["coolant_outlet_temp_c"],
            name=f"m_dot={m_dot} kg/s"), row=2, col=1)
    fig.add_hline(y=321, row=2, col=1, line_dash="dash", line_color="red",
                  annotation_text="T_out rated = 321°C")

    PLR_grid = np.linspace(0.2, 1.0, 50)
    m_grid = np.linspace(800, 1900, 50)
    Pe_map = np.zeros((50, 50))
    for i, m in enumerate(m_grid):
        rmap = model.predict({"part_load_ratio": PLR_grid, "coolant_flow_kgs": float(m)})
        Pe_map[i, :] = rmap["electric_power_mw"]
    fig.add_trace(go.Heatmap(x=PLR_grid*100, y=m_grid, z=Pe_map,
        colorscale="Plasma", colorbar=dict(title="MW_e", x=1.02)), row=2, col=2)

    fig.update_xaxes(title_text="PLR (%)", row=1, col=1)
    fig.update_xaxes(title_text="PLR (%)", row=1, col=2)
    fig.update_xaxes(title_text="PLR (%)", row=2, col=1)
    fig.update_xaxes(title_text="PLR (%)", row=2, col=2)
    fig.update_yaxes(title_text="Power (MW)", row=1, col=1)
    fig.update_yaxes(title_text="Efficiency (%)", row=1, col=2)
    fig.update_yaxes(title_text="T_out (degC)", row=2, col=1)
    fig.update_yaxes(title_text="Coolant Flow (kg/s)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}",
        height=820, template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
