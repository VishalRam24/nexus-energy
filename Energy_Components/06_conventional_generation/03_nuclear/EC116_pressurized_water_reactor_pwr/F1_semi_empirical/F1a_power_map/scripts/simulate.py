"""EC116 — PWR — F1a Steady-State Power Map — Simulation & HTML Report"""
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
            "Thermal & Electric Power vs PLR",
            "Net Efficiency vs PLR",
            "Coolant Outlet Temperature vs PLR",
            "Power Map: Electric Output vs PLR & Coolant Flow",
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.12,
    )

    PLR = np.linspace(0.5, 1.0, 200)

    # Plot 1: Thermal and electric power vs PLR
    r = model.predict({"part_load_ratio": PLR})
    fig.add_trace(
        go.Scatter(x=PLR * 100, y=r["thermal_power_mw"], name="Thermal Power (MW_th)",
                   line=dict(color="red", width=2)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=PLR * 100, y=r["electric_power_mw"], name="Electric Power (MW_e)",
                   line=dict(color="blue", width=2)),
        row=1, col=1,
    )
    fig.add_hline(y=1000, row=1, col=1, line_dash="dash", line_color="gray",
                  annotation_text="1000 MW_e rated")
    fig.add_hline(y=3000, row=1, col=1, line_dash="dash", line_color="salmon",
                  annotation_text="3000 MW_th rated")

    # Plot 2: Efficiency vs PLR
    fig.add_trace(
        go.Scatter(x=PLR * 100, y=r["efficiency"] * 100, name="Net Efficiency",
                   line=dict(color="green", width=2)),
        row=1, col=2,
    )
    fig.add_hline(y=33.0, row=1, col=2, line_dash="dash", line_color="gray",
                  annotation_text="33% Rankine cycle limit")

    # Plot 3: Coolant temperature vs PLR for different flow rates
    for m_dot in [12000, 15000, 18000]:
        r_t = model.predict({"part_load_ratio": PLR, "coolant_flow_kgs": float(m_dot)})
        fig.add_trace(
            go.Scatter(x=PLR * 100, y=r_t["coolant_outlet_temp_c"],
                       name=f"m_dot={m_dot} kg/s",
                       line=dict(width=2)),
            row=2, col=1,
        )
    fig.add_hline(y=326, row=2, col=1, line_dash="dash", line_color="red",
                  annotation_text="T_outlet rated = 326°C")
    fig.add_hline(y=292, row=2, col=1, line_dash="dot", line_color="blue",
                  annotation_text="T_inlet = 292°C")

    # Plot 4: Electric power heatmap vs PLR and coolant flow
    PLR_grid = np.linspace(0.5, 1.0, 50)
    m_grid = np.linspace(9000, 18000, 50)
    Pe_map = np.zeros((50, 50))
    for i, m in enumerate(m_grid):
        r_map = model.predict({"part_load_ratio": PLR_grid, "coolant_flow_kgs": float(m)})
        Pe_map[i, :] = r_map["electric_power_mw"]

    fig.add_trace(
        go.Heatmap(
            x=PLR_grid * 100, y=m_grid, z=Pe_map,
            colorscale="Plasma", name="P_electric (MW_e)",
            colorbar=dict(title="MW_e", x=1.02),
        ),
        row=2, col=2,
    )

    # Axes labels
    fig.update_xaxes(title_text="Part-Load Ratio (%)", row=1, col=1)
    fig.update_xaxes(title_text="Part-Load Ratio (%)", row=1, col=2)
    fig.update_xaxes(title_text="Part-Load Ratio (%)", row=2, col=1)
    fig.update_xaxes(title_text="PLR (%)", row=2, col=2)
    fig.update_yaxes(title_text="Power (MW)", row=1, col=1)
    fig.update_yaxes(title_text="Efficiency (%)", row=1, col=2)
    fig.update_yaxes(title_text="T_outlet (degC)", row=2, col=1)
    fig.update_yaxes(title_text="Coolant Flow (kg/s)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} Steady-State Power Map",
        height=750,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
