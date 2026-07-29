"""EC132 — Tidal Stream Turbine — F1a — Simulation & HTML Report"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()
    m = model._model

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Power Curve (kW vs Current Speed)",
            "Power Coefficient vs Current Speed",
            "Density Correction (P vs Current Speed)",
            "Power Map — Speed vs Water Density",
        ],
        vertical_spacing=0.13,
    )

    speeds = np.linspace(0.0, 4.5, 300)

    # Panel 1: Power curve
    r = model.predict({"current_speed_ms": speeds})
    fig.add_trace(go.Scatter(x=speeds, y=r["power_kw"], name="Power (kW)",
                             line=dict(color="teal")), row=1, col=1)
    fig.add_vline(x=m.v_cut_in, line_dash="dot", line_color="red",
                  annotation_text="v_cut_in", row=1, col=1)
    fig.add_vline(x=m.v_rated, line_dash="dot", line_color="green",
                  annotation_text="v_rated", row=1, col=1)
    fig.add_vline(x=m.v_cut_out, line_dash="dot", line_color="purple",
                  annotation_text="v_cut_out", row=1, col=1)

    # Panel 2: Cp curve
    fig.add_trace(go.Scatter(x=speeds, y=r["power_coefficient"], name="Cp operational",
                             line=dict(color="blue")), row=1, col=2)
    # Betz limit reference line
    fig.add_hline(y=0.593, line_dash="dash", line_color="red",
                  annotation_text="Betz limit 0.593", row=1, col=2)
    fig.add_hline(y=m.Cp, line_dash="dot", line_color="green",
                  annotation_text=f"Cp_rated={m.Cp}", row=1, col=2)

    # Panel 3: Density correction
    for rho in [1000.0, 1020.0, 1025.0, 1030.0]:
        r_rho = model.predict({"current_speed_ms": speeds, "water_density": rho})
        fig.add_trace(go.Scatter(x=speeds, y=r_rho["power_kw"],
                                 name=f"rho={rho:.0f}"), row=2, col=1)

    # Panel 4: Power map heatmap
    v_grid = np.linspace(0.0, 4.5, 60)
    rho_grid = np.linspace(1000.0, 1035.0, 40)
    power_map = np.zeros((len(rho_grid), len(v_grid)))
    for i, rho in enumerate(rho_grid):
        r_map = model.predict({"current_speed_ms": v_grid, "water_density": rho})
        power_map[i, :] = r_map["power_kw"]
    fig.add_trace(go.Heatmap(x=v_grid, y=rho_grid, z=power_map, colorscale="Blues",
                             colorbar=dict(title="kW"), name="Power (kW)"), row=2, col=2)

    fig.update_xaxes(title_text="Current Speed (m/s)", row=1, col=1)
    fig.update_xaxes(title_text="Current Speed (m/s)", row=1, col=2)
    fig.update_xaxes(title_text="Current Speed (m/s)", row=2, col=1)
    fig.update_xaxes(title_text="Current Speed (m/s)", row=2, col=2)
    fig.update_yaxes(title_text="Power (kW)", row=1, col=1)
    fig.update_yaxes(title_text="Cp", row=1, col=2)
    fig.update_yaxes(title_text="Power (kW)", row=2, col=1)
    fig.update_yaxes(title_text="Water Density (kg/m³)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}",
        height=800, template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
