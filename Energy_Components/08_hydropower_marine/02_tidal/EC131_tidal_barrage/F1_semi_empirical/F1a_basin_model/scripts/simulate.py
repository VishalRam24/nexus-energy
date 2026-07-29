"""EC131 — Tidal Barrage — F1a — Simulation & HTML Report"""
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
            "Average Power vs Tidal Range",
            "Energy per Tidal Cycle vs Range",
            "Power vs Basin Area (design range R=16 m)",
            "Power Map (MW) — Tidal Range vs Basin Area",
        ],
        vertical_spacing=0.13,
    )

    # Panel 1: Power vs tidal range
    ranges = np.linspace(0.0, 16.0, 200)
    r = model.predict({"tidal_range_m": ranges})
    fig.add_trace(go.Scatter(x=ranges, y=r["avg_power_mw"], name="P_avg (eta applied)",
                             line=dict(color="teal")), row=1, col=1)
    fig.add_trace(go.Scatter(x=ranges, y=r["theoretical_power_kw"] / 1000.0, name="P_theoretical",
                             line=dict(color="gray", dash="dash")), row=1, col=1)

    # Panel 2: Energy per cycle
    fig.add_trace(go.Scatter(x=ranges, y=r["energy_per_cycle_mwh"], name="E/cycle",
                             line=dict(color="blue")), row=1, col=2)

    # Panel 3: Power vs basin area
    areas = np.linspace(1e6, 200e6, 200)
    r2 = model.predict({"tidal_range_m": 16.0, "basin_area_m2": areas})
    fig.add_trace(go.Scatter(x=areas / 1e6, y=r2["avg_power_mw"], name="R=16 m",
                             line=dict(color="orange")), row=2, col=1)

    # Panel 4: Power map heatmap
    R_grid = np.linspace(0.0, 16.0, 50)
    A_grid = np.linspace(1e6, 200e6, 50)
    power_map = np.zeros((len(A_grid), len(R_grid)))
    for i, A in enumerate(A_grid):
        r_map = model.predict({"tidal_range_m": R_grid, "basin_area_m2": A})
        power_map[i, :] = r_map["avg_power_mw"]
    fig.add_trace(go.Heatmap(x=R_grid, y=A_grid / 1e6, z=power_map, colorscale="Teal",
                             colorbar=dict(title="MW"), name="Power (MW)"), row=2, col=2)

    fig.update_xaxes(title_text="Tidal Range (m)", row=1, col=1)
    fig.update_xaxes(title_text="Tidal Range (m)", row=1, col=2)
    fig.update_xaxes(title_text="Basin Area (km²)", row=2, col=1)
    fig.update_xaxes(title_text="Tidal Range (m)", row=2, col=2)
    fig.update_yaxes(title_text="Power (MW)", row=1, col=1)
    fig.update_yaxes(title_text="Energy (MWh)", row=1, col=2)
    fig.update_yaxes(title_text="Power (MW)", row=2, col=1)
    fig.update_yaxes(title_text="Basin Area (km²)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}",
        height=800, template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
