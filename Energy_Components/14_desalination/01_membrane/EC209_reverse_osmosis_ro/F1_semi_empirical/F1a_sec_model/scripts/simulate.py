"""EC209 — Reverse Osmosis (RO) — F1a — Simulation & HTML Report"""
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
            "SEC vs Recovery (various salinities)",
            "SEC vs Feed Salinity (recovery=0.45)",
            "Feed Pressure vs Salinity",
            "SEC Map (Salinity vs Recovery)",
        ],
        vertical_spacing=0.14)

    # Plot 1: SEC vs recovery for various salinities
    recoveries = np.linspace(0.2, 0.6, 100)
    for S in [5.0, 15.0, 25.0, 35.0, 45.0]:
        r = model.predict({"feed_salinity": S, "recovery": recoveries})
        fig.add_trace(go.Scatter(x=recoveries, y=r["sec_kwhm3"],
            name=f"S={S} g/L"), row=1, col=1)

    # Plot 2: SEC vs feed salinity
    salinities = np.linspace(1.0, 45.0, 100)
    for rec in [0.3, 0.4, 0.45, 0.5, 0.6]:
        r = model.predict({"feed_salinity": salinities, "recovery": rec})
        fig.add_trace(go.Scatter(x=salinities, y=r["sec_kwhm3"],
            name=f"r={rec:.2f}", showlegend=True), row=1, col=2)

    # Plot 3: Feed pressure vs salinity for several recoveries
    for rec in [0.3, 0.4, 0.5, 0.6]:
        r = model.predict({"feed_salinity": salinities, "recovery": rec})
        fig.add_trace(go.Scatter(x=salinities, y=r["feed_pressure_bar"],
            name=f"r={rec:.1f}", showlegend=False), row=2, col=1)

    # Plot 4: SEC heatmap
    S_grid = np.linspace(1.0, 45.0, 50)
    r_grid = np.linspace(0.2, 0.6, 50)
    sec_map = np.zeros((50, 50))
    for i, S in enumerate(S_grid):
        res = model.predict({"feed_salinity": S, "recovery": r_grid})
        sec_map[i, :] = res["sec_kwhm3"]
    fig.add_trace(go.Heatmap(
        x=r_grid, y=S_grid, z=sec_map,
        colorscale="Jet", colorbar=dict(title="SEC (kWh/m3)"),
        zmin=1.0, zmax=8.0), row=2, col=2)

    fig.update_xaxes(title_text="Recovery (-)", row=1, col=1)
    fig.update_xaxes(title_text="Feed Salinity (g/L)", row=1, col=2)
    fig.update_xaxes(title_text="Feed Salinity (g/L)", row=2, col=1)
    fig.update_xaxes(title_text="Recovery (-)", row=2, col=2)
    fig.update_yaxes(title_text="SEC (kWh/m3)", row=1, col=1)
    fig.update_yaxes(title_text="SEC (kWh/m3)", row=1, col=2)
    fig.update_yaxes(title_text="Feed Pressure (bar)", row=2, col=1)
    fig.update_yaxes(title_text="Feed Salinity (g/L)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}",
        height=850, template="plotly_white")

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
