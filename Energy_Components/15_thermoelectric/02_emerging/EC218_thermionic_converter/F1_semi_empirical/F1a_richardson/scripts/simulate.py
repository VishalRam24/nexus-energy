"""EC218 — Thermionic Converter — F1a — Simulation & HTML Report"""
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
            "Emission Current Density vs T_emitter",
            "Efficiency vs T_emitter (various T_collector)",
            "Net Power vs T_emitter",
            "Power Map (T_emitter vs T_collector)",
        ],
        vertical_spacing=0.14)

    T_e = np.linspace(1200.0, 2000.0, 100)

    # Plot 1: Emission current density (Richardson-Dushman)
    for T_c in [600.0, 800.0, 1000.0]:
        r = model.predict({"T_emitter": T_e, "T_collector": T_c})
        fig.add_trace(go.Scatter(x=T_e, y=r["J_emitter_Am2"],
            name=f"J_e (T_c={T_c:.0f}K)"), row=1, col=1)

    # Plot 2: Efficiency vs T_emitter
    for T_c in [600.0, 800.0, 1000.0, 1100.0]:
        r = model.predict({"T_emitter": T_e, "T_collector": T_c})
        fig.add_trace(go.Scatter(x=T_e, y=r["efficiency"] * 100.0,
            name=f"eta T_c={T_c:.0f}K"), row=1, col=2)

    # Plot 3: Net power vs T_emitter
    for T_c in [600.0, 800.0, 1000.0]:
        r = model.predict({"T_emitter": T_e, "T_collector": T_c})
        fig.add_trace(go.Scatter(x=T_e, y=r["power_w"] * 1000.0,
            name=f"P T_c={T_c:.0f}K"), row=2, col=1)

    # Plot 4: Power heatmap
    T_e_grid = np.linspace(1200.0, 2000.0, 40)
    T_c_grid = np.linspace(400.0, 1100.0, 40)
    P_map = np.zeros((40, 40))
    for i, T_c in enumerate(T_c_grid):
        r = model.predict({"T_emitter": T_e_grid, "T_collector": T_c})
        P_map[i, :] = r["power_w"] * 1000.0
    fig.add_trace(go.Heatmap(
        x=T_e_grid, y=T_c_grid, z=P_map,
        colorscale="Plasma", colorbar=dict(title="Power (mW)"),
        name="Power mW"), row=2, col=2)

    fig.update_xaxes(title_text="T_emitter (K)", row=1, col=1)
    fig.update_xaxes(title_text="T_emitter (K)", row=1, col=2)
    fig.update_xaxes(title_text="T_emitter (K)", row=2, col=1)
    fig.update_xaxes(title_text="T_emitter (K)", row=2, col=2)
    fig.update_yaxes(title_text="J (A/m^2)", row=1, col=1)
    fig.update_yaxes(title_text="Efficiency (%)", row=1, col=2)
    fig.update_yaxes(title_text="Power (mW)", row=2, col=1)
    fig.update_yaxes(title_text="T_collector (K)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}",
        height=850, template="plotly_white")

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
