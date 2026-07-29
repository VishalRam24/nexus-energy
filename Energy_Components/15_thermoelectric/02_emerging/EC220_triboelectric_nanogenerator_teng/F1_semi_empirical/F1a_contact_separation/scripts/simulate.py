"""EC220 — Triboelectric Nanogenerator (TENG) — F1a — Simulation & HTML Report"""
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
            "Power vs Load Resistance (various frequencies)",
            "Power vs Frequency (various R_load)",
            "Power Density vs Load (mW/cm²)",
            "Power Map (frequency vs R_load log scale)",
        ],
        vertical_spacing=0.14)

    R_range = np.logspace(4, 10, 100)
    f_range = np.linspace(0.5, 30.0, 100)

    # Plot 1: Power vs R_load
    for freq in [1.0, 3.0, 10.0]:
        r = model.predict({"frequency": freq, "R_load": R_range})
        fig.add_trace(go.Scatter(x=np.log10(R_range), y=r["power_avg_w"]*1000.0,
            name=f"f={freq}Hz"), row=1, col=1)

    # Plot 2: Power vs frequency
    for R in [1e6, 1e7, 1e8]:
        r = model.predict({"frequency": f_range, "R_load": R})
        fig.add_trace(go.Scatter(x=f_range, y=r["power_avg_w"]*1000.0,
            name=f"R={R:.0e}Ω"), row=1, col=2)

    # Plot 3: Power density vs R_load
    r3 = model.predict({"frequency": 3.0, "R_load": R_range})
    fig.add_trace(go.Scatter(x=np.log10(R_range), y=r3["power_density_mwcm2"],
        name="P_density @3Hz", line=dict(color="firebrick")), row=2, col=1)

    # Plot 4: Power heatmap
    f_grid = np.linspace(0.5, 20.0, 30)
    R_grid = np.logspace(5, 9, 30)
    P_map = np.zeros((30, 30))
    for i, f in enumerate(f_grid):
        r4 = model.predict({"frequency": f, "R_load": R_grid})
        P_map[i, :] = r4["power_avg_w"] * 1000.0
    fig.add_trace(go.Heatmap(
        x=np.log10(R_grid), y=f_grid, z=P_map,
        colorscale="Plasma", colorbar=dict(title="Power (mW)"),
        name="Power mW"), row=2, col=2)

    fig.update_xaxes(title_text="log10(R_load/Ω)", row=1, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=2)
    fig.update_xaxes(title_text="log10(R_load/Ω)", row=2, col=1)
    fig.update_xaxes(title_text="log10(R_load/Ω)", row=2, col=2)
    fig.update_yaxes(title_text="Power (mW)", row=1, col=1)
    fig.update_yaxes(title_text="Power (mW)", row=1, col=2)
    fig.update_yaxes(title_text="Power density (mW/cm²)", row=2, col=1)
    fig.update_yaxes(title_text="Frequency (Hz)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}",
        height=850, template="plotly_white")

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
