"""EC219 — Piezoelectric Energy Harvester — F1a — Simulation & HTML Report"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()
    f_n = model._model.f_n

    fig = make_subplots(rows=2, cols=2,
        subplot_titles=[
            "Power vs Frequency (frequency sweep)",
            "Power vs Acceleration (at resonance) — P ~ a²",
            "Power Frequency Response for Various Accelerations",
            "Power Map (acceleration vs frequency)",
        ],
        vertical_spacing=0.14)

    f = np.linspace(20.0, 300.0, 200)

    # Plot 1: Frequency sweep at various accelerations
    for a in [4.9, 9.81, 19.6]:
        r = model.predict({"acceleration": a, "frequency": f})
        fig.add_trace(go.Scatter(x=f, y=r["power_uw"],
            name=f"a={a:.1f} m/s²"), row=1, col=1)

    # Plot 2: Power vs acceleration at resonance (P ~ a^2)
    a_range = np.linspace(0.1, 30.0, 100)
    r2 = model.predict({"acceleration": a_range, "frequency": f_n})
    fig.add_trace(go.Scatter(x=a_range, y=r2["power_uw"],
        name="P(a) at resonance", line=dict(color="firebrick")), row=1, col=2)

    # Plot 3: Normalized frequency response
    for a in [5.0, 10.0, 20.0]:
        r3 = model.predict({"acceleration": a, "frequency": f})
        fig.add_trace(go.Scatter(x=f / f_n, y=r3["power_uw"],
            name=f"a={a:.0f} m/s²"), row=2, col=1)

    # Plot 4: Power heatmap
    a_grid = np.linspace(1.0, 30.0, 40)
    f_grid = np.linspace(20.0, 300.0, 40)
    P_map = np.zeros((40, 40))
    for i, a in enumerate(a_grid):
        r4 = model.predict({"acceleration": a, "frequency": f_grid})
        P_map[i, :] = r4["power_uw"]
    fig.add_trace(go.Heatmap(
        x=f_grid, y=a_grid, z=P_map,
        colorscale="Viridis", colorbar=dict(title="Power (uW)"),
        name="Power uW"), row=2, col=2)

    fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=1)
    fig.update_xaxes(title_text="Acceleration (m/s²)", row=1, col=2)
    fig.update_xaxes(title_text="f / f_n", row=2, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=2)
    fig.update_yaxes(title_text="Power (uW)", row=1, col=1)
    fig.update_yaxes(title_text="Power (uW)", row=1, col=2)
    fig.update_yaxes(title_text="Power (uW)", row=2, col=1)
    fig.update_yaxes(title_text="Acceleration (m/s²)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}",
        height=850, template="plotly_white")

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
