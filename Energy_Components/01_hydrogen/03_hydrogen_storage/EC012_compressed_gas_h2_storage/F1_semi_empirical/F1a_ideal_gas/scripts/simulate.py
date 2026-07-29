"""EC012 — Compressed Gas H2 Storage — F1a — Simulation & HTML Report"""
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
            "Stored H2 Mass vs Pressure",
            "Energy Stored vs Pressure",
            "Compressibility Factor Z vs Pressure",
            "Compression Work vs Final Pressure",
        ],
        vertical_spacing=0.14, horizontal_spacing=0.1,
    )

    P = np.linspace(10, 900, 200)
    T_vals = [253.15, 298.15, 343.15]
    colors = ["#1f77b4", "#ff7f0e", "#d62728"]
    labels = ["-20 C", "25 C", "70 C"]

    for T, c, lbl in zip(T_vals, colors, labels):
        r = model.predict({"pressure": P, "temperature": T})
        fig.add_trace(go.Scatter(x=P, y=r["stored_mass_kg"], name=lbl, line=dict(color=c)), row=1, col=1)
        fig.add_trace(go.Scatter(x=P, y=r["energy_stored_MJ"], name=lbl, line=dict(color=c), showlegend=False), row=1, col=2)

    r_ref = model.predict({"pressure": P, "temperature": 298.15})
    fig.add_trace(go.Scatter(x=P, y=r_ref["compressibility_Z"], name="Z (25 C)", line=dict(color="#2ca02c")), row=2, col=1)
    fig.add_trace(go.Scatter(x=P, y=r_ref["compression_work_kJ_per_kg"], name="W_comp (25 C)", line=dict(color="#9467bd")), row=2, col=2)

    fig.update_xaxes(title_text="Pressure (bar)", row=1, col=1)
    fig.update_xaxes(title_text="Pressure (bar)", row=1, col=2)
    fig.update_xaxes(title_text="Pressure (bar)", row=2, col=1)
    fig.update_xaxes(title_text="Final Pressure (bar)", row=2, col=2)
    fig.update_yaxes(title_text="H2 Mass (kg)", row=1, col=1)
    fig.update_yaxes(title_text="Energy (MJ)", row=1, col=2)
    fig.update_yaxes(title_text="Z", row=2, col=1)
    fig.update_yaxes(title_text="Work (kJ/kg)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} Ideal Gas + Z Model<br>"
              f"<sup>125 L Type IV tank | {info['source']}</sup>",
        height=850, template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
