"""EC009 — Alkaline Electrolyser (AEL) — F1a — Simulation & HTML Report"""
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
            "V-I Characteristic (Cell Voltage vs Current Density)",
            "Hydrogen Production Rate vs Current Density",
            "Stack Efficiency vs Current Density",
            "Power Consumption vs Current Density",
        ],
        vertical_spacing=0.14,
        horizontal_spacing=0.1,
    )

    j_arr = np.linspace(0, 3000, 300)
    temps = [40, 60, 80, 90]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for T, col in zip(temps, colors):
        r = model.predict({"current_density": j_arr, "temperature": float(T)})

        fig.add_trace(go.Scatter(
            x=j_arr, y=r["cell_voltage"],
            name=f"T={T}°C", line=dict(color=col), legendgroup=f"T{T}"
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=j_arr, y=r["hydrogen_rate_mols"] * 3600,
            name=f"T={T}°C", line=dict(color=col), legendgroup=f"T{T}",
            showlegend=False
        ), row=1, col=2)

        fig.add_trace(go.Scatter(
            x=j_arr, y=r["efficiency"] * 100,
            name=f"T={T}°C", line=dict(color=col), legendgroup=f"T{T}",
            showlegend=False
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=j_arr, y=r["power_kw"],
            name=f"T={T}°C", line=dict(color=col), legendgroup=f"T{T}",
            showlegend=False
        ), row=2, col=2)

    # Add E_rev reference line at 80C
    e_rev_80 = 1.229 - 0.0009 * (353.15 - 298.15)
    fig.add_hline(y=e_rev_80, line_dash="dash", line_color="gray",
                  annotation_text=f"E_rev(80°C)={e_rev_80:.3f}V",
                  annotation_position="top right", row=1, col=1)

    # Axis labels
    fig.update_xaxes(title_text="Current Density (A/m²)", row=1, col=1)
    fig.update_xaxes(title_text="Current Density (A/m²)", row=1, col=2)
    fig.update_xaxes(title_text="Current Density (A/m²)", row=2, col=1)
    fig.update_xaxes(title_text="Current Density (A/m²)", row=2, col=2)
    fig.update_yaxes(title_text="Cell Voltage (V)", row=1, col=1)
    fig.update_yaxes(title_text="H₂ Rate (mol/hr)", row=1, col=2)
    fig.update_yaxes(title_text="Efficiency (%)", row=2, col=1)
    fig.update_yaxes(title_text="Power (kW)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} V-I Characteristic<br>"
              f"<sup>Ulleberg (2003) model | {info['source']}</sup>",
        height=850,
        template="plotly_white",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)")
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
