"""EC010 — Solid Oxide Electrolyser (SOEC) — F1a — Simulation & HTML Report"""
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
            "ASR vs Temperature (Arrhenius)",
            "Hydrogen Production Rate vs Current Density",
            "Stack Efficiency vs Current Density",
        ],
        vertical_spacing=0.14,
        horizontal_spacing=0.1,
    )

    j_arr = np.linspace(0, 2.0, 200)
    temps = [600, 700, 800, 900]
    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]

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
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=j_arr, y=r["efficiency"] * 100,
            name=f"T={T}°C", line=dict(color=col), legendgroup=f"T{T}",
            showlegend=False
        ), row=2, col=2)

    # ASR vs Temperature
    T_range = np.linspace(600, 900, 200)
    r_T = model.predict({"current_density": 1.0, "temperature": T_range})
    fig.add_trace(go.Scatter(
        x=T_range, y=r_T["asr"],
        name="ASR(T)", line=dict(color="#9467bd", width=2), showlegend=True
    ), row=1, col=2)

    # Thermo-neutral voltage reference
    V_tn = 1.285
    fig.add_hline(y=V_tn, line_dash="dash", line_color="red",
                  annotation_text=f"V_tn={V_tn}V (thermo-neutral)",
                  annotation_position="bottom right", row=1, col=1)

    # Axis labels
    fig.update_xaxes(title_text="Current Density (A/cm²)", row=1, col=1)
    fig.update_xaxes(title_text="Temperature (°C)", row=1, col=2)
    fig.update_xaxes(title_text="Current Density (A/cm²)", row=2, col=1)
    fig.update_xaxes(title_text="Current Density (A/cm²)", row=2, col=2)
    fig.update_yaxes(title_text="Cell Voltage (V)", row=1, col=1)
    fig.update_yaxes(title_text="ASR (Ω·cm²)", row=1, col=2)
    fig.update_yaxes(title_text="H₂ Rate (mol/hr)", row=2, col=1)
    fig.update_yaxes(title_text="Efficiency (%)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} V-I Characteristic<br>"
              f"<sup>ASR-Arrhenius model | {info['source']}</sup>",
        height=850,
        template="plotly_white",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)")
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
