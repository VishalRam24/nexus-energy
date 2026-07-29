"""EC014 — Metal Hydride H2 Storage — F1a van't Hoff — Simulation & HTML Report"""
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
            "Plateau Pressure vs Temperature (SOC=0.5)",
            "PCT Diagram at 298 K (Pressure-Composition-Temperature)",
            "Stored H2 Mass vs SOC",
            "Heat of Reaction vs H2 Mass Transferred",
        ],
        vertical_spacing=0.14, horizontal_spacing=0.1,
    )

    # Plot 1: Plateau pressure vs Temperature (absorption & desorption)
    T_vals = np.linspace(253, 373, 100)
    colors = {"desorption": "#1f77b4", "absorption": "#ff7f0e"}
    for mode, color in colors.items():
        r = model.predict({"temperature": T_vals, "soc": 0.5, "mode": mode})
        fig.add_trace(go.Scatter(
            x=T_vals - 273.15, y=r["plateau_pressure_bar"],
            name=f"P_plateau ({mode})", line=dict(color=color)
        ), row=1, col=1)

    # Plot 2: PCT diagram — pressure vs SOC at multiple temperatures
    soc_vals = np.linspace(0.01, 0.99, 100)
    T_pct = [273.15, 298.15, 323.15, 353.15]
    pct_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    pct_labels = ["-0 C", "25 C", "50 C", "80 C"]
    for T, c, lbl in zip(T_pct, pct_colors, pct_labels):
        r = model.predict({"temperature": T, "soc": soc_vals, "mode": "desorption"})
        fig.add_trace(go.Scatter(
            x=soc_vals * 100, y=r["plateau_pressure_bar"],
            name=lbl, line=dict(color=c)
        ), row=1, col=2)

    # Plot 3: Stored H2 mass vs SOC
    soc_range = np.linspace(0, 1, 50)
    r3 = model.predict({"temperature": 298.15, "soc": soc_range, "mode": "desorption"})
    fig.add_trace(go.Scatter(
        x=soc_range * 100, y=r3["stored_mass_kg"],
        name="Stored H2 (298 K)", line=dict(color="#2ca02c")
    ), row=2, col=1)

    # Plot 4: Heat of reaction vs delta_m_H2
    delta_m = np.linspace(0, 0.14, 50)
    r4 = model.predict({"temperature": 298.15, "soc": 0.5, "mode": "absorption",
                        "delta_m_H2_kg": delta_m})
    fig.add_trace(go.Scatter(
        x=delta_m * 1000, y=r4["heat_of_reaction_kJ"],
        name="Heat released (absorption)", line=dict(color="#9467bd")
    ), row=2, col=2)

    fig.update_xaxes(title_text="Temperature (°C)", row=1, col=1)
    fig.update_xaxes(title_text="H2 Content (% of max)", row=1, col=2)
    fig.update_xaxes(title_text="SOC (%)", row=2, col=1)
    fig.update_xaxes(title_text="H2 Transferred (g)", row=2, col=2)
    fig.update_yaxes(title_text="Plateau Pressure (bar)", row=1, col=1)
    fig.update_yaxes(title_text="Pressure (bar)", row=1, col=2)
    fig.update_yaxes(title_text="H2 Mass (kg)", row=2, col=1)
    fig.update_yaxes(title_text="Heat (kJ)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} van't Hoff<br>"
              f"<sup>LaNi5 reference material | {info['source']}</sup>",
        height=850, template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
