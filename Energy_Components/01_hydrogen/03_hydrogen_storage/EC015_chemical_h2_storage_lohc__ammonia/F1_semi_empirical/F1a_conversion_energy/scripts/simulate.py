"""EC015 — Chemical H2 Storage (LOHC/Ammonia) — F1a — Simulation & HTML Report"""
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
            "Carrier Mass Required vs H2 Mass",
            "Thermal Energy Demand vs H2 Mass (Release)",
            "Specific Energy Demand (MJ/kg H2)",
            "Round-Trip Efficiency Comparison",
        ],
        vertical_spacing=0.14, horizontal_spacing=0.12,
    )

    m_H2 = np.logspace(-1, 3, 100)  # 0.1 to 1000 kg H2

    # Plot 1: Carrier mass required
    r_lohc = model.predict({"h2_mass_kg": m_H2, "mode": "lohc", "direction": "dehydrogenation"})
    r_nh3 = model.predict({"h2_mass_kg": m_H2, "mode": "ammonia", "direction": "cracking"})
    fig.add_trace(go.Scatter(x=m_H2, y=r_lohc["carrier_mass_kg"],
                             name="LOHC (DBT)", line=dict(color="#1f77b4")), row=1, col=1)
    fig.add_trace(go.Scatter(x=m_H2, y=r_nh3["carrier_mass_kg"],
                             name="Ammonia", line=dict(color="#ff7f0e")), row=1, col=1)

    # Plot 2: Thermal energy demand for release
    fig.add_trace(go.Scatter(x=m_H2, y=r_lohc["thermal_energy_MJ"],
                             name="LOHC dehydrogenation", line=dict(color="#1f77b4"),
                             showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=m_H2, y=r_nh3["thermal_energy_MJ"],
                             name="NH3 cracking", line=dict(color="#ff7f0e"),
                             showlegend=False), row=1, col=2)

    # Plot 3: Specific energy (bar chart)
    carriers = ["LOHC\nDehydrogenation", "LOHC\nHydrogenation", "NH3\nCracking", "NH3\nSynthesis"]
    r_lohc_h = model.predict({"h2_mass_kg": 1.0, "mode": "lohc", "direction": "hydrogenation"})
    r_nh3_s = model.predict({"h2_mass_kg": 1.0, "mode": "ammonia", "direction": "synthesis"})
    spec_energies = [
        float(np.atleast_1d(r_lohc["specific_energy_MJ_per_kg_H2"])[0]),
        float(np.atleast_1d(r_lohc_h["specific_energy_MJ_per_kg_H2"])[0]),
        float(np.atleast_1d(r_nh3["specific_energy_MJ_per_kg_H2"])[0]),
        float(np.atleast_1d(r_nh3_s["specific_energy_MJ_per_kg_H2"])[0]),
    ]
    bar_colors = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78"]
    fig.add_trace(go.Bar(
        x=["LOHC\nDehydrog.", "LOHC\nHydrog.", "NH3\nCracking", "NH3\nSynthesis"],
        y=spec_energies,
        marker_color=bar_colors, name="Specific Energy"
    ), row=2, col=1)

    # Plot 4: Round-trip efficiency
    fig.add_trace(go.Bar(
        x=["LOHC (DBT)", "Ammonia"],
        y=[r_lohc["roundtrip_efficiency"] * 100, r_nh3["roundtrip_efficiency"] * 100],
        marker_color=["#1f77b4", "#ff7f0e"],
        name="Round-trip η"
    ), row=2, col=2)

    fig.update_xaxes(title_text="H2 Mass (kg)", type="log", row=1, col=1)
    fig.update_xaxes(title_text="H2 Mass (kg)", type="log", row=1, col=2)
    fig.update_xaxes(title_text="Process", row=2, col=1)
    fig.update_xaxes(title_text="Carrier", row=2, col=2)
    fig.update_yaxes(title_text="Carrier Mass (kg)", type="log", row=1, col=1)
    fig.update_yaxes(title_text="Thermal Energy (MJ)", type="log", row=1, col=2)
    fig.update_yaxes(title_text="Specific Energy (MJ/kg H2)", row=2, col=1)
    fig.update_yaxes(title_text="Round-trip η (%)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}<br>"
              f"<sup>{info['source']}</sup>",
        height=850, template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
