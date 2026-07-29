"""EC198 — Post-Combustion Capture (Amine Scrubbing) — F1a Energy Model — Simulation & HTML Report"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()

    crs   = np.linspace(0.80, 0.95, 100)
    flows = np.linspace(100, 1000, 100)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Specific Energy vs Capture Rate",
            "CO2 Captured & Reboiler Duty vs Flue Gas Rate",
            "Reboiler Duty vs Capture Rate (different CO2 fractions)",
            "Specific Energy Map: Capture Rate × CO2 Fraction",
        ],
        vertical_spacing=0.14,
    )

    # Plot 1: Specific energy vs capture rate at different CO2 fractions
    for x in [0.04, 0.08, 0.12, 0.15]:
        r = model.predict({"flue_gas_rate": 500.0, "co2_fraction": x, "capture_rate": crs})
        fig.add_trace(
            go.Scatter(x=crs * 100, y=r["specific_energy_gjt"],
                       name=f"CO2={int(x*100)}%", line=dict(width=2)),
            row=1, col=1
        )

    # Reference band 3–5 GJ/t
    fig.add_hrect(y0=3.0, y1=5.0, fillcolor="lightgreen", opacity=0.2,
                  annotation_text="Typical 3–5 GJ/t", row=1, col=1)

    # Plot 2: CO2 captured and reboiler vs flow rate
    for co2_frac in [0.08, 0.12, 0.15]:
        r = model.predict({"flue_gas_rate": flows, "co2_fraction": co2_frac, "capture_rate": 0.90})
        fig.add_trace(
            go.Scatter(x=flows, y=r["co2_captured_kgs"],
                       name=f"CO2_cap CO2={int(co2_frac*100)}%", line=dict(width=2)),
            row=1, col=2
        )

    # Reboiler power on second y axis
    for co2_frac in [0.12]:
        r = model.predict({"flue_gas_rate": flows, "co2_fraction": co2_frac, "capture_rate": 0.90})
        fig.add_trace(
            go.Scatter(x=flows, y=r["reboiler_duty_mw"],
                       name=f"Reboiler MW CO2=12%",
                       line=dict(width=2, dash="dash"), showlegend=True),
            row=1, col=2
        )

    # Plot 3: Reboiler duty vs capture rate
    for x in [0.04, 0.08, 0.12, 0.15]:
        r = model.predict({"flue_gas_rate": 500.0, "co2_fraction": x, "capture_rate": crs})
        fig.add_trace(
            go.Scatter(x=crs * 100, y=r["reboiler_duty_mw"],
                       name=f"Reb CO2={int(x*100)}%",
                       line=dict(width=2), showlegend=False),
            row=2, col=1
        )

    # Plot 4: Specific energy heatmap (capture rate x CO2 fraction)
    cr_g  = np.linspace(0.80, 0.95, 30)
    x_g   = np.linspace(0.04, 0.15, 30)
    E_map = np.zeros((len(x_g), len(cr_g)))
    for i, xco2 in enumerate(x_g):
        r = model.predict({"flue_gas_rate": 500.0, "co2_fraction": float(xco2), "capture_rate": cr_g})
        E_map[i, :] = r["specific_energy_gjt"]

    fig.add_trace(
        go.Heatmap(x=cr_g * 100, y=x_g * 100, z=E_map,
                   colorscale="Reds", colorbar=dict(title="GJ/tCO2"),
                   name="E_spec"),
        row=2, col=2
    )

    # Design point
    r_dp = model.predict({"flue_gas_rate": 500.0, "co2_fraction": 0.12, "capture_rate": 0.90})
    fig.add_trace(
        go.Scatter(x=[90.0], y=[float(r_dp["specific_energy_gjt"])],
                   mode="markers", marker=dict(size=14, color="black", symbol="star"),
                   name="Design Point (90%, CO2=12%)", showlegend=True),
        row=1, col=1
    )

    fig.update_xaxes(title_text="Capture Rate (%)", row=1, col=1)
    fig.update_xaxes(title_text="Flue Gas Rate (kg/s)", row=1, col=2)
    fig.update_xaxes(title_text="Capture Rate (%)", row=2, col=1)
    fig.update_xaxes(title_text="Capture Rate (%)", row=2, col=2)
    fig.update_yaxes(title_text="Specific Energy (GJ/tCO2)", row=1, col=1)
    fig.update_yaxes(title_text="CO2 captured (kg/s) / Reboiler (MW)", row=1, col=2)
    fig.update_yaxes(title_text="Reboiler Duty (MW)", row=2, col=1)
    fig.update_yaxes(title_text="CO2 Fraction (%)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}<br>"
              f"<sup>30 wt% MEA | Source: Abu-Zahra et al. (2007), Int. J. GHG Control</sup>",
        height=850,
        template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")

    print("\n--- Design Point Summary (500 kg/s flue gas, CO2=12%, capture=90%) ---")
    for k, v in r_dp.items():
        print(f"  {k} = {float(v):.4f}")


if __name__ == "__main__":
    generate_report()
