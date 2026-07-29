"""EC064 — Small/Micro Wind Turbine — F1b — Simulation Scenarios + HTML Report"""

import json
import numpy as np
from pathlib import Path

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

import sys
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel


def run_scenarios(model: ComponentModel):
    v = np.linspace(0, 22, 200)
    ti_levels = [0.0, 0.10, 0.20, 0.30]

    # Scenario 1: TI sweep at standard atmosphere
    ti_results = {}
    for ti in ti_levels:
        r = model.predict({"wind_speed_m_s": v, "turbulence_intensity": ti})
        ti_results[ti] = r

    # Scenario 2: Air density effect (altitude / temperature)
    site_conditions = [
        {"label": "Sea level, 15°C, dry", "pressure_pa": 101325, "air_temperature_degC": 15.0, "relative_humidity": 0.0},
        {"label": "Sea level, 35°C, dry", "pressure_pa": 101325, "air_temperature_degC": 35.0, "relative_humidity": 0.0},
        {"label": "1500 m altitude, 15°C", "pressure_pa": 84560, "air_temperature_degC": 15.0, "relative_humidity": 0.0},
        {"label": "Sea level, 15°C, 80% RH", "pressure_pa": 101325, "air_temperature_degC": 15.0, "relative_humidity": 0.80},
    ]
    site_results = {}
    for sc in site_conditions:
        r = model.predict({
            "wind_speed_m_s": v,
            "turbulence_intensity": 0.15,
            "pressure_pa": sc["pressure_pa"],
            "air_temperature_degC": sc["air_temperature_degC"],
            "relative_humidity": sc["relative_humidity"],
        })
        site_results[sc["label"]] = r

    # Scenario 3: Cp vs wind speed
    r_cp = model.predict({"wind_speed_m_s": v, "turbulence_intensity": 0.15})

    return v, ti_results, site_results, r_cp


def generate_report(output_path=None):
    model = ComponentModel()
    v, ti_results, site_results, r_cp = run_scenarios(model)

    if not HAS_PLOTLY:
        print("Plotly not installed. Skipping HTML report generation.")
        return

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Power Curve — Turbulence Intensity Sweep",
            "Air Density Effect on Power",
            "Power Coefficient (Cp) vs Wind Speed",
            "Turbulence Correction Factor",
        ],
    )

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    # Plot 1: TI sweep
    for (ti, r), color in zip(ti_results.items(), colors):
        fig.add_trace(
            go.Scatter(x=v, y=r["power_kw"], mode="lines",
                       name=f"TI={ti:.2f}", line=dict(color=color)),
            row=1, col=1,
        )

    # Plot 2: Site conditions
    for (label, r), color in zip(site_results.items(), colors):
        fig.add_trace(
            go.Scatter(x=v, y=r["power_kw"], mode="lines",
                       name=label, line=dict(color=color), showlegend=True),
            row=1, col=2,
        )

    # Plot 3: Cp
    fig.add_trace(
        go.Scatter(x=v, y=r_cp["power_coefficient"], mode="lines",
                   name="Cp (TI=0.15)", line=dict(color="#1f77b4")),
        row=2, col=1,
    )
    betz = np.full_like(v, 16.0 / 27.0)
    fig.add_trace(
        go.Scatter(x=v, y=betz, mode="lines", name="Betz limit",
                   line=dict(color="red", dash="dash")),
        row=2, col=1,
    )

    # Plot 4: Turbulence correction at various TI
    for ti, color in zip([0.10, 0.20, 0.30], colors[1:]):
        r = model.predict({"wind_speed_m_s": v, "turbulence_intensity": ti})
        fig.add_trace(
            go.Scatter(x=v, y=r["turbulence_correction"] * 100,
                       mode="lines", name=f"TI={ti:.2f}",
                       line=dict(color=color)),
            row=2, col=2,
        )

    fig.update_xaxes(title_text="Wind Speed (m/s)")
    fig.update_yaxes(title_text="Power (kW)", row=1, col=1)
    fig.update_yaxes(title_text="Power (kW)", row=1, col=2)
    fig.update_yaxes(title_text="Cp (–)", row=2, col=1)
    fig.update_yaxes(title_text="Turbulence Correction (% of rated)", row=2, col=2)

    fig.update_layout(
        title_text="EC064 Small/Micro Wind Turbine — F1b: Turbulence + Air Density",
        height=700,
    )

    if output_path is None:
        output_path = Path(__file__).parent.parent / "simulation_report.html"

    fig.write_html(str(output_path))
    print(f"Report saved: {output_path}")


if __name__ == "__main__":
    generate_report()
