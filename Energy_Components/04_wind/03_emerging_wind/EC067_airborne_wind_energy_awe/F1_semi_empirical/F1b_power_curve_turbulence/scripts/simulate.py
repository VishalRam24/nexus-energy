"""EC067 — Airborne Wind Energy (AWE) — F1b — Simulation Scenarios + HTML Report"""

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
    v = np.linspace(0, 26, 200)

    # Scenario 1: TI sweep at standard altitude
    ti_levels = [0.0, 0.05, 0.10, 0.15]
    ti_results = {}
    for ti in ti_levels:
        r = model.predict({"wind_speed_m_s": v, "turbulence_intensity": ti})
        ti_results[ti] = r

    # Scenario 2: Altitude effect
    altitudes = [100, 200, 400, 600]
    alt_results = {}
    for alt in altitudes:
        r = model.predict({"wind_speed_m_s": v, "turbulence_intensity": 0.08,
                           "altitude_m": alt})
        alt_results[alt] = r

    # Scenario 3: Power vs Loyd limit at 400 m
    r_loyd = model.predict({"wind_speed_m_s": v, "turbulence_intensity": 0.08,
                             "altitude_m": 400})

    # Scenario 4: Turbulence correction
    tc_results = {}
    for ti in [0.05, 0.10, 0.15]:
        r = model.predict({"wind_speed_m_s": v, "turbulence_intensity": ti})
        tc_results[ti] = r

    return v, ti_results, alt_results, r_loyd, tc_results


def generate_report(output_path=None):
    model = ComponentModel()
    v, ti_results, alt_results, r_loyd, tc_results = run_scenarios(model)

    if not HAS_PLOTLY:
        print("Plotly not installed. Skipping HTML report generation.")
        return

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "AWE Power Curve — Turbulence Sweep",
            "Altitude Effect on Power (TI=0.08)",
            "Power vs Loyd Theoretical Limit (400 m)",
            "Turbulence Correction Factor",
        ],
    )

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for (ti, r), color in zip(ti_results.items(), colors):
        fig.add_trace(
            go.Scatter(x=v, y=r["power_kw"], mode="lines",
                       name=f"TI={ti:.2f}", line=dict(color=color)),
            row=1, col=1,
        )

    for (alt, r), color in zip(alt_results.items(), colors):
        rho_val = float(r["air_density"])
        fig.add_trace(
            go.Scatter(x=v, y=r["power_kw"], mode="lines",
                       name=f"{alt} m (rho={rho_val:.3f})", line=dict(color=color)),
            row=1, col=2,
        )

    fig.add_trace(
        go.Scatter(x=v, y=r_loyd["power_kw"], mode="lines",
                   name="F1b Power", line=dict(color="#1f77b4")),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(x=v, y=r_loyd["loyd_limit_kw"], mode="lines",
                   name="Loyd Limit", line=dict(color="red", dash="dash")),
        row=2, col=1,
    )

    for (ti, r), color in zip(tc_results.items(), colors[1:]):
        fig.add_trace(
            go.Scatter(x=v, y=r["turbulence_correction"] * 100,
                       mode="lines", name=f"TI={ti:.2f}",
                       line=dict(color=color)),
            row=2, col=2,
        )

    fig.update_xaxes(title_text="Wind Speed at Altitude (m/s)")
    fig.update_yaxes(title_text="Power (kW)", row=1, col=1)
    fig.update_yaxes(title_text="Power (kW)", row=1, col=2)
    fig.update_yaxes(title_text="Power (kW)", row=2, col=1)
    fig.update_yaxes(title_text="Turbulence Correction (% rated)", row=2, col=2)

    fig.update_layout(
        title_text="EC067 Airborne Wind Energy (AWE) — F1b: Turbulence + Altitude Density",
        height=700,
    )

    if output_path is None:
        output_path = Path(__file__).parent.parent / "simulation_report.html"

    fig.write_html(str(output_path))
    print(f"Report saved: {output_path}")


if __name__ == "__main__":
    generate_report()
