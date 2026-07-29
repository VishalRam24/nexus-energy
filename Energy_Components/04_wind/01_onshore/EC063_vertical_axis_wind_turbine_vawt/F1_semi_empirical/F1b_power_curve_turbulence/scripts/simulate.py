"""EC063 — VAWT — F1b Turbulence + Air Density — Simulation & HTML Report"""

import sys
import json
import numpy as np
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
            "Power vs Wind Speed (TI effect)",
            "Air Density vs Temperature & Altitude",
            "Power Coefficient vs Wind Speed",
            "Power at 7 m/s vs TI",
        ],
        vertical_spacing=0.14)

    v = np.linspace(0, 22, 100)

    # Power curves at different TI
    for ti in [0.0, 0.05, 0.10, 0.20, 0.30]:
        r = model.predict({"wind_speed_m_s": v, "turbulence_intensity": ti})
        fig.add_trace(go.Scatter(x=v, y=r["power_kw"],
                                  name=f"TI={ti:.2f}"), row=1, col=1)

    # Air density vs temperature at different altitudes
    T_range = np.linspace(-15, 45, 60)
    for alt in [0, 500, 1500, 3000]:
        rho = model._model.air_density(T_range, alt)
        fig.add_trace(go.Scatter(x=T_range, y=rho, name=f"z={alt} m",
                                  showlegend=False), row=1, col=2)

    # Cp vs wind speed
    r = model.predict({"wind_speed_m_s": v, "turbulence_intensity": 0.0})
    fig.add_trace(go.Scatter(x=v, y=r["power_coefficient"], name="Cp (TI=0)",
                              showlegend=False), row=2, col=1)
    # Betz reference line
    fig.add_trace(go.Scatter(x=[0, 22], y=[0.593, 0.593],
                              mode="lines", line=dict(dash="dash", color="red"),
                              name="Betz limit", showlegend=False), row=2, col=1)

    # Power at 7 m/s vs TI (partial load, key physics check)
    ti_range = np.linspace(0.0, 0.40, 60)
    r = model.predict({"wind_speed_m_s": 7.0, "turbulence_intensity": ti_range})
    fig.add_trace(go.Scatter(x=ti_range * 100, y=r["power_kw"],
                              name="P(TI) at 7 m/s", showlegend=False), row=2, col=2)

    fig.update_xaxes(title_text="Wind Speed (m/s)", row=1, col=1)
    fig.update_xaxes(title_text="Temperature (°C)", row=1, col=2)
    fig.update_xaxes(title_text="Wind Speed (m/s)", row=2, col=1)
    fig.update_xaxes(title_text="TI (%)", row=2, col=2)
    fig.update_yaxes(title_text="Power (kW)", row=1, col=1)
    fig.update_yaxes(title_text="Air Density (kg/m³)", row=1, col=2)
    fig.update_yaxes(title_text="Cp", row=2, col=1)
    fig.update_yaxes(title_text="Power (kW)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} Turbulence + Air Density",
        height=800, template="plotly_white"
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
