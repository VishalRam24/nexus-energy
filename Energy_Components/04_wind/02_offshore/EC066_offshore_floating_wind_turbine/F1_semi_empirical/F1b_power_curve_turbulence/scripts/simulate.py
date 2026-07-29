"""EC066 — Offshore Floating Wind — F1b Turbulence + Pitch — Simulation & HTML Report"""

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
            "Power vs Wind Speed (pitch effect)",
            "Power vs Wind Speed (TI effect)",
            "Platform Pitch Factor vs Pitch Angle",
            "Air Density vs Temperature & Humidity",
        ],
        vertical_spacing=0.14)

    v = np.linspace(0, 25, 100)

    # Power vs pitch
    for pitch in [0, 2, 5, 8, 10]:
        r = model.predict({"wind_speed_m_s": v, "turbulence_intensity": 0.07,
                           "platform_pitch_deg": pitch})
        fig.add_trace(go.Scatter(x=v, y=r["power_kw"],
                                  name=f"pitch={pitch}°"), row=1, col=1)

    # Power vs TI (offshore range)
    for ti in [0.0, 0.06, 0.08, 0.10, 0.15]:
        r = model.predict({"wind_speed_m_s": v, "turbulence_intensity": ti,
                           "platform_pitch_deg": 0.0})
        fig.add_trace(go.Scatter(x=v, y=r["power_kw"],
                                  name=f"TI={ti:.2f}",
                                  showlegend=False), row=1, col=2)

    # Pitch factor vs angle
    pitch_range = np.linspace(0, 15, 60)
    pf = model._model.platform_pitch_factor(pitch_range)
    fig.add_trace(go.Scatter(x=pitch_range, y=pf, name="cos²(θ)",
                              showlegend=False), row=2, col=1)

    # Air density
    T_range = np.linspace(-5, 35, 50)
    for rh in [0.5, 0.8, 1.0]:
        rho = model._model.humid_air_density(T_range, rh)
        fig.add_trace(go.Scatter(x=T_range, y=rho, name=f"RH={rh:.0%}",
                                  showlegend=False), row=2, col=2)

    fig.update_xaxes(title_text="Wind Speed (m/s)", row=1, col=1)
    fig.update_xaxes(title_text="Wind Speed (m/s)", row=1, col=2)
    fig.update_xaxes(title_text="Platform Pitch (°)", row=2, col=1)
    fig.update_xaxes(title_text="Temperature (°C)", row=2, col=2)
    fig.update_yaxes(title_text="Power (kW)", row=1, col=1)
    fig.update_yaxes(title_text="Power (kW)", row=1, col=2)
    fig.update_yaxes(title_text="Pitch Factor", row=2, col=1)
    fig.update_yaxes(title_text="Air Density (kg/m³)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} Turbulence + Pitch Penalty",
        height=800, template="plotly_white"
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
