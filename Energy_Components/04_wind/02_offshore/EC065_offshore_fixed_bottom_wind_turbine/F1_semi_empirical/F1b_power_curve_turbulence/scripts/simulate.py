"""EC065 — Offshore Wind — F1b Turbulence — Simulation & HTML Report"""

import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()

    fig = make_subplots(rows=2, cols=2,
        subplot_titles=["Power vs Wind Speed (TI effect)",
                        "Air Density vs Temperature & Humidity",
                        "Power Coefficient vs Wind Speed",
                        "Power at 10 m/s vs Humidity"],
        vertical_spacing=0.12)

    v = np.linspace(0, 25, 100)

    # Power curves at different TI (offshore range)
    for ti in [0.0, 0.06, 0.08, 0.10, 0.15]:
        r = model.predict({"wind_speed_m_s": v, "turbulence_intensity": ti})
        fig.add_trace(go.Scatter(x=v, y=r["power_kw"],
                                  name=f"TI={ti:.2f}"), row=1, col=1)

    # Air density vs temperature at different humidities
    T = np.linspace(-5, 35, 50)
    for rh in [0.0, 0.5, 1.0]:
        rho = model._model.humid_air_density(T, rh)
        fig.add_trace(go.Scatter(x=T, y=rho, name=f"RH={rh:.0%}",
                                  showlegend=False), row=1, col=2)

    # Cp
    r = model.predict({"wind_speed_m_s": v, "turbulence_intensity": 0.08})
    fig.add_trace(go.Scatter(x=v, y=r["power_coefficient"], name="Cp",
                              showlegend=False), row=2, col=1)

    # Power at 10 m/s vs humidity
    rh_range = np.linspace(0, 1, 50)
    r = model.predict({"wind_speed_m_s": 10.0, "turbulence_intensity": 0.08,
                        "air_temperature_degC": 15.0, "relative_humidity": rh_range})
    fig.add_trace(go.Scatter(x=rh_range * 100, y=r["power_kw"], name="P(RH)",
                              showlegend=False), row=2, col=2)

    fig.update_xaxes(title_text="Wind Speed (m/s)", row=1, col=1)
    fig.update_xaxes(title_text="Temperature (C)", row=1, col=2)
    fig.update_xaxes(title_text="Wind Speed (m/s)", row=2, col=1)
    fig.update_xaxes(title_text="Relative Humidity (%)", row=2, col=2)
    fig.update_yaxes(title_text="Power (kW)", row=1, col=1)
    fig.update_yaxes(title_text="rho (kg/m3)", row=1, col=2)
    fig.update_yaxes(title_text="Cp", row=2, col=1)
    fig.update_yaxes(title_text="Power (kW)", row=2, col=2)

    fig.update_layout(title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} Turbulence + Humid Density",
                      height=800, template="plotly_white")

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
