"""EC062 — HAWT Onshore — F1b Turbulence — Simulation & HTML Report"""

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
                        "Turbulence Correction vs Wind Speed",
                        "Power Coefficient vs Wind Speed",
                        "CF Correction vs TI at 10 m/s"],
        vertical_spacing=0.12)

    v = np.linspace(0, 25, 100)

    # Power curves at different TI
    for ti in [0.0, 0.10, 0.15, 0.20, 0.25]:
        r = model.predict({"wind_speed_m_s": v, "turbulence_intensity": ti})
        fig.add_trace(go.Scatter(x=v, y=r["power_kw"],
                                  name=f"TI={ti:.2f}"), row=1, col=1)

    # CF correction at different TI
    for ti in [0.10, 0.15, 0.20]:
        r = model.predict({"wind_speed_m_s": v, "turbulence_intensity": ti})
        fig.add_trace(go.Scatter(x=v, y=r["capacity_factor_correction"]*100,
                                  name=f"dCF TI={ti}", showlegend=False), row=1, col=2)

    # Cp at low turbulence
    r = model.predict({"wind_speed_m_s": v, "turbulence_intensity": 0.10})
    fig.add_trace(go.Scatter(x=v, y=r["power_coefficient"], name="Cp(TI=0.10)",
                              showlegend=False), row=2, col=1)

    # CF correction vs TI at fixed wind speed
    ti_range = np.linspace(0, 0.30, 30)
    r = model.predict({"wind_speed_m_s": 10.0, "turbulence_intensity": ti_range})
    fig.add_trace(go.Scatter(x=ti_range, y=r["capacity_factor_correction"]*100,
                              name="dCF(10m/s)", showlegend=False), row=2, col=2)

    fig.update_xaxes(title_text="Wind Speed (m/s)", row=1, col=1)
    fig.update_xaxes(title_text="Wind Speed (m/s)", row=1, col=2)
    fig.update_xaxes(title_text="Wind Speed (m/s)", row=2, col=1)
    fig.update_xaxes(title_text="Turbulence Intensity", row=2, col=2)
    fig.update_yaxes(title_text="Power (kW)", row=1, col=1)
    fig.update_yaxes(title_text="CF Correction (%)", row=1, col=2)
    fig.update_yaxes(title_text="Cp", row=2, col=1)
    fig.update_yaxes(title_text="CF Correction (%)", row=2, col=2)

    fig.update_layout(title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} Turbulence",
                      height=800, template="plotly_white")

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
