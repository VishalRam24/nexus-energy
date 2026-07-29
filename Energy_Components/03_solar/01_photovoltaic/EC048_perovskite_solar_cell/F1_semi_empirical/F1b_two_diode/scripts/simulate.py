"""EC048 — Perovskite PV — F1b Two-Diode + Hysteresis — Simulation & HTML Report"""

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
        subplot_titles=["Power vs Irradiance", "Efficiency vs Temperature",
                        "Hysteresis Effect", "Power vs Irradiance Rate"],
        vertical_spacing=0.12)

    # Power vs Irradiance (steady-state)
    G = np.linspace(50, 1200, 24)
    for T in [15, 25, 45]:
        r = model.predict({"irradiance_w_m2": G, "temperature_cell_degC": T})
        fig.add_trace(go.Scatter(x=G, y=r["p_mp"] * 1000, name=f"T={T}C"), row=1, col=1)

    # Efficiency vs Temperature
    T = np.linspace(-10, 70, 17)
    for Gi in [400, 700, 1000]:
        r = model.predict({"irradiance_w_m2": Gi, "temperature_cell_degC": T})
        fig.add_trace(go.Scatter(x=T, y=r["efficiency"] * 100, name=f"G={Gi}"), row=1, col=2)

    # Hysteresis: power at different irradiance rates
    rates = np.linspace(0, 500, 20)
    r_ss = model.predict({"irradiance_w_m2": 1000.0, "temperature_cell_degC": 25.0,
                           "irradiance_rate_w_m2_s": rates})
    fig.add_trace(go.Scatter(x=rates, y=r_ss["p_mp"] * 1000, name="Hysteresis",
                              showlegend=False), row=2, col=1)

    # Power at fixed G with varying rate
    fig.add_trace(go.Scatter(x=rates, y=r_ss["hysteresis_index"] * 100, name="HI %",
                              showlegend=False), row=2, col=2)

    fig.update_xaxes(title_text="Irradiance (W/m2)", row=1, col=1)
    fig.update_xaxes(title_text="Cell Temp (C)", row=1, col=2)
    fig.update_xaxes(title_text="dG/dt (W/m2/s)", row=2, col=1)
    fig.update_xaxes(title_text="dG/dt (W/m2/s)", row=2, col=2)
    fig.update_yaxes(title_text="Power (mW)", row=1, col=1)
    fig.update_yaxes(title_text="Efficiency (%)", row=1, col=2)
    fig.update_yaxes(title_text="Power (mW)", row=2, col=1)
    fig.update_yaxes(title_text="Hysteresis Index (%)", row=2, col=2)

    fig.update_layout(title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} Two-Diode + Hysteresis",
                      height=800, template="plotly_white")

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
