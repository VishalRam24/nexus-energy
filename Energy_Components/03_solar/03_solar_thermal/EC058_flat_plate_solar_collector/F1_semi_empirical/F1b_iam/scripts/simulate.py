"""EC058 — Flat Plate Collector — F1b IAM — Simulation & HTML Report"""

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
        subplot_titles=["Thermal Output vs Irradiance", "IAM vs Incidence Angle",
                        "Efficiency vs (T_in - T_amb)/G", "Outlet Temp vs Irradiance"],
        vertical_spacing=0.12)

    base = {"incidence_angle_deg": 0.0, "T_ambient_degC": 20.0}

    # Thermal output vs irradiance at different angles
    G = np.linspace(100, 1200, 50)
    for theta in [0, 20, 40, 60]:
        r = model.predict({**base, "irradiance_w_m2": G, "incidence_angle_deg": theta,
                           "T_inlet_degC": 40.0})
        fig.add_trace(go.Scatter(x=G, y=r["thermal_output_w"],
                                  name=f"theta={theta}"), row=1, col=1)

    # IAM curve
    theta = np.linspace(0, 80, 80)
    iam = model._model.iam(theta)
    fig.add_trace(go.Scatter(x=theta, y=iam, name="IAM", showlegend=False), row=1, col=2)

    # Efficiency curve (classic Hottel-Whillier plot)
    T_in = np.linspace(20, 80, 50)
    G_fixed = 800.0
    x_param = (T_in - 20.0) / G_fixed
    r = model.predict({"irradiance_w_m2": G_fixed, "incidence_angle_deg": 0.0,
                        "T_inlet_degC": T_in, "T_ambient_degC": 20.0})
    fig.add_trace(go.Scatter(x=x_param, y=r["efficiency"], name="eta(T*)",
                              showlegend=False), row=2, col=1)

    # Outlet temperature vs irradiance
    G = np.linspace(100, 1200, 50)
    for T_in_val in [30, 50, 70]:
        r = model.predict({"irradiance_w_m2": G, "incidence_angle_deg": 0.0,
                           "T_inlet_degC": T_in_val, "T_ambient_degC": 20.0})
        fig.add_trace(go.Scatter(x=G, y=r["T_outlet_degC"],
                                  name=f"T_in={T_in_val}C", showlegend=False), row=2, col=2)

    fig.update_xaxes(title_text="Irradiance (W/m2)", row=1, col=1)
    fig.update_xaxes(title_text="Incidence Angle (deg)", row=1, col=2)
    fig.update_xaxes(title_text="(T_in - T_amb) / G (m2K/W)", row=2, col=1)
    fig.update_xaxes(title_text="Irradiance (W/m2)", row=2, col=2)
    fig.update_yaxes(title_text="Q (W)", row=1, col=1)
    fig.update_yaxes(title_text="IAM", row=1, col=2)
    fig.update_yaxes(title_text="Efficiency", row=2, col=1)
    fig.update_yaxes(title_text="T_out (C)", row=2, col=2)

    fig.update_layout(title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} IAM",
                      height=800, template="plotly_white")

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
