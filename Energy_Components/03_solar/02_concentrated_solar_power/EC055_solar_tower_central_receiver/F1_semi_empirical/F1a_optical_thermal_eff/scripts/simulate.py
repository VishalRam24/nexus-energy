"""EC055 — Solar Tower CSP — F1a — Simulation & HTML Report"""

import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()

    fig = make_subplots(rows=2, cols=2,
        subplot_titles=["Field Optical Efficiency vs Solar Zenith",
                        "Receiver Thermal Loss vs T_receiver",
                        "Useful Heat vs DNI (several T_recv)",
                        "Overall Efficiency Map (DNI vs Zenith)"],
        vertical_spacing=0.13)

    # Panel 1
    z = np.linspace(0, 80, 200)
    r = model.predict({"dni": 1000.0, "solar_zenith": z,
                       "T_receiver": 565.0, "T_ambient": 25.0})
    fig.add_trace(go.Scatter(x=z, y=r["optical_efficiency"],
                             name="eta_optical"), row=1, col=1)

    # Panel 2
    T = np.linspace(300, 800, 200)
    for T_amb in [0.0, 25.0, 45.0]:
        r = model.predict({"dni": 0.0, "solar_zenith": 0.0,
                           "T_receiver": T, "T_ambient": T_amb})
        fig.add_trace(go.Scatter(x=T, y=r["thermal_loss_kw"],
                                 name=f"Q_loss T_amb={T_amb}C"), row=1, col=2)

    # Panel 3
    dni = np.linspace(0, 1100, 200)
    for Tr in [450.0, 565.0, 700.0]:
        r = model.predict({"dni": dni, "solar_zenith": 20.0,
                           "T_receiver": Tr, "T_ambient": 25.0})
        fig.add_trace(go.Scatter(x=dni, y=r["useful_heat_kw"],
                                 name=f"Q_useful T={Tr}C"), row=2, col=1)

    # Panel 4: heatmap eta_overall vs DNI and zenith
    dni_g = np.linspace(100, 1000, 50)
    z_g = np.linspace(0, 80, 50)
    eta_map = np.zeros((50, 50))
    for i, zi in enumerate(z_g):
        r = model.predict({"dni": dni_g, "solar_zenith": zi,
                           "T_receiver": 565.0, "T_ambient": 25.0})
        eta_map[i, :] = np.asarray(r["overall_efficiency"])
    fig.add_trace(go.Heatmap(x=dni_g, y=z_g, z=eta_map,
                             colorscale="RdYlGn",
                             colorbar=dict(title="eta_overall", x=1.02)), row=2, col=2)

    fig.update_xaxes(title_text="Solar Zenith (deg)", row=1, col=1)
    fig.update_xaxes(title_text="T_receiver (degC)", row=1, col=2)
    fig.update_xaxes(title_text="DNI (W/m2)", row=2, col=1)
    fig.update_xaxes(title_text="DNI (W/m2)", row=2, col=2)
    fig.update_yaxes(title_text="eta_optical (-)", row=1, col=1)
    fig.update_yaxes(title_text="Heat Loss (kW)", row=1, col=2)
    fig.update_yaxes(title_text="Useful Heat (kW)", row=2, col=1)
    fig.update_yaxes(title_text="Solar Zenith (deg)", row=2, col=2)

    fig.update_layout(title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}",
                      height=850, template="plotly_white")
    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
