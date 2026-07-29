"""EC054 — Parabolic Trough CSP — F1b Receiver Losses — Simulation & HTML Report"""

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
        subplot_titles=["Thermal Output vs DNI", "Receiver Loss vs Absorber Temp",
                        "IAM & End Loss vs Incidence Angle", "Efficiency vs DNI"],
        vertical_spacing=0.12)

    base = {"T_htf_in_degC": 290.0, "T_htf_out_degC": 390.0,
            "T_ambient_degC": 25.0, "incidence_angle_deg": 10.0}

    # Thermal output vs DNI
    G = np.linspace(100, 1000, 50)
    for theta in [0, 10, 30, 50]:
        inp = {**base, "DNI_w_m2": G, "incidence_angle_deg": theta}
        r = model.predict(inp)
        fig.add_trace(go.Scatter(x=G, y=r["thermal_output_kw_per_m"],
                                  name=f"theta={theta}"), row=1, col=1)

    # Receiver loss vs absorber temp
    T_abs = np.linspace(100, 400, 50)
    q_loss = model._model.receiver_loss_kw_per_m(T_abs, 25.0)
    fig.add_trace(go.Scatter(x=T_abs, y=q_loss, name="Q_loss",
                              showlegend=False), row=1, col=2)

    # IAM and end loss
    theta = np.linspace(0, 80, 80)
    iam = model._model.iam(theta)
    f_end = model._model.end_loss_factor(theta)
    fig.add_trace(go.Scatter(x=theta, y=iam, name="IAM", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=theta, y=f_end, name="End Loss", showlegend=False), row=2, col=1)

    # Efficiency vs DNI
    G = np.linspace(100, 1000, 50)
    inp = {**base, "DNI_w_m2": G}
    r = model.predict(inp)
    fig.add_trace(go.Scatter(x=G, y=r["optical_efficiency"], name="Optical",
                              showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=G, y=r["thermal_efficiency"], name="Thermal",
                              showlegend=False), row=2, col=2)

    fig.update_xaxes(title_text="DNI (W/m2)", row=1, col=1)
    fig.update_xaxes(title_text="Absorber Temp (C)", row=1, col=2)
    fig.update_xaxes(title_text="Incidence Angle (deg)", row=2, col=1)
    fig.update_xaxes(title_text="DNI (W/m2)", row=2, col=2)
    fig.update_yaxes(title_text="kW/m", row=1, col=1)
    fig.update_yaxes(title_text="kW/m", row=1, col=2)
    fig.update_yaxes(title_text="Factor", row=2, col=1)
    fig.update_yaxes(title_text="Efficiency", row=2, col=2)

    fig.update_layout(title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} Receiver Losses",
                      height=800, template="plotly_white")

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
