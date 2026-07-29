"""EC054 — Parabolic Trough CSP — F1a — Simulation & HTML Report"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Incidence Angle Modifier (IAM) vs Theta",
            "Heat Loss vs Absorber Temperature",
            "Useful Heat & Overall Efficiency vs DNI",
            "Overall Efficiency Map (T_abs vs DNI Heatmap)",
        ],
        vertical_spacing=0.14,
    )

    # Panel 1: IAM vs theta
    thetas = np.linspace(0, 80, 200)
    r_iam = model.predict({"dni": 1000.0, "T_absorber": 100.0, "T_ambient": 25.0,
                           "incidence_angle": thetas})
    iam = np.asarray(r_iam["optical_efficiency"]) / 0.75  # normalize by peak eta_opt
    fig.add_trace(go.Scatter(x=thetas, y=iam, name="IAM = 1 - 0.0003*theta^2",
                             line=dict(color="orange")), row=1, col=1)
    fig.add_hline(y=1.0, line_dash="dash", line_color="grey", row=1, col=1)

    # Panel 2: Heat loss vs T_absorber at several T_amb
    T_abs = np.linspace(100, 400, 200)
    for T_amb in [0.0, 20.0, 40.0]:
        r = model.predict({"dni": 0.0, "T_absorber": T_abs, "T_ambient": T_amb,
                           "incidence_angle": 0.0})
        fig.add_trace(go.Scatter(x=T_abs, y=r["thermal_loss_kw"],
                                 name=f"Q_loss T_amb={T_amb}C"), row=1, col=2)

    # Panel 3: Q_useful and eta_overall vs DNI at T_abs=300C, theta=15deg
    dnis = np.linspace(0, 1000, 200)
    for T_abs_val in [200.0, 300.0, 350.0]:
        r = model.predict({"dni": dnis, "T_absorber": T_abs_val, "T_ambient": 25.0,
                           "incidence_angle": 15.0})
        fig.add_trace(go.Scatter(x=dnis, y=r["useful_heat_kw"],
                                 name=f"Q_useful T={T_abs_val}C"), row=2, col=1)

    # Panel 4: Heatmap of overall efficiency vs DNI and T_abs
    dni_g = np.linspace(100, 1000, 50)
    T_abs_g = np.linspace(100, 400, 50)
    eta_map = np.zeros((50, 50))
    for i, T_a in enumerate(T_abs_g):
        r = model.predict({"dni": dni_g, "T_absorber": T_a, "T_ambient": 25.0,
                           "incidence_angle": 0.0})
        eta_map[i, :] = np.asarray(r["overall_efficiency"])
    fig.add_trace(go.Heatmap(
        x=dni_g, y=T_abs_g, z=eta_map,
        colorscale="RdYlGn", colorbar=dict(title="eta_overall", x=1.02),
        name="Overall Efficiency"), row=2, col=2)

    fig.update_xaxes(title_text="Incidence Angle (deg)", row=1, col=1)
    fig.update_xaxes(title_text="T_absorber (degC)", row=1, col=2)
    fig.update_xaxes(title_text="DNI (W/m2)", row=2, col=1)
    fig.update_xaxes(title_text="DNI (W/m2)", row=2, col=2)
    fig.update_yaxes(title_text="IAM (-)", row=1, col=1)
    fig.update_yaxes(title_text="Heat Loss (kW)", row=1, col=2)
    fig.update_yaxes(title_text="Useful Heat (kW)", row=2, col=1)
    fig.update_yaxes(title_text="T_absorber (degC)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} Optical + Heat Loss",
        height=850,
        template="plotly_white",
        legend=dict(orientation="v", x=1.08, y=0.95),
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
