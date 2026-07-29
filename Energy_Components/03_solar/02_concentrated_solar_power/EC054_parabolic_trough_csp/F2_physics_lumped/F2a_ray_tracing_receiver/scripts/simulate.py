"""EC054 -- Parabolic Trough CSP -- F2a -- Simulation & HTML Report"""
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
        rows=3, cols=2,
        subplot_titles=[
            "Thermal Efficiency vs DNI (various T_in)",
            "Q_useful & Q_loss vs DNI",
            "HTF Outlet Temperature vs DNI",
            "Efficiency vs Incidence Angle",
            "Absorber & Glass Temperature vs T_in",
            "Efficiency Map: T_in vs DNI",
        ],
        vertical_spacing=0.10,
    )

    # ---- Panel 1: eta_thermal vs DNI for various T_in ----
    dnis = np.linspace(200, 1100, 50)
    for T_in in [200.0, 250.0, 300.0, 350.0, 380.0]:
        etas = []
        for g in dnis:
            r = model.predict({"dni": float(g), "incidence_angle": 0.0,
                               "T_htf_in": T_in, "m_dot": 6.0})
            etas.append(r["eta_thermal"])
        fig.add_trace(go.Scatter(x=dnis, y=etas,
                                 name=f"T_in={T_in:.0f}C"), row=1, col=1)

    # ---- Panel 2: Q_useful and Q_loss vs DNI ----
    q_use, q_loss = [], []
    for g in dnis:
        r = model.predict({"dni": float(g), "incidence_angle": 0.0,
                           "T_htf_in": 300.0, "m_dot": 6.0})
        q_use.append(r["Q_useful_W"] / 1000)
        q_loss.append(r["Q_loss_W"] / 1000)
    fig.add_trace(go.Scatter(x=dnis, y=q_use, name="Q_useful (kW)",
                             line=dict(color="green")), row=1, col=2)
    fig.add_trace(go.Scatter(x=dnis, y=q_loss, name="Q_loss (kW)",
                             line=dict(color="red", dash="dash")), row=1, col=2)

    # ---- Panel 3: T_out vs DNI ----
    for m_dot in [3.0, 6.0, 9.0]:
        t_outs = []
        for g in dnis:
            r = model.predict({"dni": float(g), "incidence_angle": 0.0,
                               "T_htf_in": 300.0, "m_dot": m_dot})
            t_outs.append(r["T_htf_out_C"])
        fig.add_trace(go.Scatter(x=dnis, y=t_outs,
                                 name=f"m_dot={m_dot} kg/s"), row=2, col=1)

    # ---- Panel 4: Efficiency vs incidence angle ----
    thetas = np.linspace(0, 70, 40)
    eta_opt_arr, eta_th_arr = [], []
    for th in thetas:
        r = model.predict({"dni": 900.0, "incidence_angle": float(th),
                           "T_htf_in": 300.0, "m_dot": 6.0})
        eta_opt_arr.append(r["eta_optical"])
        eta_th_arr.append(r["eta_thermal"])
    fig.add_trace(go.Scatter(x=thetas, y=eta_opt_arr, name="eta_optical",
                             line=dict(color="orange")), row=2, col=2)
    fig.add_trace(go.Scatter(x=thetas, y=eta_th_arr, name="eta_thermal",
                             line=dict(color="blue")), row=2, col=2)

    # ---- Panel 5: Absorber & glass temp vs T_in ----
    T_ins = np.linspace(150, 390, 30)
    t_abs_arr, t_glass_arr = [], []
    for tin in T_ins:
        r = model.predict({"dni": 900.0, "incidence_angle": 0.0,
                           "T_htf_in": float(tin), "m_dot": 6.0})
        t_abs_arr.append(r["T_abs_C"])
        t_glass_arr.append(r["T_glass_C"])
    fig.add_trace(go.Scatter(x=T_ins, y=t_abs_arr, name="T_absorber",
                             line=dict(color="red")), row=3, col=1)
    fig.add_trace(go.Scatter(x=T_ins, y=t_glass_arr, name="T_glass",
                             line=dict(color="skyblue")), row=3, col=1)
    fig.add_trace(go.Scatter(x=T_ins, y=T_ins.tolist(), name="T_in (ref)",
                             line=dict(color="grey", dash="dot")), row=3, col=1)

    # ---- Panel 6: Heatmap of eta_thermal vs T_in and DNI ----
    dni_g = np.linspace(300, 1100, 40)
    T_in_g = np.linspace(150, 390, 40)
    eta_map = np.zeros((len(T_in_g), len(dni_g)))
    for i, tin in enumerate(T_in_g):
        for j, g in enumerate(dni_g):
            r = model.predict({"dni": float(g), "incidence_angle": 0.0,
                               "T_htf_in": float(tin), "m_dot": 6.0})
            eta_map[i, j] = r["eta_thermal"]
    fig.add_trace(go.Heatmap(
        x=dni_g, y=T_in_g, z=eta_map,
        colorscale="RdYlGn", colorbar=dict(title="eta_th", x=1.02),
        name="eta_thermal"), row=3, col=2)

    # Axis labels
    fig.update_xaxes(title_text="DNI (W/m2)", row=1, col=1)
    fig.update_xaxes(title_text="DNI (W/m2)", row=1, col=2)
    fig.update_xaxes(title_text="DNI (W/m2)", row=2, col=1)
    fig.update_xaxes(title_text="Incidence Angle (deg)", row=2, col=2)
    fig.update_xaxes(title_text="T_htf_in (degC)", row=3, col=1)
    fig.update_xaxes(title_text="DNI (W/m2)", row=3, col=2)

    fig.update_yaxes(title_text="eta_thermal", row=1, col=1)
    fig.update_yaxes(title_text="Power (kW)", row=1, col=2)
    fig.update_yaxes(title_text="T_htf_out (degC)", row=2, col=1)
    fig.update_yaxes(title_text="Efficiency", row=2, col=2)
    fig.update_yaxes(title_text="Temperature (degC)", row=3, col=1)
    fig.update_yaxes(title_text="T_htf_in (degC)", row=3, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} -- {info['name']} -- F2a Lumped HCE Thermal Model",
        height=1100,
        template="plotly_white",
        legend=dict(orientation="v", x=1.08, y=0.95),
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
