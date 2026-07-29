"""EC058 — Flat Plate Solar Collector — F1a — Simulation & HTML Report"""
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
            "Efficiency vs Irradiance (several T_in)",
            "Efficiency vs (T_in - T_amb)/G — HWB Characteristic",
            "Useful Heat vs Irradiance",
            "Outlet Temperature vs T_inlet at G=800 W/m2",
        ],
        vertical_spacing=0.14,
    )

    # Panel 1: eta vs G
    G = np.linspace(50, 1200, 150)
    T_amb = 20.0
    for T_in in [20.0, 35.0, 50.0, 65.0, 80.0]:
        r = model.predict({"irradiance": G, "T_inlet": T_in, "T_ambient": T_amb})
        eta = np.asarray(r["efficiency"])
        fig.add_trace(go.Scatter(x=G, y=eta, name=f"T_in={T_in}C"), row=1, col=1)

    # Panel 2: HWB characteristic eta vs X = (T_in - T_amb)/G
    X_vals = np.linspace(0, 0.12, 200)
    # eta = FR_tau_alpha - FR_UL * X
    FR_ta = 0.75
    FR_UL = 4.5 / 1000  # W/m2K -> kW/m2K -> same units since X is in K/(W/m2)
    FR_UL_WpM2K = 4.5
    eta_theory = np.clip(FR_ta - FR_UL_WpM2K * X_vals, 0, FR_ta)
    fig.add_trace(go.Scatter(x=X_vals, y=eta_theory, name="HWB Linear",
                             line=dict(color="blue")), row=1, col=2)
    # Overlay sampled points
    G_sample = np.array([400.0, 600.0, 800.0, 1000.0])
    T_in_s = np.array([20.0, 40.0, 60.0, 80.0])
    for Ti in T_in_s:
        r = model.predict({"irradiance": G_sample, "T_inlet": Ti, "T_ambient": T_amb})
        X_s = (Ti - T_amb) / G_sample
        fig.add_trace(go.Scatter(x=X_s, y=np.asarray(r["efficiency"]),
                                 mode="markers", marker=dict(size=6),
                                 name=f"T_in={Ti}C pts", showlegend=False), row=1, col=2)

    # Panel 3: Q_u vs G
    G = np.linspace(0, 1200, 200)
    for T_in in [20.0, 40.0, 60.0]:
        r = model.predict({"irradiance": G, "T_inlet": T_in, "T_ambient": T_amb})
        fig.add_trace(go.Scatter(x=G, y=r["useful_heat_w"],
                                 name=f"Q_u T_in={T_in}C"), row=2, col=1)

    # Panel 4: T_out vs T_in at G=800
    T_ins = np.linspace(10, 80, 150)
    for G_val in [400.0, 800.0, 1200.0]:
        r = model.predict({"irradiance": G_val, "T_inlet": T_ins, "T_ambient": T_amb})
        fig.add_trace(go.Scatter(x=T_ins, y=r["T_outlet_approx"],
                                 name=f"T_out G={G_val}W/m2"), row=2, col=2)
    fig.add_trace(go.Scatter(x=T_ins, y=T_ins, name="T_out=T_in line",
                             line=dict(dash="dash", color="grey")), row=2, col=2)

    fig.update_xaxes(title_text="Irradiance (W/m2)", row=1, col=1)
    fig.update_xaxes(title_text="(T_in - T_amb) / G (K/(W/m2))", row=1, col=2)
    fig.update_xaxes(title_text="Irradiance (W/m2)", row=2, col=1)
    fig.update_xaxes(title_text="T_inlet (degC)", row=2, col=2)
    fig.update_yaxes(title_text="Efficiency (-)", row=1, col=1)
    fig.update_yaxes(title_text="Efficiency (-)", row=1, col=2)
    fig.update_yaxes(title_text="Useful Heat (W)", row=2, col=1)
    fig.update_yaxes(title_text="T_outlet (degC)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} Hottel-Whillier-Bliss",
        height=850,
        template="plotly_white",
        legend=dict(orientation="v", x=1.08, y=0.95),
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
