"""EC059 — Evacuated Tube Solar Collector — F1a — Simulation & HTML Report"""
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
            "ETC vs Flat-Plate Comparison (HWB curve)",
            "Useful Heat vs Irradiance",
            "Outlet Temperature vs T_inlet at G=800 W/m2",
        ],
        vertical_spacing=0.14,
    )

    # Panel 1: eta vs G
    G = np.linspace(50, 1200, 150)
    T_amb = 20.0
    for T_in in [20.0, 40.0, 60.0, 80.0, 100.0]:
        r = model.predict({"irradiance": G, "T_inlet": T_in, "T_ambient": T_amb})
        fig.add_trace(go.Scatter(x=G, y=r["efficiency"], name=f"T_in={T_in}C"), row=1, col=1)

    # Panel 2: ETC vs flat plate HWB curves
    X = np.linspace(0, 0.20, 200)
    eta_etc = np.clip(0.72 - 1.4 * X, 0, 0.72)
    eta_fpc = np.clip(0.75 - 4.5 * X, 0, 0.75)
    fig.add_trace(go.Scatter(x=X, y=eta_etc, name="ETC (F_R*U_L=1.4)",
                             line=dict(color="blue")), row=1, col=2)
    fig.add_trace(go.Scatter(x=X, y=eta_fpc, name="Flat-plate (F_R*U_L=4.5)",
                             line=dict(color="red", dash="dash")), row=1, col=2)

    # Panel 3: Q_u vs G
    G = np.linspace(0, 1200, 200)
    for T_in in [40.0, 70.0, 100.0]:
        r = model.predict({"irradiance": G, "T_inlet": T_in, "T_ambient": T_amb})
        fig.add_trace(go.Scatter(x=G, y=r["useful_heat_w"],
                                 name=f"Q_u T_in={T_in}C"), row=2, col=1)

    # Panel 4: T_out vs T_in
    T_in_arr = np.linspace(10, 110, 150)
    for G_val in [400.0, 800.0, 1200.0]:
        r = model.predict({"irradiance": G_val, "T_inlet": T_in_arr, "T_ambient": T_amb})
        fig.add_trace(go.Scatter(x=T_in_arr, y=r["T_outlet_approx"],
                                 name=f"T_out G={G_val}W/m2"), row=2, col=2)
    fig.add_trace(go.Scatter(x=T_in_arr, y=T_in_arr, name="T_out=T_in",
                             line=dict(dash="dash", color="grey")), row=2, col=2)

    fig.update_xaxes(title_text="Irradiance (W/m2)", row=1, col=1)
    fig.update_xaxes(title_text="(T_in - T_amb)/G  (K/(W/m2))", row=1, col=2)
    fig.update_xaxes(title_text="Irradiance (W/m2)", row=2, col=1)
    fig.update_xaxes(title_text="T_inlet (degC)", row=2, col=2)
    fig.update_yaxes(title_text="Efficiency (-)", row=1, col=1)
    fig.update_yaxes(title_text="Efficiency (-)", row=1, col=2)
    fig.update_yaxes(title_text="Useful Heat (W)", row=2, col=1)
    fig.update_yaxes(title_text="T_outlet (degC)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} HWB (vacuum-insulated)",
        height=850, template="plotly_white",
        legend=dict(orientation="v", x=1.08, y=0.95),
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
