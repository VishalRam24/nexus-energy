"""EC192 — Gas Pressure Regulator — F1a — Simulation & HTML Report"""
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
            "JT Temperature Drop vs Pressure Drop",
            "Gas Flow vs Downstream Pressure",
            "Expansion Factor Y vs ΔP/P_up",
            "Flow vs Cv (varying ΔP)",
        ],
        vertical_spacing=0.14, horizontal_spacing=0.1,
    )

    P_up = 80.0
    P_down_arr = np.linspace(2, 79, 200)
    T_up_vals = [270.0, 288.15, 310.0]
    colors = ["#1f77b4", "#ff7f0e", "#d62728"]

    for T_up, c in zip(T_up_vals, colors):
        r = model.predict({"P_up_bar": P_up, "P_down_bar": P_down_arr, "T_up_K": T_up})
        lbl = f"T_up={T_up:.0f} K"
        fig.add_trace(go.Scatter(x=P_up - P_down_arr, y=r["delta_T_K"],
                                  name=lbl, line=dict(color=c)), row=1, col=1)
        fig.add_trace(go.Scatter(x=P_down_arr, y=r["Q_std_m3_per_h"],
                                  name=lbl, line=dict(color=c), showlegend=False), row=1, col=2)

    r_Y = model.predict({"P_up_bar": P_up, "P_down_bar": P_down_arr, "T_up_K": 288.15})
    dp_ratio = (P_up - P_down_arr) / P_up
    fig.add_trace(go.Scatter(x=dp_ratio, y=r_Y["expansion_Y"],
                              name="Y factor", line=dict(color="#2ca02c")), row=2, col=1)
    # Mark choke threshold
    choke_x = 0.7 * 0.972  # Fk * Xt
    fig.add_vline(x=choke_x, line_dash="dash", line_color="red", row=2, col=1)

    Cv_arr = np.linspace(100, 2000, 100)
    for dP, c in zip([10.0, 30.0, 60.0], ["#1f77b4", "#ff7f0e", "#2ca02c"]):
        r = model.predict({"P_up_bar": P_up, "P_down_bar": P_up - dP,
                           "T_up_K": 288.15, "Cv": Cv_arr})
        fig.add_trace(go.Scatter(x=Cv_arr, y=r["Q_std_m3_per_h"],
                                  name=f"ΔP={dP} bar", line=dict(color=c)), row=2, col=2)

    fig.update_xaxes(title_text="ΔP (bar)", row=1, col=1)
    fig.update_xaxes(title_text="P_down (bar)", row=1, col=2)
    fig.update_xaxes(title_text="ΔP/P_up [-]", row=2, col=1)
    fig.update_xaxes(title_text="Cv [gal/min/psi^0.5]", row=2, col=2)
    fig.update_yaxes(title_text="ΔT (K)", row=1, col=1)
    fig.update_yaxes(title_text="Q (m³/h)", row=1, col=2)
    fig.update_yaxes(title_text="Y [-]", row=2, col=1)
    fig.update_yaxes(title_text="Q (m³/h)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}<br>"
              f"<sup>Isenthalpic throttle + JT cooling + ISA Cv | {info['source']}</sup>",
        height=850, template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
