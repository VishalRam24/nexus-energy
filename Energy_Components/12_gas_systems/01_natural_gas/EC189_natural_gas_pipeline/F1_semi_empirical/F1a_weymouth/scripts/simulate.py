"""EC189 — Natural Gas Pipeline — F1a — Simulation & HTML Report"""
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
            "Flow Rate vs Outlet Pressure (Weymouth)",
            "Flow Rate vs Pipeline Length",
            "Q scaling: Q vs sqrt(P_in²-P_out²)",
            "Weymouth Friction Factor vs Diameter",
        ],
        vertical_spacing=0.14, horizontal_spacing=0.1,
    )

    P_out = np.linspace(10, 74, 200)
    D_vals = [0.4064, 0.6096, 0.9144]  # 16", 24", 36" pipes
    colors = ["#1f77b4", "#ff7f0e", "#d62728"]

    for D, c in zip(D_vals, colors):
        r = model.predict({"length_km": 100.0, "diameter_m": D,
                           "P_in_bar": 75.0, "P_out_bar": P_out})
        lbl = f"D={D*1000:.0f} mm"
        fig.add_trace(go.Scatter(x=P_out, y=r["Q_std_m3_per_day"] / 1e6, name=lbl,
                                  line=dict(color=c)), row=1, col=1)

    L_arr = np.linspace(10, 500, 200)
    for D, c in zip(D_vals, colors):
        r = model.predict({"length_km": L_arr, "diameter_m": D,
                           "P_in_bar": 75.0, "P_out_bar": 50.0})
        lbl = f"D={D*1000:.0f} mm"
        fig.add_trace(go.Scatter(x=L_arr, y=r["Q_std_m3_per_day"] / 1e6, name=lbl,
                                  line=dict(color=c), showlegend=False), row=1, col=2)

    # Verify Q ∝ sqrt(P1²-P2²)
    P1_arr = np.full(50, 75.0)
    P2_arr = np.linspace(10, 74, 50)
    sqrt_dp2 = np.sqrt(P1_arr ** 2 - P2_arr ** 2)
    r = model.predict({"length_km": 100.0, "diameter_m": 0.6096,
                       "P_in_bar": P1_arr, "P_out_bar": P2_arr})
    fig.add_trace(go.Scatter(x=sqrt_dp2, y=r["Q_std_m3_per_day"] / 1e6,
                              mode="markers", name="Q vs √(P1²-P2²)",
                              marker=dict(color="#2ca02c")), row=2, col=1)

    D_arr = np.linspace(0.1, 1.2, 200)
    r_D = model.predict({"length_km": 100.0, "diameter_m": D_arr,
                          "P_in_bar": 75.0, "P_out_bar": 50.0})
    fig.add_trace(go.Scatter(x=D_arr * 1000, y=r_D["weymouth_f"],
                              name="f_Weymouth", line=dict(color="#9467bd")), row=2, col=2)

    fig.update_xaxes(title_text="P_out (bar)", row=1, col=1)
    fig.update_xaxes(title_text="Length (km)", row=1, col=2)
    fig.update_xaxes(title_text="√(P1²-P2²) (bar)", row=2, col=1)
    fig.update_xaxes(title_text="Diameter (mm)", row=2, col=2)
    fig.update_yaxes(title_text="Q (Mm³/day)", row=1, col=1)
    fig.update_yaxes(title_text="Q (Mm³/day)", row=1, col=2)
    fig.update_yaxes(title_text="Q (Mm³/day)", row=2, col=1)
    fig.update_yaxes(title_text="f_Weymouth [-]", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} Weymouth Model<br>"
              f"<sup>G=0.6, Z=0.9, E=0.92 | {info['source']}</sup>",
        height=850, template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
