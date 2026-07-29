"""EC073 — Shell-and-Tube HX — F1a LMTD/NTU — Simulation & HTML Report"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info  = model.get_info()

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Heat duty vs m_dot (T_h=90, T_c=25)",
            "Effectiveness vs m_dot (various m_cold/m_hot)",
            "Outlet temperatures vs m_dot_hot",
            "Effectiveness Map — m_dot_hot vs m_dot_cold",
        ],
        vertical_spacing=0.13,
    )

    m = np.linspace(0.5, 20.0, 100)

    for dT, col in [(40, "blue"), (60, "orange"), (90, "green")]:
        r = model.predict({"T_h_in": 25.0 + dT, "T_c_in": 25.0,
                           "m_dot_hot": m, "m_dot_cold": m})
        fig.add_trace(go.Scatter(x=m, y=r["Q_kw"], name=f"ΔT={dT}°C", line=dict(color=col)), row=1, col=1)

    for ratio in [0.5, 1.0, 2.0]:
        r = model.predict({"T_h_in": 90.0, "T_c_in": 25.0,
                           "m_dot_hot": m, "m_dot_cold": m * ratio})
        fig.add_trace(go.Scatter(x=m, y=r["effectiveness"], name=f"m_c/m_h={ratio}"), row=1, col=2)

    r = model.predict({"T_h_in": 90.0, "T_c_in": 25.0, "m_dot_hot": m, "m_dot_cold": 5.0})
    fig.add_trace(go.Scatter(x=m, y=r["T_h_out"], name="T_h_out", line=dict(color="red")), row=2, col=1)
    fig.add_trace(go.Scatter(x=m, y=r["T_c_out"], name="T_c_out", line=dict(color="blue")), row=2, col=1)

    m_h_grid = np.linspace(0.5, 20.0, 50)
    m_c_grid = np.linspace(0.5, 20.0, 50)
    eps_map  = np.zeros((50, 50))
    for i, mc in enumerate(m_c_grid):
        r = model.predict({"T_h_in": 90.0, "T_c_in": 25.0, "m_dot_hot": m_h_grid, "m_dot_cold": mc})
        eps_map[i, :] = r["effectiveness"]
    fig.add_trace(go.Heatmap(x=m_h_grid, y=m_c_grid, z=eps_map, colorscale="RdYlGn",
                             colorbar=dict(title="ε")), row=2, col=2)

    fig.update_xaxes(title_text="m_dot (kg/s)", row=1, col=1)
    fig.update_xaxes(title_text="m_dot_hot (kg/s)", row=1, col=2)
    fig.update_xaxes(title_text="m_dot_hot (kg/s)", row=2, col=1)
    fig.update_xaxes(title_text="m_dot_hot (kg/s)", row=2, col=2)
    fig.update_yaxes(title_text="Q (kW)", row=1, col=1)
    fig.update_yaxes(title_text="Effectiveness", row=1, col=2)
    fig.update_yaxes(title_text="Temperature (°C)", row=2, col=1)
    fig.update_yaxes(title_text="m_dot_cold (kg/s)", row=2, col=2)

    fig.update_layout(title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}",
                      height=800, template="plotly_white")
    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
