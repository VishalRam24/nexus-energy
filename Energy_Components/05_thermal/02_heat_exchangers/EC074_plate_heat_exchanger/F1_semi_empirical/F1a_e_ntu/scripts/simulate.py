"""EC074 — Plate Heat Exchanger — F1a e-NTU — Simulation & HTML Report"""
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
            "Heat Transfer Rate vs m_dot_hot (T_h=80, T_c=20, equal flows)",
            "Effectiveness vs m_dot (T_h=80, T_c=20)",
            "Outlet Temperatures vs m_dot_hot (m_dot_cold=1 kg/s)",
            "Effectiveness Map — m_dot_hot vs m_dot_cold",
        ],
        vertical_spacing=0.13,
    )

    m_vals = np.linspace(0.1, 4.0, 100)

    # Panel 1: Q vs m_dot (equal flow rates)
    for dT, col in [(30, "blue"), (50, "orange"), (70, "green")]:
        r = model.predict({"T_h_in": 20.0 + dT, "T_c_in": 20.0,
                           "m_dot_hot": m_vals, "m_dot_cold": m_vals})
        fig.add_trace(go.Scatter(x=m_vals, y=r["Q_kw"],
                                 name=f"ΔT={dT}°C", line=dict(color=col)), row=1, col=1)

    # Panel 2: Effectiveness vs m_dot
    for ratio, col in [(0.5, "blue"), (1.0, "orange"), (2.0, "green")]:
        m_cold = m_vals * ratio
        r = model.predict({"T_h_in": 80.0, "T_c_in": 20.0,
                           "m_dot_hot": m_vals, "m_dot_cold": m_cold})
        fig.add_trace(go.Scatter(x=m_vals, y=r["effectiveness"],
                                 name=f"m_cold/m_hot={ratio}"), row=1, col=2)

    # Panel 3: Outlet temps vs m_dot_hot (m_cold fixed)
    r = model.predict({"T_h_in": 80.0, "T_c_in": 20.0,
                       "m_dot_hot": m_vals, "m_dot_cold": 1.0})
    fig.add_trace(go.Scatter(x=m_vals, y=r["T_h_out"], name="T_h_out", line=dict(color="red")), row=2, col=1)
    fig.add_trace(go.Scatter(x=m_vals, y=r["T_c_out"], name="T_c_out", line=dict(color="blue")), row=2, col=1)

    # Panel 4: Effectiveness heatmap
    m_h_grid = np.linspace(0.1, 4.0, 50)
    m_c_grid = np.linspace(0.1, 4.0, 50)
    eps_map  = np.zeros((len(m_c_grid), len(m_h_grid)))
    for i, mc in enumerate(m_c_grid):
        r = model.predict({"T_h_in": 80.0, "T_c_in": 20.0,
                           "m_dot_hot": m_h_grid, "m_dot_cold": mc})
        eps_map[i, :] = r["effectiveness"]
    fig.add_trace(go.Heatmap(x=m_h_grid, y=m_c_grid, z=eps_map,
                             colorscale="RdYlGn", colorbar=dict(title="ε"),
                             name="Effectiveness"), row=2, col=2)

    fig.update_xaxes(title_text="m_dot_hot (kg/s)", row=1, col=1)
    fig.update_xaxes(title_text="m_dot_hot (kg/s)", row=1, col=2)
    fig.update_xaxes(title_text="m_dot_hot (kg/s)", row=2, col=1)
    fig.update_xaxes(title_text="m_dot_hot (kg/s)", row=2, col=2)
    fig.update_yaxes(title_text="Q (kW)", row=1, col=1)
    fig.update_yaxes(title_text="Effectiveness", row=1, col=2)
    fig.update_yaxes(title_text="Temperature (°C)", row=2, col=1)
    fig.update_yaxes(title_text="m_dot_cold (kg/s)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}",
        height=800,
        template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
