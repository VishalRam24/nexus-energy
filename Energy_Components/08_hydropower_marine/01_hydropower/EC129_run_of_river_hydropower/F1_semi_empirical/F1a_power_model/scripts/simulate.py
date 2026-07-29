"""EC129 — Run-of-River Hydropower — F1a — Simulation & HTML Report"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()
    m = model._model

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Power vs Flow Rate (parametric gross head)",
            "Turbine Efficiency vs Flow Ratio (q = Q/Q_design)",
            "Power vs Gross Head (parametric flow)",
            "Power Map (kW) — Flow vs Gross Head",
        ],
        vertical_spacing=0.13,
    )

    # Panel 1: Power vs Flow for several heads
    flows = np.linspace(0.0, 57.5, 200)
    for H in [4.0, 6.0, 8.0, 12.0, 18.0]:
        r = model.predict({"flow_rate_m3s": flows, "gross_head_m": H})
        fig.add_trace(go.Scatter(x=flows, y=r["power_kw"], name=f"H={H:.0f} m"), row=1, col=1)

    # Panel 2: Efficiency vs normalised flow q
    q_vals = np.linspace(0.2, 1.2, 200)
    Q_vals = q_vals * m.Q_design
    r_eta = model.predict({"flow_rate_m3s": Q_vals, "gross_head_m": 8.0})
    fig.add_trace(go.Scatter(x=q_vals, y=r_eta["turbine_efficiency"], name="eta_turbine",
                             line=dict(color="blue")), row=1, col=2)
    fig.add_trace(go.Scatter(x=q_vals, y=r_eta["overall_efficiency"], name="eta_overall",
                             line=dict(color="orange", dash="dash")), row=1, col=2)
    fig.add_vline(x=1.0, line_dash="dot", line_color="gray", row=1, col=2)

    # Panel 3: Power vs head for several flows
    heads = np.linspace(2.0, 20.0, 200)
    for Q in [20.0, 30.0, 40.0, 50.0]:
        r = model.predict({"flow_rate_m3s": Q, "gross_head_m": heads})
        fig.add_trace(go.Scatter(x=heads, y=r["power_kw"], name=f"Q={Q:.0f} m³/s"), row=2, col=1)

    # Panel 4: Power map heatmap
    Q_grid = np.linspace(0.0, 57.5, 60)
    H_grid = np.linspace(2.0, 20.0, 60)
    power_map = np.zeros((len(H_grid), len(Q_grid)))
    for i, H in enumerate(H_grid):
        r = model.predict({"flow_rate_m3s": Q_grid, "gross_head_m": H})
        power_map[i, :] = r["power_kw"]
    fig.add_trace(go.Heatmap(x=Q_grid, y=H_grid, z=power_map, colorscale="Blues",
                             colorbar=dict(title="kW"), name="Power (kW)"), row=2, col=2)

    fig.update_xaxes(title_text="Flow Rate (m³/s)", row=1, col=1)
    fig.update_xaxes(title_text="q = Q/Q_design", row=1, col=2)
    fig.update_xaxes(title_text="Gross Head (m)", row=2, col=1)
    fig.update_xaxes(title_text="Flow Rate (m³/s)", row=2, col=2)
    fig.update_yaxes(title_text="Power (kW)", row=1, col=1)
    fig.update_yaxes(title_text="Efficiency", row=1, col=2)
    fig.update_yaxes(title_text="Power (kW)", row=2, col=1)
    fig.update_yaxes(title_text="Gross Head (m)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}",
        height=800, template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
