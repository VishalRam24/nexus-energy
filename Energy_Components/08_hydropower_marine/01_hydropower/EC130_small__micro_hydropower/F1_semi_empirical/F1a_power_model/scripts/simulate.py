"""EC130 — Small/Micro Hydropower — F1a — Simulation & HTML Report"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
from model import TURBINE_TYPES
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()
    m = model._model

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Power vs Flow (design head, auto turbine)",
            "Turbine Efficiency vs q by Turbine Type",
            "Power vs Net Head at Design Flow",
            "Power Map (kW) — Flow vs Net Head",
        ],
        vertical_spacing=0.13,
    )

    # Panel 1: Power vs Flow
    flows = np.linspace(0.0, 1.65, 200)
    r = model.predict({"flow_rate_m3s": flows, "net_head_m": 40.0, "turbine_type": "auto"})
    fig.add_trace(go.Scatter(x=flows, y=r["power_kw"], name="Francis (auto, H=40m)",
                             line=dict(color="blue")), row=1, col=1)

    # Panel 2: Efficiency vs q for each turbine type
    q_vals = np.linspace(0.2, 1.15, 200)
    Q_vals = q_vals * m.Q_design
    colors = {"pelton": "green", "francis": "blue", "kaplan": "orange"}
    for t_name in ["pelton", "francis", "kaplan"]:
        t = TURBINE_TYPES[t_name]
        q = Q_vals / m.Q_design
        eta_t = t["eta_peak"] * (1.0 - t["k"] * (q - 1.0) ** 2)
        eta_t = np.where((q < m.q_min) | (q > m.q_max), 0.0, eta_t)
        eta_t = np.clip(eta_t, 0.0, t["eta_peak"])
        fig.add_trace(go.Scatter(x=q_vals, y=eta_t, name=t_name.capitalize(),
                                 line=dict(color=colors[t_name])), row=1, col=2)
    fig.add_vline(x=1.0, line_dash="dot", line_color="gray", row=1, col=2)

    # Panel 3: Power vs head (design flow, auto turbine)
    heads = np.array([5.0, 10.0, 20.0, 40.0, 80.0, 150.0, 300.0])
    powers = [float(model.predict({"flow_rate_m3s": m.Q_design, "net_head_m": H, "turbine_type": "auto"})["power_kw"])
              for H in heads]
    fig.add_trace(go.Scatter(x=heads, y=powers, mode="lines+markers",
                             name="Q=Q_design", line=dict(color="purple")), row=2, col=1)

    # Panel 4: Power map (Francis range)
    Q_grid = np.linspace(0.0, 1.65, 60)
    H_grid = np.linspace(2.0, 150.0, 60)
    power_map = np.zeros((len(H_grid), len(Q_grid)))
    for i, H in enumerate(H_grid):
        r = model.predict({"flow_rate_m3s": Q_grid, "net_head_m": H, "turbine_type": "auto"})
        power_map[i, :] = r["power_kw"]
    fig.add_trace(go.Heatmap(x=Q_grid, y=H_grid, z=power_map, colorscale="Greens",
                             colorbar=dict(title="kW"), name="Power (kW)"), row=2, col=2)

    fig.update_xaxes(title_text="Flow Rate (m³/s)", row=1, col=1)
    fig.update_xaxes(title_text="q = Q/Q_design", row=1, col=2)
    fig.update_xaxes(title_text="Net Head (m)", row=2, col=1)
    fig.update_xaxes(title_text="Flow Rate (m³/s)", row=2, col=2)
    fig.update_yaxes(title_text="Power (kW)", row=1, col=1)
    fig.update_yaxes(title_text="Turbine Efficiency", row=1, col=2)
    fig.update_yaxes(title_text="Power (kW)", row=2, col=1)
    fig.update_yaxes(title_text="Net Head (m)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}",
        height=800, template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
