"""EC122 — Pumped Hydro Storage — F1b Head Variation — Simulation & HTML Report"""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
from model import PHSF1b
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Generation Power vs Flow Rate at Different SOC",
            "Friction Head Loss vs Flow Rate",
            "Efficiency vs Flow Rate (Discharge & Charge)",
            "Round-Trip Efficiency vs SOC at Different Flow Rates",
        ],
        vertical_spacing=0.14,
    )

    Q_arr = np.linspace(1, 100, 200)
    soc_arr = np.linspace(0.0, 1.0, 200)
    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]

    # Panel 1: Generation power vs flow rate at different SOC
    for i, soc in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
        r = model.predict({"SOC": soc, "flow_rate_m3s": Q_arr, "mode": "discharge"})
        fig.add_trace(go.Scatter(
            x=Q_arr, y=r["power_kw"] / 1000.0,
            name=f"SOC={soc:.2f}", line=dict(color=colors[i], width=2),
        ), row=1, col=1)

    # Panel 2: Friction head loss vs flow rate (quadratic)
    with open(Path(__file__).parent.parent / "data" / "parameters.json") as f:
        params = json.load(f)
    m = PHSF1b(params)
    h_f = m.friction_loss(Q_arr)
    fig.add_trace(go.Scatter(
        x=Q_arr, y=h_f,
        name="h_f (Darcy-Weisbach)", line=dict(color="#636EFA", width=2.5),
        fill="tozeroy", fillcolor="rgba(99,110,250,0.1)",
    ), row=1, col=2)
    # Add percentage of head
    fig.add_trace(go.Scatter(
        x=Q_arr, y=h_f / 490.0 * 100,  # mid-head ~490m
        name="h_f as % of head", line=dict(color="#EF553B", width=2, dash="dash"),
        yaxis="y2",
    ), row=1, col=2)

    # Panel 3: Efficiency vs flow rate for both modes
    for mode, ls in [("discharge", "solid"), ("charge", "dash")]:
        eta = m.efficiency(0.5, Q_arr, mode)
        fig.add_trace(go.Scatter(
            x=Q_arr, y=eta * 100,
            name=f"eta_{mode}", line=dict(dash=ls, width=2),
        ), row=2, col=1)

    # Panel 4: Round-trip efficiency vs SOC at different flow rates
    for i, Q in enumerate([10, 25, 50, 75, 100]):
        rte = m.round_trip_efficiency(soc_arr, Q)
        fig.add_trace(go.Scatter(
            x=soc_arr, y=rte * 100,
            name=f"Q={Q} m3/s", line=dict(color=colors[i], width=2),
        ), row=2, col=2)

    fig.update_xaxes(title_text="Flow Rate (m3/s)", row=1, col=1)
    fig.update_xaxes(title_text="Flow Rate (m3/s)", row=1, col=2)
    fig.update_xaxes(title_text="Flow Rate (m3/s)", row=2, col=1)
    fig.update_xaxes(title_text="SOC", row=2, col=2)
    fig.update_yaxes(title_text="Power (MW)", row=1, col=1)
    fig.update_yaxes(title_text="Friction Loss (m)", row=1, col=2)
    fig.update_yaxes(title_text="Efficiency (%)", row=2, col=1)
    fig.update_yaxes(title_text="Round-Trip Efficiency (%)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}",
        height=850,
        template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
