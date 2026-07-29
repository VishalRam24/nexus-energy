"""EC175 — Induction Motor/Generator — F1b Thermal — Simulation & HTML Report"""
import sys, json, numpy as np
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
            "Efficiency vs PLR at Different Winding Temperatures",
            "Losses vs PLR at Different Winding Temperatures",
            "Derating Factor vs Ambient Temperature",
            "Current vs PLR at Different Winding Temperatures",
        ],
        vertical_spacing=0.14,
    )

    plr = np.linspace(0.05, 1.2, 200)
    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]

    # Panel 1: Efficiency vs PLR at different winding temperatures
    for i, T_w in enumerate([25, 50, 75, 120, 155]):
        r = model.predict({"load_fraction": plr, "winding_temperature": T_w})
        fig.add_trace(go.Scatter(
            x=plr, y=r["efficiency"] * 100,
            name=f"T_w={T_w}C", line=dict(color=colors[i], width=2),
            legendgroup="temp",
        ), row=1, col=1)

    # Panel 2: Losses vs PLR
    for i, T_w in enumerate([25, 75, 120, 155]):
        r = model.predict({"load_fraction": plr, "winding_temperature": T_w})
        fig.add_trace(go.Scatter(
            x=plr, y=r["losses_kw"],
            name=f"Loss T_w={T_w}C", line=dict(color=colors[i], width=2),
            legendgroup="loss", showlegend=True,
        ), row=1, col=2)

    # Panel 3: Derating factor vs ambient temperature
    T_amb = np.linspace(-20, 60, 200)
    r_derate = model.predict({
        "load_fraction": 1.0,
        "ambient_temperature": T_amb,
        "winding_temperature": 75.0,
    })
    derate = r_derate["derating_factor"]
    # Handle scalar vs array
    if np.ndim(derate) == 0:
        # derating_factor is scalar because it only depends on ambient
        from model import InductionMotorF1b
        with open(Path(__file__).parent.parent / "data" / "parameters.json") as f:
            params = json.load(f)
        m = InductionMotorF1b(params)
        derate = m.derating_factor(T_amb)

    fig.add_trace(go.Scatter(
        x=T_amb, y=derate * 100,
        name="Derating (%)", line=dict(color="#636EFA", width=2.5),
        fill="tozeroy", fillcolor="rgba(99,110,250,0.1)",
    ), row=2, col=1)
    fig.add_vline(x=40, line_dash="dash", line_color="red",
                  annotation_text="40C threshold", row=2, col=1)

    # Panel 4: Current vs PLR
    for i, T_w in enumerate([25, 75, 120, 155]):
        r = model.predict({"load_fraction": plr, "winding_temperature": T_w})
        fig.add_trace(go.Scatter(
            x=plr, y=r["current_A"],
            name=f"I T_w={T_w}C", line=dict(color=colors[i], width=2),
            legendgroup="current",
        ), row=2, col=2)

    fig.update_xaxes(title_text="Part-Load Ratio", row=1, col=1)
    fig.update_xaxes(title_text="Part-Load Ratio", row=1, col=2)
    fig.update_xaxes(title_text="Ambient Temperature (C)", row=2, col=1)
    fig.update_xaxes(title_text="Part-Load Ratio", row=2, col=2)
    fig.update_yaxes(title_text="Efficiency (%)", row=1, col=1)
    fig.update_yaxes(title_text="Losses (kW)", row=1, col=2)
    fig.update_yaxes(title_text="Derating Factor (%)", row=2, col=1)
    fig.update_yaxes(title_text="Current (A)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} Thermal Model",
        height=850,
        template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")

    # Summary
    print("\n--- Thermal Efficiency Summary ---")
    print(f"{'PLR':>6} {'T_w(C)':>7} {'eta(%)':>8} {'Loss(kW)':>9} {'I(A)':>7}")
    for p in [0.5, 0.75, 1.0]:
        for T_w in [25, 75, 120, 155]:
            rv = model.predict({"load_fraction": p, "winding_temperature": T_w})
            print(
                f"{p:>6.2f} {T_w:>7} {float(rv['efficiency'])*100:>8.2f} "
                f"{float(rv['losses_kw']):>9.4f} {float(rv['current_A']):>7.2f}"
            )


if __name__ == "__main__":
    generate_report()
