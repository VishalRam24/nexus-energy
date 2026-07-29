"""EC037 — Zinc-Bromine Flow Battery — F1a — Simulation & HTML Report"""
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
            "Nernst Voltage vs SOC (open-circuit)",
            "Stack Voltage vs SOC (various discharge currents)",
            "Voltage Efficiency vs SOC",
            "Discharge Cycle Simulation",
        ],
        vertical_spacing=0.13,
    )

    soc = np.linspace(0.05, 0.95, 200)

    r = model.predict({"soc": soc, "current": 0.0})
    fig.add_trace(
        go.Scatter(x=soc, y=r["cell_voltage"], name="E_Nernst (cell)", line=dict(color="black")),
        row=1, col=1,
    )

    for I in [25, 50, 100, 150]:
        r = model.predict({"soc": soc, "current": float(I)})
        fig.add_trace(go.Scatter(x=soc, y=r["stack_voltage"], name=f"I={I} A"), row=1, col=2)

    for I in [25, 50, 100]:
        r = model.predict({"soc": soc, "current": float(I)})
        fig.add_trace(go.Scatter(x=soc, y=r["efficiency"] * 100.0, name=f"eta I={I}A"), row=2, col=1)

    # Discharge simulation
    dt = 30.0
    # Rated capacity ~ 100 Ah => 360,000 As
    C_As = 360000.0
    I_dis = 50.0
    soc_sim = [0.95]
    v_stack_sim = []
    t_sim = []
    step = 0
    while soc_sim[-1] > 0.05:
        s = soc_sim[-1]
        r = model.predict({"soc": s, "current": I_dis})
        v_stack_sim.append(float(r["stack_voltage"]))
        t_sim.append(step * dt / 3600.0)
        dsoc = -I_dis * dt / C_As
        soc_sim.append(max(0.05, s + dsoc))
        step += 1
        if step > 5000:
            break

    fig.add_trace(
        go.Scatter(x=t_sim, y=v_stack_sim, name=f"Discharge {int(I_dis)}A",
                   line=dict(color="firebrick"), showlegend=False),
        row=2, col=2,
    )

    fig.update_xaxes(title_text="SOC (-)", row=1, col=1)
    fig.update_xaxes(title_text="SOC (-)", row=1, col=2)
    fig.update_xaxes(title_text="SOC (-)", row=2, col=1)
    fig.update_xaxes(title_text="Time (h)", row=2, col=2)
    fig.update_yaxes(title_text="E_Nernst (V)", row=1, col=1)
    fig.update_yaxes(title_text="Stack Voltage (V)", row=1, col=2)
    fig.update_yaxes(title_text="Voltage Efficiency (%)", row=2, col=1)
    fig.update_yaxes(title_text="Stack Voltage (V)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} | 60-cell, 1000 cm2",
        height=800, template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
