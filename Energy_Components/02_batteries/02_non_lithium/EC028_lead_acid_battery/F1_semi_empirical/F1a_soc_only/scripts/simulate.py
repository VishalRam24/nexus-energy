"""EC028 — Lead-Acid Battery — F1a — Simulation & HTML Report"""
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
            "OCV vs SOC",
            "Terminal Voltage vs SOC (various discharge currents)",
            "Voltage vs Current at Fixed SOC",
            "Discharge Simulation (constant current)",
        ],
        vertical_spacing=0.13,
    )

    soc = np.linspace(0.0, 1.0, 200)

    # Row 1 Col 1 — OCV vs SOC
    r = model.predict({"soc": soc, "current": 0.0})
    fig.add_trace(go.Scatter(x=soc, y=r["ocv"], name="OCV", line=dict(color="black")), row=1, col=1)
    fig.add_hline(y=10.5, line_dash="dot", annotation_text="V_min=10.5V", row=1, col=1)
    fig.add_hline(y=12.8, line_dash="dot", annotation_text="V_nom=12.8V", row=1, col=1)

    # Row 1 Col 2 — terminal voltage vs SOC at various discharge rates
    for I in [5, 10, 20, 40]:
        r = model.predict({"soc": soc, "current": float(I)})
        fig.add_trace(go.Scatter(x=soc, y=r["voltage"], name=f"I={I} A"), row=1, col=2)

    # Row 2 Col 1 — voltage vs current at 3 SOC levels
    currents = np.linspace(-50, 50, 200)
    for soc_val in [0.2, 0.5, 0.8]:
        r = model.predict({"soc": float(soc_val), "current": currents})
        fig.add_trace(
            go.Scatter(x=currents, y=r["voltage"], name=f"SOC={int(soc_val*100)}%"),
            row=2, col=1,
        )

    # Row 2 Col 2 — discharge simulation (Euler integration)
    dt = 10.0      # seconds
    n_steps = 2000
    I_dis = 20.0   # A constant discharge
    soc_t = [1.0]
    v_t   = []
    t_arr = []
    for i in range(n_steps):
        s = soc_t[-1]
        r = model.predict({"soc": s, "current": I_dis})
        v_t.append(float(r["voltage"]))
        t_arr.append(i * dt / 3600.0)   # hours
        dsoc = float(r["dsoc_dt"]) * dt
        new_soc = max(0.0, s + dsoc)
        soc_t.append(new_soc)
        if float(r["voltage"]) < 10.5:
            break

    fig.add_trace(
        go.Scatter(x=t_arr, y=v_t, name=f"{int(I_dis)}A constant discharge",
                   line=dict(color="steelblue"), showlegend=False),
        row=2, col=2,
    )

    fig.update_xaxes(title_text="SOC (-)", row=1, col=1)
    fig.update_xaxes(title_text="SOC (-)", row=1, col=2)
    fig.update_xaxes(title_text="Current (A)", row=2, col=1)
    fig.update_xaxes(title_text="Time (h)", row=2, col=2)

    fig.update_yaxes(title_text="OCV (V)", row=1, col=1)
    fig.update_yaxes(title_text="Voltage (V)", row=1, col=2)
    fig.update_yaxes(title_text="Voltage (V)", row=2, col=1)
    fig.update_yaxes(title_text="Terminal Voltage (V)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} | 12V / 100Ah",
        height=800,
        template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
