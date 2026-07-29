"""EC082 — Ice Thermal Storage — F1a — Simulation & HTML Report"""
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
            "Energy Stored vs SOC",
            "Charge / Discharge Limits vs SOC",
            "Heat Loss vs Ambient Temperature",
            "24-h Charge / Discharge Cycle",
        ],
        vertical_spacing=0.13,
    )

    soc = np.linspace(0.0, 1.0, 200)

    r = model.predict({"soc": soc})
    fig.add_trace(go.Scatter(x=soc, y=r["energy_stored_kwh"], name="E stored",
                             line=dict(color="steelblue")), row=1, col=1)

    fig.add_trace(go.Scatter(x=soc, y=r["max_charge_kw"],    name="Q_charge max",
                             line=dict(color="dodgerblue")), row=1, col=2)
    fig.add_trace(go.Scatter(x=soc, y=r["max_discharge_kw"], name="Q_discharge max",
                             line=dict(color="firebrick")),  row=1, col=2)

    Ta = np.linspace(-5, 40, 100)
    r = model.predict({"soc": 0.5, "t_ambient": Ta})
    fig.add_trace(go.Scatter(x=Ta, y=r["heat_loss_w"], name="Heat loss",
                             line=dict(color="purple"), showlegend=False), row=2, col=1)

    # 24-h scenario: charge overnight (0–10h), discharge midday (10–22h), idle 22-24
    dt = 60.0
    soc_arr = [0.05]
    t_arr   = []
    for i in range(int(24 * 3600 / dt)):
        t_h = i * dt / 3600.0
        if t_h < 10:
            qc, qd = 100000.0, 0.0
        elif t_h < 22:
            qc, qd = 0.0, 150000.0
        else:
            qc, qd = 0.0, 0.0
        s = soc_arr[-1]
        r = model.predict({"soc": s, "q_charge": qc, "q_discharge": qd, "t_ambient": 22.0})
        soc_arr.append(np.clip(s + float(r["dSOC_dt"]) * dt, 0.0, 1.0))
        t_arr.append(t_h)
    fig.add_trace(go.Scatter(x=t_arr, y=soc_arr[:-1], name="SOC",
                             line=dict(color="darkcyan"), showlegend=False), row=2, col=2)

    fig.update_xaxes(title_text="SOC (-)", row=1, col=1)
    fig.update_xaxes(title_text="SOC (-)", row=1, col=2)
    fig.update_xaxes(title_text="T_amb (degC)", row=2, col=1)
    fig.update_xaxes(title_text="Time (h)", row=2, col=2)
    fig.update_yaxes(title_text="kWh_th", row=1, col=1)
    fig.update_yaxes(title_text="kW_th",  row=1, col=2)
    fig.update_yaxes(title_text="W",      row=2, col=1)
    fig.update_yaxes(title_text="SOC (-)", row=2, col=2)

    fig.update_layout(title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} | 500 kWh",
                      height=800, template="plotly_white")
    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
