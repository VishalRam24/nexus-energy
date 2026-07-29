"""EC018 — LFP Battery — F1a — Simulation & HTML Report"""

import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info = model.get_info()
    cap = model.params["cell"]["capacity"]["value"]
    soc = np.linspace(0.0, 1.0, 200)

    fig = make_subplots(rows=2, cols=2,
        subplot_titles=["OCV vs SOC", "Terminal Voltage vs SOC", "Discharge Curve", "Power vs SOC"],
        vertical_spacing=0.12)

    # OCV
    r = model.predict({"soc": soc, "current": 0.0})
    fig.add_trace(go.Scatter(x=soc, y=r["ocv"], name="OCV"), row=1, col=1)

    # V vs SOC at C-rates
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, c in enumerate([0.0, 0.5, 1.0, 2.0, 5.0]):
        r = model.predict({"soc": soc, "current": c * cap})
        fig.add_trace(go.Scatter(x=soc, y=r["voltage"], name=f"{c}C", line=dict(color=colors[i])), row=1, col=2)

    # Discharge curves
    for i, c in enumerate([0.5, 1.0, 2.0]):
        I = c * cap
        soc_t, voltages, ah = 1.0, [], []
        total = 0.0
        while soc_t > 0.0:
            r = model.predict({"soc": soc_t, "current": I})
            v = float(r["voltage"])
            if v <= 2.0: break
            voltages.append(v); ah.append(total)
            soc_t += float(r["dsoc_dt"])
            total += I / 3600.0
            if total > cap * 1.1: break
        fig.add_trace(go.Scatter(x=ah, y=voltages, name=f"{c}C", showlegend=False,
                                  line=dict(color=colors[i+1])), row=2, col=1)

    # Power
    for i, c in enumerate([0.5, 1.0, 2.0, 5.0]):
        r = model.predict({"soc": soc, "current": c * cap})
        fig.add_trace(go.Scatter(x=soc, y=r["power"], name=f"{c}C", showlegend=False,
                                  line=dict(color=colors[i+1])), row=2, col=2)

    fig.update_xaxes(title_text="SOC", row=1, col=1); fig.update_xaxes(title_text="SOC", row=1, col=2)
    fig.update_xaxes(title_text="Ah", row=2, col=1); fig.update_xaxes(title_text="SOC", row=2, col=2)
    fig.update_yaxes(title_text="V", row=1, col=1); fig.update_yaxes(title_text="V", row=1, col=2)
    fig.update_yaxes(title_text="V", row=2, col=1); fig.update_yaxes(title_text="W", row=2, col=2)
    fig.update_layout(title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}", height=800, template="plotly_white")

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
