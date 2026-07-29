"""EC069 — GSHP — F2a — Simulation & HTML Report"""
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
            "COP vs Condensing Temp",
            "Evaporator Temp (self-consistent) vs T_cond",
            "Compressor Power vs T_cond",
            "Brine Source Temperature vs T_cond",
        ],
        vertical_spacing=0.14,
    )

    T_conds = np.linspace(28, 55, 25)
    cops, t_evaps, powers, t_sources = [], [], [], []
    for Tc in T_conds:
        r = model.predict({"T_cond_degC": float(Tc)})
        cops.append(r["cop"])
        t_evaps.append(r["T_evap_degC"])
        powers.append(r["compressor_power_kw"])
        t_sources.append(r["T_source_degC"])

    fig.add_trace(go.Scatter(x=T_conds, y=cops, name="COP", mode="lines+markers"), row=1, col=1)
    fig.add_trace(go.Scatter(x=T_conds, y=t_evaps, name="T_evap", mode="lines+markers"), row=1, col=2)
    fig.add_trace(go.Scatter(x=T_conds, y=powers, name="W_comp", mode="lines+markers"), row=2, col=1)
    fig.add_trace(go.Scatter(x=T_conds, y=t_sources, name="T_source", mode="lines+markers"), row=2, col=2)

    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(title_text="T_cond (C)", row=i, col=j)
    fig.update_yaxes(title_text="COP", row=1, col=1)
    fig.update_yaxes(title_text="T_evap (C)", row=1, col=2)
    fig.update_yaxes(title_text="kW_e", row=2, col=1)
    fig.update_yaxes(title_text="T_source (C)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} Vapor Cycle + Ground Loop",
        height=800, template="plotly_white",
    )
    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
