"""EC191 — Gas Compressor Station — F1a — Simulation & HTML Report"""
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
            "Specific Work vs Discharge Pressure",
            "SEC (kWh/kg) vs Discharge Pressure",
            "Stage Discharge Temperature vs Discharge Pressure",
            "Compression Efficiency vs Discharge Pressure",
        ],
        vertical_spacing=0.14, horizontal_spacing=0.1,
    )

    P_out = np.linspace(55, 150, 200)
    P_in_vals = [40.0, 50.0, 70.0]
    colors = ["#1f77b4", "#ff7f0e", "#d62728"]

    for P_in, c in zip(P_in_vals, colors):
        r = model.predict({"P_inlet": P_in, "P_outlet": P_out, "T_inlet": 288.15, "m_dot": 100.0})
        lbl = f"P_in={P_in} bar"
        fig.add_trace(go.Scatter(x=P_out, y=r["specific_work_kJ_per_kg"], name=lbl,
                                  line=dict(color=c)), row=1, col=1)
        fig.add_trace(go.Scatter(x=P_out, y=r["sec_kwh_per_kg"], name=lbl,
                                  line=dict(color=c), showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=P_out, y=r["stage_discharge_T_K"], name=lbl,
                                  line=dict(color=c), showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=P_out, y=r["compression_efficiency"] * 100, name=lbl,
                                  line=dict(color=c), showlegend=False), row=2, col=2)

    fig.update_xaxes(title_text="Discharge Pressure (bar)", row=1, col=1)
    fig.update_xaxes(title_text="Discharge Pressure (bar)", row=1, col=2)
    fig.update_xaxes(title_text="Discharge Pressure (bar)", row=2, col=1)
    fig.update_xaxes(title_text="Discharge Pressure (bar)", row=2, col=2)
    fig.update_yaxes(title_text="w (kJ/kg)", row=1, col=1)
    fig.update_yaxes(title_text="SEC (kWh/kg)", row=1, col=2)
    fig.update_yaxes(title_text="T_stage (K)", row=2, col=1)
    fig.update_yaxes(title_text="Efficiency (%)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} Polytropic Model<br>"
              f"<sup>2-stage intercooled, γ=1.31, eta_p=0.82 | {info['source']}</sup>",
        height=850, template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
