"""EC011 — AEM Electrolyser — F1a — Simulation & HTML Report"""
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
            "V-I Polarization Curve",
            "H2 Production Rate vs Current Density",
            "Stack Power vs Current Density",
            "LHV Efficiency vs Current Density",
        ],
        vertical_spacing=0.14, horizontal_spacing=0.1,
    )

    j = np.linspace(50, 15000, 200)
    T_vals = [40, 60, 80]
    colors = ["#1f77b4", "#ff7f0e", "#d62728"]

    for T, c in zip(T_vals, colors):
        r = model.predict({"current_density": j, "temperature": T})
        lbl = f"{T} C"
        fig.add_trace(go.Scatter(x=j/1e4, y=r["cell_voltage"], name=lbl,
                                  line=dict(color=c)), row=1, col=1)
        fig.add_trace(go.Scatter(x=j/1e4, y=r["hydrogen_rate_mols"]*3600, name=lbl,
                                  line=dict(color=c), showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=j/1e4, y=r["power_kw"], name=lbl,
                                  line=dict(color=c), showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=j/1e4, y=r["efficiency"]*100, name=lbl,
                                  line=dict(color=c), showlegend=False), row=2, col=2)

    fig.update_xaxes(title_text="Current Density (A/cm2)", row=1, col=1)
    fig.update_xaxes(title_text="Current Density (A/cm2)", row=1, col=2)
    fig.update_xaxes(title_text="Current Density (A/cm2)", row=2, col=1)
    fig.update_xaxes(title_text="Current Density (A/cm2)", row=2, col=2)
    fig.update_yaxes(title_text="Cell Voltage (V)", row=1, col=1)
    fig.update_yaxes(title_text="H2 (mol/hr)", row=1, col=2)
    fig.update_yaxes(title_text="Power (kW)", row=2, col=1)
    fig.update_yaxes(title_text="Efficiency (% LHV)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} Tafel + Ohmic Model<br>"
              f"<sup>10-cell, 100 cm2 stack | {info['source']}</sup>",
        height=850, template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
