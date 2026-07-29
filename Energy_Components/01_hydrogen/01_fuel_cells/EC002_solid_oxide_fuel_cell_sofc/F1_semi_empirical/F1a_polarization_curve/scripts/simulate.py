"""EC002 — SOFC — F1a Polarization Curve — Simulation & HTML Report"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_report():
    model = ComponentModel()
    info  = model.get_info()
    m     = model._model

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Polarization Curve V(j) — parametric temperature",
            "Power Density P(j) — parametric temperature",
            "Overpotential Breakdown at T=800°C",
            "Efficiency vs Current Density — parametric temperature",
        ],
        vertical_spacing=0.13,
    )

    j = np.linspace(0.01, 1.79, 300)

    # Panel 1: Polarization curves for several temperatures
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for T, col in zip([650.0, 700.0, 800.0, 900.0, 1000.0], colors):
        r = model.predict({"current_density": j, "temperature": T})
        fig.add_trace(go.Scatter(x=j, y=r["cell_voltage"], name=f"T={T:.0f}°C",
                                 line=dict(color=col)), row=1, col=1)

    # Panel 2: Power density curves
    for T, col in zip([650.0, 800.0, 1000.0], ["#1f77b4", "#2ca02c", "#d62728"]):
        r = model.predict({"current_density": j, "temperature": T})
        fig.add_trace(go.Scatter(x=j, y=r["power_density"], name=f"P T={T:.0f}°C",
                                 line=dict(color=col, dash="dash")), row=1, col=2)

    # Panel 3: Overpotential breakdown at T=800C
    T_K = 800.0 + 273.15
    EN   = m.E_nernst(T_K) * np.ones_like(j)
    VA   = m.V_act(j, T_K)
    VO   = m.V_ohm(j)
    VC   = m.V_conc(j, T_K)
    V_cell = EN - VA - VO - VC
    V_cell = np.clip(V_cell, 0.0, EN)
    fig.add_trace(go.Scatter(x=j, y=EN,    name="E_Nernst", fill=None,  line=dict(color="black", dash="dot")), row=2, col=1)
    fig.add_trace(go.Scatter(x=j, y=V_cell, name="V_cell", fill=None,  line=dict(color="blue")), row=2, col=1)
    fig.add_trace(go.Scatter(x=j, y=VA,    name="V_act",   fill=None,  line=dict(color="orange")), row=2, col=1)
    fig.add_trace(go.Scatter(x=j, y=VO,    name="V_ohm",   fill=None,  line=dict(color="red")), row=2, col=1)
    fig.add_trace(go.Scatter(x=j, y=VC,    name="V_conc",  fill=None,  line=dict(color="purple")), row=2, col=1)

    # Panel 4: Efficiency
    for T, col in zip([650.0, 800.0, 1000.0], ["#1f77b4", "#2ca02c", "#d62728"]):
        r = model.predict({"current_density": j, "temperature": T})
        fig.add_trace(go.Scatter(x=j, y=r["efficiency"] * 100.0, name=f"eta T={T:.0f}°C",
                                 line=dict(color=col, dash="longdash")), row=2, col=2)

    fig.update_xaxes(title_text="Current Density (A/cm²)", row=1, col=1)
    fig.update_xaxes(title_text="Current Density (A/cm²)", row=1, col=2)
    fig.update_xaxes(title_text="Current Density (A/cm²)", row=2, col=1)
    fig.update_xaxes(title_text="Current Density (A/cm²)", row=2, col=2)
    fig.update_yaxes(title_text="Cell Voltage (V)", row=1, col=1)
    fig.update_yaxes(title_text="Power Density (W/cm²)", row=1, col=2)
    fig.update_yaxes(title_text="Voltage / Loss (V)", row=2, col=1)
    fig.update_yaxes(title_text="Efficiency (%)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}",
        height=800,
        template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
