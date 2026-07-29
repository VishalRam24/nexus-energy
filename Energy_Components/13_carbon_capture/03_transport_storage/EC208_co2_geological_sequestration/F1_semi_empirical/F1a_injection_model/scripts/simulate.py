"""EC208 — CO2 Geological Sequestration — F1a — Simulation & HTML Report"""
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
            "Injection Rate vs Wellhead Pressure",
            "Injection Rate vs Permeability",
            "Storage Capacity vs Reservoir Area × Efficiency",
            "Years to Fill vs Injection Rate",
        ],
        vertical_spacing=0.14, horizontal_spacing=0.1,
    )

    P_wh_arr = np.linspace(100, 300, 100)
    k_vals = [10.0, 50.0, 200.0, 1000.0]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for k, c in zip(k_vals, colors):
        r = model.predict({"P_wellhead_bar": P_wh_arr, "k_mD": k})
        fig.add_trace(go.Scatter(x=P_wh_arr, y=r["injection_rate_tco2_per_day"],
                                  name=f"k={k} mD", line=dict(color=c)), row=1, col=1)

    k_arr = np.logspace(-1, 3.5, 100)  # 0.1 to ~3000 mD
    P_wh_vals = [120.0, 150.0, 200.0]
    for P_wh, c in zip(P_wh_vals, colors[:3]):
        r = model.predict({"P_wellhead_bar": P_wh, "k_mD": k_arr})
        fig.add_trace(go.Scatter(x=k_arr, y=r["injection_rate_tco2_per_day"],
                                  name=f"P_wh={P_wh} bar", line=dict(color=c)), row=1, col=2)

    eff_arr = np.linspace(0.005, 0.10, 100)
    area_vals = [50.0, 100.0, 500.0]
    for area, c in zip(area_vals, colors[:3]):
        r = model.predict({"area_km2": area, "storage_efficiency": eff_arr})
        fig.add_trace(go.Scatter(x=eff_arr * 100, y=r["storage_capacity_tco2"] / 1e6,
                                  name=f"Area={area} km²", line=dict(color=c)), row=2, col=1)

    m_arr = np.linspace(1, 100, 100)
    eff_vals = [0.01, 0.02, 0.04]
    for eff, c in zip(eff_vals, colors[:3]):
        cap = model.predict({"storage_efficiency": eff})["storage_capacity_tco2"]
        years = float(cap) / (m_arr * 86400.0 * 365.25 / 1000.0)
        fig.add_trace(go.Scatter(x=m_arr, y=years,
                                  name=f"E={eff*100:.0f}%", line=dict(color=c)), row=2, col=2)

    fig.update_xaxes(title_text="Wellhead Pressure (bar)", row=1, col=1)
    fig.update_xaxes(title_text="Permeability (mD)", type="log", row=1, col=2)
    fig.update_xaxes(title_text="Storage Efficiency (%)", row=2, col=1)
    fig.update_xaxes(title_text="Injection Rate (kg/s)", row=2, col=2)
    fig.update_yaxes(title_text="Injection (tCO2/day)", row=1, col=1)
    fig.update_yaxes(title_text="Injection (tCO2/day)", row=1, col=2)
    fig.update_yaxes(title_text="Storage (MtCO2)", row=2, col=1)
    fig.update_yaxes(title_text="Years to Fill", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}<br>"
              f"<sup>Darcy radial injection + pore-volume capacity | E=1-4% (IPCC 2005) | {info['source']}</sup>",
        height=850, template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
