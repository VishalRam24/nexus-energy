"""EC207 — CO2 Compression & Pipeline — F1a — Simulation & HTML Report"""
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
            "SEC (kWh/tCO2) vs Outlet Pressure",
            "Shaft Power vs Mass Flow (150 bar target)",
            "Pipeline Pressure Drop vs Length",
            "Stage Discharge Temperature vs Outlet Pressure",
        ],
        vertical_spacing=0.14, horizontal_spacing=0.1,
    )

    P_out_arr = np.linspace(20, 200, 200)
    P_in_vals = [1.5, 3.0, 5.0]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for P_in, c in zip(P_in_vals, colors):
        r = model.predict({"P_inlet_bar": P_in, "P_outlet_bar": P_out_arr,
                            "T_inlet_K": 308.15, "m_dot_kg_s": 100.0})
        lbl = f"P_in={P_in} bar"
        fig.add_trace(go.Scatter(x=P_out_arr, y=r["sec_kwh_per_tco2"], name=lbl,
                                  line=dict(color=c)), row=1, col=1)
        fig.add_trace(go.Scatter(x=P_out_arr, y=r["stage_discharge_T_K"] - 273.15, name=lbl,
                                  line=dict(color=c), showlegend=False), row=2, col=2)

    # Critical pressure marker
    fig.add_vline(x=73.8, line_dash="dash", line_color="red",
                  annotation_text="P_crit", row=1, col=1)

    m_arr = np.linspace(10, 500, 100)
    r_power = model.predict({"P_inlet_bar": 1.5, "P_outlet_bar": 150.0,
                              "T_inlet_K": 308.15, "m_dot_kg_s": m_arr})
    fig.add_trace(go.Scatter(x=m_arr, y=r_power["shaft_power_kw"] / 1e3,
                              name="Shaft power", line=dict(color="#d62728")), row=1, col=2)

    L_arr = np.linspace(10, 400, 100)
    D_vals = [0.2, 0.3, 0.5]
    for D, c in zip(D_vals, colors):
        r_pipe = model.predict({"P_inlet_bar": 1.5, "P_outlet_bar": 150.0,
                                 "m_dot_kg_s": 100.0,
                                 "pipeline_length_km": L_arr, "pipeline_diameter_m": D})
        fig.add_trace(go.Scatter(x=L_arr, y=r_pipe["pipeline_dp_bar"],
                                  name=f"D={D*1000:.0f} mm", line=dict(color=c)), row=2, col=1)

    fig.update_xaxes(title_text="Outlet Pressure (bar)", row=1, col=1)
    fig.update_xaxes(title_text="Mass Flow (kg/s)", row=1, col=2)
    fig.update_xaxes(title_text="Pipeline Length (km)", row=2, col=1)
    fig.update_xaxes(title_text="Outlet Pressure (bar)", row=2, col=2)
    fig.update_yaxes(title_text="SEC (kWh/tCO2)", row=1, col=1)
    fig.update_yaxes(title_text="Power (MW)", row=1, col=2)
    fig.update_yaxes(title_text="ΔP (bar)", row=2, col=1)
    fig.update_yaxes(title_text="Stage T_discharge (°C)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}<br>"
              f"<sup>4-stage polytropic, CO2 γ=1.29, benchmark ~100 kWh/tCO2 | {info['source']}</sup>",
        height=850, template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
