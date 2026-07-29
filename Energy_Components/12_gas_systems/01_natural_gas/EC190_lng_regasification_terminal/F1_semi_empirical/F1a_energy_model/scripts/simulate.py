"""EC190 — LNG Regasification Terminal — F1a — Simulation & HTML Report"""
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
            "Power Demand vs Sendout Rate (varying SEC)",
            "Net SEC vs Cold Recovery Fraction",
            "Gas Sendout (m³/day) vs Sendout Rate",
            "Cold Recovery Power vs Ambient Temperature",
        ],
        vertical_spacing=0.14, horizontal_spacing=0.1,
    )

    m_arr = np.linspace(50, 2000, 200)
    sec_vals = [25.0, 50.0, 80.0, 100.0]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for sec, c in zip(sec_vals, colors):
        r = model.predict({"sendout_rate_ton_per_h": m_arr, "sec_kwh_per_ton": sec})
        fig.add_trace(go.Scatter(x=m_arr, y=r["net_power_kw"] / 1e3,
                                  name=f"SEC={sec} kWh/t",
                                  line=dict(color=c)), row=1, col=1)

    f_cold_arr = np.linspace(0, 0.5, 100)
    for sec, c in zip(sec_vals, colors):
        r = model.predict({"sendout_rate_ton_per_h": 500.0,
                            "sec_kwh_per_ton": sec, "f_cold": f_cold_arr})
        fig.add_trace(go.Scatter(x=f_cold_arr * 100, y=r["net_sec_kwh_per_ton"],
                                  name=f"SEC={sec} kWh/t",
                                  line=dict(color=c), showlegend=False), row=1, col=2)

    r_gas = model.predict({"sendout_rate_ton_per_h": m_arr})
    fig.add_trace(go.Scatter(x=m_arr, y=r_gas["gas_sendout_m3_per_day"] / 1e6,
                              name="Gas sendout", line=dict(color="#9467bd")), row=2, col=1)

    T_arr = np.linspace(260, 310, 100)
    for fc, c in zip([0.1, 0.3, 0.5], ["#1f77b4", "#ff7f0e", "#2ca02c"]):
        r = model.predict({"sendout_rate_ton_per_h": 500.0, "T_ambient_K": T_arr, "f_cold": fc})
        fig.add_trace(go.Scatter(x=T_arr - 273.15, y=r["cold_recovery_kw"],
                                  name=f"f_cold={fc}", line=dict(color=c)), row=2, col=2)

    fig.update_xaxes(title_text="Sendout Rate (ton/h)", row=1, col=1)
    fig.update_xaxes(title_text="Cold Recovery Fraction (%)", row=1, col=2)
    fig.update_xaxes(title_text="Sendout Rate (ton/h)", row=2, col=1)
    fig.update_xaxes(title_text="Ambient Temperature (°C)", row=2, col=2)
    fig.update_yaxes(title_text="Power (MW)", row=1, col=1)
    fig.update_yaxes(title_text="Net SEC (kWh/ton)", row=1, col=2)
    fig.update_yaxes(title_text="Gas Sendout (Mm³/day)", row=2, col=1)
    fig.update_yaxes(title_text="Cold Recovery (kW)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}<br>"
              f"<sup>SEC model with optional cold energy recovery | {info['source']}</sup>",
        height=850, template="plotly_white",
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
