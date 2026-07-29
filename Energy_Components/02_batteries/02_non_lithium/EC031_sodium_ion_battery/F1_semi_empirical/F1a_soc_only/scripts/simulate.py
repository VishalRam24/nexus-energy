"""EC031 — Sodium-Ion Battery — F1a SOC-Only — Simulation & HTML Report"""
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
            "OCV vs SOC",
            "Terminal Voltage vs SOC (various C-rates)",
            "Power vs SOC (discharge)",
            "SOC Trajectory (1C discharge simulation)",
        ],
        vertical_spacing=0.14,
        horizontal_spacing=0.1,
    )

    soc_arr = np.linspace(0.0, 1.0, 200)

    # OCV curve
    r_ocv = model.predict({"soc": soc_arr, "current": 0.0})
    fig.add_trace(go.Scatter(
        x=soc_arr, y=r_ocv["ocv"],
        name="OCV (Na-ion)", line=dict(color="#2ca02c", width=2.5)
    ), row=1, col=1)

    # Terminal voltage at various C-rates
    # Capacity = 10 Ah => 1C = 10 A
    C = 10.0
    c_rates = [0.5, 1.0, 2.0, 3.0]
    colors = ["#1f77b4", "#ff7f0e", "#d62728", "#9467bd"]
    for crate, col in zip(c_rates, colors):
        I = crate * C
        r = model.predict({"soc": soc_arr, "current": I})
        fig.add_trace(go.Scatter(
            x=soc_arr, y=r["voltage"],
            name=f"{crate}C ({I:.0f}A)", line=dict(color=col), legendgroup=f"C{crate}"
        ), row=1, col=2)

    # Power vs SOC (1C discharge)
    r_pwr = model.predict({"soc": soc_arr, "current": 1.0 * C})
    fig.add_trace(go.Scatter(
        x=soc_arr, y=r_pwr["power"],
        name="Power (1C)", line=dict(color="#e377c2"), showlegend=True
    ), row=2, col=1)

    # SOC trajectory simulation (1C discharge from SOC=1 to ~0)
    dt = 10.0  # s
    soc_sim = [1.0]
    t_sim = [0.0]
    I_sim = 1.0 * C
    for _ in range(4000):
        soc_now = soc_sim[-1]
        if soc_now <= 0.01:
            break
        r_step = model.predict({"soc": soc_now, "current": I_sim})
        dsoc = float(r_step["dsoc_dt"]) * dt
        soc_sim.append(max(0.0, soc_now + dsoc))
        t_sim.append(t_sim[-1] + dt)

    fig.add_trace(go.Scatter(
        x=np.array(t_sim) / 3600, y=soc_sim,
        name="SOC (1C)", line=dict(color="#8c564b"), showlegend=True
    ), row=2, col=2)

    # Axis labels
    fig.update_xaxes(title_text="SOC", row=1, col=1)
    fig.update_xaxes(title_text="SOC", row=1, col=2)
    fig.update_xaxes(title_text="SOC", row=2, col=1)
    fig.update_xaxes(title_text="Time (hr)", row=2, col=2)
    fig.update_yaxes(title_text="OCV (V)", row=1, col=1)
    fig.update_yaxes(title_text="Voltage (V)", row=1, col=2)
    fig.update_yaxes(title_text="Power (W)", row=2, col=1)
    fig.update_yaxes(title_text="SOC", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']} SOC-Voltage Model<br>"
              f"<sup>CATL-inspired Na-ion Prismatic | {info['source']}</sup>",
        height=850,
        template="plotly_white",
        legend=dict(x=0.01, y=0.55, bgcolor="rgba(255,255,255,0.8)")
    )

    out = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Report saved to {out}")


if __name__ == "__main__":
    generate_report()
