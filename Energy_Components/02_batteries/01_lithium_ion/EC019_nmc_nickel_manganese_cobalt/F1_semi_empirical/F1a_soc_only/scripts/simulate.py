"""
EC019 — NMC Battery — F1a SOC-Only — Simulation & HTML Report Generator

Generates interactive Plotly charts showing:
1. OCV vs SOC curve
2. Terminal voltage vs SOC at various C-rates
3. Discharge curve (voltage vs capacity)
4. Power vs SOC at various C-rates
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from predict import ComponentModel

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("plotly required: pip install plotly")
    sys.exit(1)


def generate_report():
    model = ComponentModel()
    info = model.get_info()
    capacity_ah = model.params["cell"]["capacity"]["value"]

    soc = np.linspace(0.0, 1.0, 200)

    # --- Fig 1: OCV vs SOC ---
    ocv_result = model.predict({"soc": soc, "current": 0.0})

    # --- Fig 2: V vs SOC at different C-rates ---
    c_rates = [0.0, 0.2, 0.5, 1.0, 2.0, 5.0]
    currents = [c * capacity_ah for c in c_rates]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "OCV vs SOC",
            "Terminal Voltage vs SOC (Discharge)",
            "Discharge Curve (V vs Delivered Ah)",
            "Power vs SOC (Discharge)",
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    # Plot 1: OCV
    fig.add_trace(
        go.Scatter(x=soc, y=ocv_result["ocv"], name="OCV", line=dict(width=2)),
        row=1, col=1,
    )

    # Plot 2: V vs SOC at C-rates
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for i, (c, I) in enumerate(zip(c_rates, currents)):
        result = model.predict({"soc": soc, "current": I})
        fig.add_trace(
            go.Scatter(x=soc, y=result["voltage"], name=f"{c}C ({I:.1f}A)",
                       line=dict(color=colors[i])),
            row=1, col=2,
        )

    # Plot 3: Discharge curve (V vs delivered Ah) — simulate constant-current discharge
    for i, (c, I) in enumerate(zip([0.2, 0.5, 1.0, 2.0], [c * capacity_ah for c in [0.2, 0.5, 1.0, 2.0]])):
        dt = 1.0  # 1 second steps
        soc_t = 1.0
        voltages, ah_delivered = [], []
        total_ah = 0.0
        while soc_t > 0.0:
            result = model.predict({"soc": soc_t, "current": I})
            v = float(result["voltage"])
            if v <= 2.5:
                break
            voltages.append(v)
            ah_delivered.append(total_ah)
            dsoc = float(result["dsoc_dt"]) * dt
            soc_t += dsoc
            total_ah += I * dt / 3600.0
            if total_ah > capacity_ah * 1.1:
                break
        fig.add_trace(
            go.Scatter(x=ah_delivered, y=voltages, name=f"{c}C discharge",
                       line=dict(color=colors[i + 1]), showlegend=False),
            row=2, col=1,
        )

    # Plot 4: Power vs SOC
    for i, (c, I) in enumerate(zip(c_rates[1:], currents[1:])):
        result = model.predict({"soc": soc, "current": I})
        fig.add_trace(
            go.Scatter(x=soc, y=result["power"], name=f"{c}C power",
                       line=dict(color=colors[i + 1]), showlegend=False),
            row=2, col=2,
        )

    fig.update_xaxes(title_text="SOC", row=1, col=1)
    fig.update_xaxes(title_text="SOC", row=1, col=2)
    fig.update_xaxes(title_text="Delivered Capacity (Ah)", row=2, col=1)
    fig.update_xaxes(title_text="SOC", row=2, col=2)
    fig.update_yaxes(title_text="Voltage (V)", row=1, col=1)
    fig.update_yaxes(title_text="Voltage (V)", row=1, col=2)
    fig.update_yaxes(title_text="Voltage (V)", row=2, col=1)
    fig.update_yaxes(title_text="Power (W)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} — {info['name']} — {info['fidelity']}: {info['description']}",
        height=800,
        template="plotly_white",
    )

    output_path = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    generate_report()
