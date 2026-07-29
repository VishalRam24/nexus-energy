"""
EC042 -- Pseudocapacitor -- F1b RC-Thermal -- Simulation & HTML Report Generator

Pseudocapacitor (RuO2): -30 to 60 degC, 1 V window, higher ESR than EDLC.
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

    temperatures = [243.15, 263.15, 298.15, 313.15, 333.15]
    temp_labels = ["-30C (243K)", "-10C (263K)", "25C (298K)", "40C (313K)", "60C (333K)"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    v_cap = np.linspace(0.01, 1.0, 200)
    discharge_current = 50.0  # A

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            f"Terminal Voltage vs V_cap at {discharge_current:.0f}A discharge",
            "ESR vs Temperature (Arrhenius, E_a=12 kJ/mol)",
            "Heat Generation vs Current",
            "Capacitance vs Temperature",
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
    )

    for i, (T, label) in enumerate(zip(temperatures, temp_labels)):
        result = model.predict({"v_cap": v_cap, "current": discharge_current, "temperature": T})
        fig.add_trace(
            go.Scatter(x=v_cap, y=result["terminal_voltage"], name=label,
                       line=dict(color=colors[i])),
            row=1, col=1,
        )

    T_range = np.linspace(243.15, 333.15, 200)
    result_T = model.predict({"v_cap": 0.5, "current": 0.0, "temperature": T_range})
    fig.add_trace(
        go.Scatter(x=T_range - 273.15, y=result_T["esr"] * 1000,
                   name="ESR(T)", line=dict(color="#d62728", width=2), showlegend=False),
        row=1, col=2,
    )

    currents = np.linspace(0.0, 100.0, 200)
    for i, (T, label) in enumerate(zip(temperatures, temp_labels)):
        q_vals = [float(model.predict({"v_cap": 0.5, "current": I, "temperature": T})["heat_generation"])
                  for I in currents]
        fig.add_trace(
            go.Scatter(x=currents, y=q_vals, name=label,
                       line=dict(color=colors[i]), showlegend=False),
            row=2, col=1,
        )

    result_cap = model.predict({"v_cap": 0.5, "current": 0.0, "temperature": T_range})
    fig.add_trace(
        go.Scatter(x=T_range - 273.15, y=result_cap["capacitance"],
                   name="C(T)", line=dict(color="#2ca02c", width=2), showlegend=False),
        row=2, col=2,
    )

    fig.update_xaxes(title_text="Capacitor Voltage V_cap (V)", row=1, col=1)
    fig.update_xaxes(title_text="Temperature (degC)", row=1, col=2)
    fig.update_xaxes(title_text="Current (A)", row=2, col=1)
    fig.update_xaxes(title_text="Temperature (degC)", row=2, col=2)
    fig.update_yaxes(title_text="Terminal Voltage (V)", row=1, col=1)
    fig.update_yaxes(title_text="ESR (mOhm)", row=1, col=2)
    fig.update_yaxes(title_text="Heat Generation (W)", row=2, col=1)
    fig.update_yaxes(title_text="Capacitance (F)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} -- {info['name']} -- {info['fidelity']}: {info['description']}",
        height=800,
        template="plotly_white",
    )

    output_path = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    generate_report()
