"""
EC037 -- Zinc-Bromine Flow Battery -- F1b SOC-Thermal -- Simulation & HTML Report Generator
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

    # ZBFB: 15-40 degC
    temperatures = [288.15, 293.15, 298.15, 308.15, 313.15]
    temp_labels = ["15C (288K)", "20C (293K)", "25C (298K)", "35C (308K)", "40C (313K)"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    soc = np.linspace(0.05, 0.95, 200)
    discharge_current = 50.0

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            f"Stack Voltage vs SOC at {discharge_current:.0f}A discharge",
            "Cell Resistance vs Temperature (Arrhenius)",
            "Heat Generation vs Current",
            "Voltage Efficiency vs SOC",
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
    )

    for i, (T, label) in enumerate(zip(temperatures, temp_labels)):
        result = model.predict({"soc": soc, "current": discharge_current, "temperature": T})
        fig.add_trace(
            go.Scatter(x=soc, y=result["stack_voltage"], name=label,
                       line=dict(color=colors[i])),
            row=1, col=1,
        )

    T_range = np.linspace(288.15, 313.15, 200)
    result_T = model.predict({"soc": 0.5, "current": 0.0, "temperature": T_range})
    fig.add_trace(
        go.Scatter(x=T_range - 273.15, y=result_T["internal_resistance_cell"],
                   name="R_cell(T)", line=dict(color="#d62728", width=2), showlegend=False),
        row=1, col=2,
    )

    currents = np.linspace(0.0, 150.0, 200)
    for i, (T, label) in enumerate(zip(temperatures, temp_labels)):
        q_vals = [float(model.predict({"soc": 0.5, "current": I, "temperature": T})["heat_generation"])
                  for I in currents]
        fig.add_trace(
            go.Scatter(x=currents, y=q_vals, name=label,
                       line=dict(color=colors[i]), showlegend=False),
            row=2, col=1,
        )

    for i, (T, label) in enumerate(zip(temperatures, temp_labels)):
        result = model.predict({"soc": soc, "current": discharge_current, "temperature": T})
        fig.add_trace(
            go.Scatter(x=soc, y=result["efficiency"], name=label,
                       line=dict(color=colors[i]), showlegend=False),
            row=2, col=2,
        )

    fig.update_xaxes(title_text="SOC", row=1, col=1)
    fig.update_xaxes(title_text="Temperature (degC)", row=1, col=2)
    fig.update_xaxes(title_text="Current (A)", row=2, col=1)
    fig.update_xaxes(title_text="SOC", row=2, col=2)
    fig.update_yaxes(title_text="Stack Voltage (V)", row=1, col=1)
    fig.update_yaxes(title_text="Cell Resistance (Ohm)", row=1, col=2)
    fig.update_yaxes(title_text="Heat Generation (W)", row=2, col=1)
    fig.update_yaxes(title_text="Efficiency", row=2, col=2)

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
