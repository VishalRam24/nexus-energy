"""
EC025 -- Lithium-Sulfur Battery -- F1b SOC-Thermal -- Simulation & HTML Report Generator
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

    temperatures = [258.15, 273.15, 298.15, 313.15, 333.15]
    temp_labels = ["-15C (258K)", "0C (273K)", "25C (298K)", "40C (313K)", "60C (333K)"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    soc = np.linspace(0.05, 0.95, 200)
    discharge_current = 1.5  # A (0.5C for 3Ah Li-S)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            f"Terminal Voltage vs SOC at {discharge_current:.1f}A discharge",
            "Internal Resistance vs Temperature",
            "Heat Generation vs Current (note: can be negative at low I)",
            "Effective Capacity vs Temperature",
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
    )

    for i, (T, label) in enumerate(zip(temperatures, temp_labels)):
        result = model.predict({"soc": soc, "current": discharge_current, "temperature": T})
        fig.add_trace(go.Scatter(x=soc, y=result["terminal_voltage"], name=label,
                                 line=dict(color=colors[i])), row=1, col=1)

    T_range = np.linspace(258.15, 333.15, 200)
    result_T = model.predict({"soc": 0.5, "current": 0.0, "temperature": T_range})
    fig.add_trace(go.Scatter(x=T_range - 273.15, y=result_T["internal_resistance"],
                             name="R(T)", line=dict(color="#d62728", width=2), showlegend=False),
                  row=1, col=2)

    currents = np.linspace(0.0, 6.0, 200)
    for i, (T, label) in enumerate(zip(temperatures, temp_labels)):
        q_vals = [float(model.predict({"soc": 0.5, "current": I, "temperature": T})["heat_generation"])
                  for I in currents]
        fig.add_trace(go.Scatter(x=currents, y=q_vals, name=label,
                                 line=dict(color=colors[i]), showlegend=False), row=2, col=1)

    result_cap = model.predict({"soc": 0.5, "current": 0.0, "temperature": T_range})
    fig.add_trace(go.Scatter(x=T_range - 273.15, y=result_cap["effective_capacity"],
                             name="C(T)", line=dict(color="#2ca02c", width=2), showlegend=False),
                  row=2, col=2)

    fig.update_xaxes(title_text="SOC", row=1, col=1)
    fig.update_xaxes(title_text="Temperature (degC)", row=1, col=2)
    fig.update_xaxes(title_text="Current (A)", row=2, col=1)
    fig.update_xaxes(title_text="Temperature (degC)", row=2, col=2)
    fig.update_yaxes(title_text="Voltage (V)", row=1, col=1)
    fig.update_yaxes(title_text="Resistance (Ohm)", row=1, col=2)
    fig.update_yaxes(title_text="Heat Generation (W)", row=2, col=1)
    fig.update_yaxes(title_text="Capacity (Ah)", row=2, col=2)

    fig.update_layout(
        title=f"{info['ec_id']} -- {info['name']} -- {info['fidelity']}",
        height=800, template="plotly_white",
    )

    output_path = Path(__file__).parent.parent / "simulation_report.html"
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    generate_report()
