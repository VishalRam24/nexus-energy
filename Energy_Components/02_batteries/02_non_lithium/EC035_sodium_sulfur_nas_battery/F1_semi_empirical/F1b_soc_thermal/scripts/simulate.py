"""
EC035 -- NaS Battery -- F1b SOC-Thermal -- Simulation & HTML Report Generator

NaS is a high-temperature battery operating at 300-350 degC.
Plots show behavior within the functional operating window.
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

    # Operating window: 573.15 K (300C) to 623.15 K (350C)
    temperatures = [573.15, 583.15, 593.15, 608.15, 623.15]
    temp_labels = ["300C (573K)", "310C (583K)", "320C (593K)", "335C (608K)", "350C (623K)"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    soc = np.linspace(0.05, 0.95, 200)
    discharge_current = 50.0  # A (0.5C for 100 Ah cell)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            f"Terminal Voltage vs SOC at {discharge_current:.0f}A discharge",
            "Internal Resistance vs Temperature (300-350 degC window)",
            "Heat Generation vs Current",
            "Functional Window: Voltage vs Temperature",
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
    )

    for i, (T, label) in enumerate(zip(temperatures, temp_labels)):
        result = model.predict({"soc": soc, "current": discharge_current, "temperature": T})
        fig.add_trace(
            go.Scatter(x=soc, y=result["terminal_voltage"], name=label,
                       line=dict(color=colors[i])),
            row=1, col=1,
        )

    T_range = np.linspace(573.15, 623.15, 200)
    result_T = model.predict({"soc": 0.5, "current": 0.0, "temperature": T_range})
    fig.add_trace(
        go.Scatter(x=T_range - 273.15, y=result_T["internal_resistance"],
                   name="R(T)", line=dict(color="#d62728", width=2), showlegend=False),
        row=1, col=2,
    )

    currents = np.linspace(0.0, 200.0, 200)
    for i, (T, label) in enumerate(zip(temperatures, temp_labels)):
        q_vals = [float(model.predict({"soc": 0.5, "current": I, "temperature": T})["heat_generation"])
                  for I in currents]
        fig.add_trace(
            go.Scatter(x=currents, y=q_vals, name=label,
                       line=dict(color=colors[i]), showlegend=False),
            row=2, col=1,
        )

    # Panel 4: Voltage vs temperature (wider range showing non-functional regions)
    T_wide = np.linspace(543.15, 653.15, 300)  # 270C to 380C
    result_wide = model.predict({"soc": 0.5, "current": 50.0, "temperature": T_wide})
    fig.add_trace(
        go.Scatter(x=T_wide - 273.15, y=result_wide["terminal_voltage"],
                   name="V(T) at 50A", line=dict(color="#2ca02c", width=2), showlegend=False),
        row=2, col=2,
    )
    # Shade non-functional regions
    fig.add_vrect(x0=270, x1=300, fillcolor="red", opacity=0.1,
                  annotation_text="Non-functional", row=2, col=2)
    fig.add_vrect(x0=350, x1=380, fillcolor="red", opacity=0.1,
                  annotation_text="Non-functional", row=2, col=2)

    fig.update_xaxes(title_text="SOC", row=1, col=1)
    fig.update_xaxes(title_text="Temperature (degC)", row=1, col=2)
    fig.update_xaxes(title_text="Current (A)", row=2, col=1)
    fig.update_xaxes(title_text="Temperature (degC)", row=2, col=2)
    fig.update_yaxes(title_text="Voltage (V)", row=1, col=1)
    fig.update_yaxes(title_text="Resistance (Ohm)", row=1, col=2)
    fig.update_yaxes(title_text="Heat Generation (W)", row=2, col=1)
    fig.update_yaxes(title_text="Voltage (V)", row=2, col=2)

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
