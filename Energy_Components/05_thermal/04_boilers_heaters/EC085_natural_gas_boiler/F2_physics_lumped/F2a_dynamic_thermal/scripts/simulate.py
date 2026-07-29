"""
EC085 -- Natural Gas Boiler -- F2a Dynamic Thermal
Plotly HTML simulation report generator.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("ERROR: plotly not installed. Run: pip install plotly")
    sys.exit(1)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "model_files")
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = ComponentModel()

# --- Scenario 1: Cold start to setpoint ---
r1 = model.predict({
    "T_init_K": 293.15,
    "T_in_K": 333.15,
    "m_dot": 6.0,
    "T_set_K": 353.15,
    "dt": 1.0,
    "duration_s": 600.0,
})

# --- Scenario 2: Part-load operation ---
r2 = model.predict({
    "T_init_K": 353.15,
    "T_in_K": 343.15,
    "m_dot": 3.0,
    "T_set_K": 353.15,
    "dt": 1.0,
    "duration_s": 600.0,
})

# --- Scenario 3: Step load change ---
def step_flow(t):
    return 2.0 if t < 300 else 8.0

r3 = model._model.simulate(353.15, 333.15, step_flow, 353.15, 1.0, 600.0)

fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=[
        "Cold Start: Boiler Temperature",
        "Cold Start: Burner Modulation",
        "Part Load: Temperature & Efficiency",
        "Part Load: Power Flows",
        "Step Load: Temperature Response",
        "Step Load: Burner Modulation",
    ],
    vertical_spacing=0.10,
    horizontal_spacing=0.10,
)

# Row 1: Cold start
fig.add_trace(go.Scatter(x=r1["t"]/60, y=r1["T_boiler"]-273.15, name="T_boiler (cold start)",
              line=dict(color="#d62728")), row=1, col=1)
fig.add_trace(go.Scatter(x=r1["t"]/60, y=r1["modulation"], name="Modulation",
              line=dict(color="#1f77b4")), row=1, col=2)

# Row 2: Part load
fig.add_trace(go.Scatter(x=r2["t"]/60, y=r2["T_boiler"]-273.15, name="T_boiler (part load)",
              line=dict(color="#2ca02c")), row=2, col=1)
fig.add_trace(go.Scatter(x=r2["t"]/60, y=r2["Q_burner_W"]/1000, name="Q_burner (kW)",
              line=dict(color="#ff7f0e")), row=2, col=2)
fig.add_trace(go.Scatter(x=r2["t"]/60, y=r2["Q_output_W"]/1000, name="Q_output (kW)",
              line=dict(color="#2196F3")), row=2, col=2)

# Row 3: Step load
fig.add_trace(go.Scatter(x=r3["t"]/60, y=r3["T_boiler"]-273.15, name="T_boiler (step load)",
              line=dict(color="#9467bd")), row=3, col=1)
fig.add_trace(go.Scatter(x=r3["t"]/60, y=r3["modulation"], name="Modulation (step)",
              line=dict(color="#8c564b")), row=3, col=2)

for r, c, xl, yl in [
    (1, 1, "Time (min)", "Temperature (C)"),
    (1, 2, "Time (min)", "Modulation (-)"),
    (2, 1, "Time (min)", "Temperature (C)"),
    (2, 2, "Time (min)", "Power (kW)"),
    (3, 1, "Time (min)", "Temperature (C)"),
    (3, 2, "Time (min)", "Modulation (-)"),
]:
    fig.update_xaxes(title_text=xl, row=r, col=c)
    fig.update_yaxes(title_text=yl, row=r, col=c)

fig.update_layout(
    title="<b>EC085 Natural Gas Boiler -- F2a Dynamic Thermal Mass</b>",
    height=1000, template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

out_path = os.path.join(OUTPUT_DIR, "EC085_F2a_dynamic_thermal_report.html")
fig.write_html(out_path, include_plotlyjs="cdn")
print(f"Report saved: {out_path}")
