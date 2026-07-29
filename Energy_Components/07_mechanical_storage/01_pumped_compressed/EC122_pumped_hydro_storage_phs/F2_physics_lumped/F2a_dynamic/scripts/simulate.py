"""
EC122 -- Pumped Hydro Storage (PHS) -- F2a Dynamic
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

# Scenario 1: Turbine generation at rated power
r1 = model.predict({
    "P_electrical_W": 200e6, "mode": "turbine",
    "dt": 5.0, "duration_s": 3600.0
})

# Scenario 2: Pump mode (charging)
r2 = model.predict({
    "P_electrical_W": -200e6, "mode": "pump",
    "dt": 5.0, "duration_s": 3600.0
})

# Scenario 3: Mode switching (turbine -> idle -> pump)
def mode_switch(t):
    if t < 1200:
        return "turbine"
    elif t < 1800:
        return "idle"
    else:
        return "pump"

def P_switch(t):
    if t < 1200:
        return 150e6
    elif t < 1800:
        return 0.0
    else:
        return -150e6

r3 = model._model.simulate(P_switch, mode_switch, dt=5.0, duration_s=3600.0)

fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=[
        "Turbine Mode: Flow Rate", "Turbine Mode: Reservoir Levels",
        "Pump Mode: Flow Rate", "Pump Mode: SOC",
        "Mode Switching: Power", "Mode Switching: Reservoir Levels",
    ],
    vertical_spacing=0.10, horizontal_spacing=0.10,
)

fig.add_trace(go.Scatter(x=r1["t"]/60, y=r1["Q"], name="Q turbine",
              line=dict(color="#1f77b4")), row=1, col=1)
fig.add_trace(go.Scatter(x=r1["t"]/60, y=r1["H_upper"], name="H_upper",
              line=dict(color="#d62728")), row=1, col=2)
fig.add_trace(go.Scatter(x=r1["t"]/60, y=r1["H_lower"], name="H_lower",
              line=dict(color="#2ca02c")), row=1, col=2)

fig.add_trace(go.Scatter(x=r2["t"]/60, y=r2["Q"], name="Q pump",
              line=dict(color="#ff7f0e")), row=2, col=1)
fig.add_trace(go.Scatter(x=r2["t"]/60, y=r2["SOC"], name="SOC pump",
              line=dict(color="#9467bd")), row=2, col=2)

fig.add_trace(go.Scatter(x=r3["t"]/60, y=r3["P_electrical"]/1e6, name="P_elec",
              line=dict(color="#1f77b4")), row=3, col=1)
fig.add_trace(go.Scatter(x=r3["t"]/60, y=r3["H_upper"], name="H_up switch",
              line=dict(color="#d62728")), row=3, col=2)
fig.add_trace(go.Scatter(x=r3["t"]/60, y=r3["H_lower"], name="H_low switch",
              line=dict(color="#2ca02c")), row=3, col=2)

for r, c, xl, yl in [
    (1, 1, "Time (min)", "Flow Rate (m3/s)"),
    (1, 2, "Time (min)", "Level (m)"),
    (2, 1, "Time (min)", "Flow Rate (m3/s)"),
    (2, 2, "Time (min)", "SOC (-)"),
    (3, 1, "Time (min)", "Power (MW)"),
    (3, 2, "Time (min)", "Level (m)"),
]:
    fig.update_xaxes(title_text=xl, row=r, col=c)
    fig.update_yaxes(title_text=yl, row=r, col=c)

fig.update_layout(
    title="<b>EC122 Pumped Hydro Storage -- F2a Dynamic ODE Model</b>",
    height=1000, template="plotly_white",
)

out_path = os.path.join(OUTPUT_DIR, "EC122_F2a_dynamic_report.html")
fig.write_html(out_path, include_plotlyjs="cdn")
print(f"Report saved: {out_path}")
