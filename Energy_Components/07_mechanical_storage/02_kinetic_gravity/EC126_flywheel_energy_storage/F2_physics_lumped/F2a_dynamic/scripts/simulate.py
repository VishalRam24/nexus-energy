"""
EC126 -- Flywheel Energy Storage -- F2a Dynamic
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

# Scenario 1: Discharge from full
r1 = model.predict({
    "P_command_W": -200000, "omega0": 3665.0,
    "dt": 1.0, "duration_s": 600.0
})

# Scenario 2: Charge from half SOC
r2 = model.predict({
    "P_command_W": 200000, "omega0": 2750.0,
    "dt": 1.0, "duration_s": 600.0
})

# Scenario 3: Charge-discharge cycle
def cycle_power(t):
    cycle_t = t % 600
    if cycle_t < 300:
        return 200000.0  # Charge
    else:
        return -200000.0  # Discharge

r3 = model._model.simulate(cycle_power, omega0=2750.0, dt=1.0, duration_s=1800.0)

# Scenario 4: Self-discharge (idle)
r4 = model._model.simulate(0.0, omega0=3665.0, dt=10.0, duration_s=7200.0)

fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=[
        "Discharge: Speed vs Time", "Discharge: SOC & Energy",
        "Charge: Speed vs Time", "Charge-Discharge Cycle: SOC",
        "Self-Discharge: SOC over 2hr", "Friction Losses",
    ],
    vertical_spacing=0.10, horizontal_spacing=0.10,
)

fig.add_trace(go.Scatter(x=r1["t"]/60, y=r1["omega"], name="omega (discharge)",
              line=dict(color="#d62728")), row=1, col=1)
fig.add_trace(go.Scatter(x=r1["t"]/60, y=r1["SOC"], name="SOC",
              line=dict(color="#1f77b4")), row=1, col=2)
fig.add_trace(go.Scatter(x=r1["t"]/60, y=r1["E_stored"]/3.6e6, name="E (kWh)",
              line=dict(color="#ff7f0e")), row=1, col=2)

fig.add_trace(go.Scatter(x=r2["t"]/60, y=r2["omega"], name="omega (charge)",
              line=dict(color="#2ca02c")), row=2, col=1)
fig.add_trace(go.Scatter(x=r3["t"]/60, y=r3["SOC"], name="SOC cycle",
              line=dict(color="#9467bd")), row=2, col=2)

fig.add_trace(go.Scatter(x=r4["t"]/3600, y=r4["SOC"], name="SOC self-discharge",
              line=dict(color="#d62728")), row=3, col=1)
fig.add_trace(go.Scatter(x=r1["t"]/60, y=r1["P_loss"]/1000, name="P_loss (kW)",
              line=dict(color="#ff7f0e")), row=3, col=2)

for r, c, xl, yl in [
    (1, 1, "Time (min)", "Angular Speed (rad/s)"),
    (1, 2, "Time (min)", "SOC / Energy (kWh)"),
    (2, 1, "Time (min)", "Angular Speed (rad/s)"),
    (2, 2, "Time (min)", "SOC (-)"),
    (3, 1, "Time (hr)", "SOC (-)"),
    (3, 2, "Time (min)", "Friction Loss (kW)"),
]:
    fig.update_xaxes(title_text=xl, row=r, col=c)
    fig.update_yaxes(title_text=yl, row=r, col=c)

fig.update_layout(
    title="<b>EC126 Flywheel Energy Storage -- F2a Dynamic ODE Model</b>",
    height=1000, template="plotly_white",
)

out_path = os.path.join(OUTPUT_DIR, "EC126_F2a_dynamic_report.html")
fig.write_html(out_path, include_plotlyjs="cdn")
print(f"Report saved: {out_path}")
