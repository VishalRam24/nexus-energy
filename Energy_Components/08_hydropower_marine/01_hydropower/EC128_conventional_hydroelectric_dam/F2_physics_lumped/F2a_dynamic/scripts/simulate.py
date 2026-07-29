"""
EC128 -- Conventional Hydroelectric Dam -- F2a Dynamic
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

# Scenario 1: Steady state at half gate
r1 = model.predict({
    "G_ref": 0.5, "dt": 5.0, "duration_s": 3600.0
})

# Scenario 2: Gate step (0.5 -> 0.9 at t=600s) -- load acceptance
def gate_step(t):
    return 0.5 if t < 600 else 0.9

r2 = model._model.simulate(gate_step, dt=2.0, duration_s=3600.0)

# Scenario 3: Load rejection (0.8 -> 0.1 at t=300s)
def gate_reject(t):
    return 0.8 if t < 300 else 0.1

r3 = model._model.simulate(gate_reject, dt=1.0, duration_s=1800.0)

# Scenario 4: Variable inflow
def var_inflow(t):
    return 60.0 + 30.0 * np.sin(2 * np.pi * t / 3600)

r4 = model._model.simulate(0.6, Q_inflow=var_inflow, dt=10.0, duration_s=7200.0)

fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=[
        "Steady State: Power Output", "Steady State: Reservoir Level",
        "Gate Step (Load Accept): Power", "Gate Step: Gate & Flow",
        "Load Rejection: Power", "Variable Inflow: Reservoir Level",
    ],
    vertical_spacing=0.10, horizontal_spacing=0.10,
)

fig.add_trace(go.Scatter(x=r1["t"]/60, y=r1["P_output"]/1e6, name="P steady",
              line=dict(color="#1f77b4")), row=1, col=1)
fig.add_trace(go.Scatter(x=r1["t"]/60, y=r1["H_reservoir"], name="H steady",
              line=dict(color="#d62728")), row=1, col=2)

fig.add_trace(go.Scatter(x=r2["t"]/60, y=r2["P_output"]/1e6, name="P step",
              line=dict(color="#2ca02c")), row=2, col=1)
fig.add_trace(go.Scatter(x=r2["t"]/60, y=r2["G_gate"], name="Gate",
              line=dict(color="#ff7f0e")), row=2, col=2)
fig.add_trace(go.Scatter(x=r2["t"]/60, y=r2["Q_turbine"]/r2["Q_turbine"].max(),
              name="Q/Qmax", line=dict(color="#9467bd", dash="dash")), row=2, col=2)

fig.add_trace(go.Scatter(x=r3["t"]/60, y=r3["P_output"]/1e6, name="P reject",
              line=dict(color="#d62728")), row=3, col=1)
fig.add_trace(go.Scatter(x=r4["t"]/3600, y=r4["H_reservoir"], name="H var inflow",
              line=dict(color="#1f77b4")), row=3, col=2)

for r, c, xl, yl in [
    (1, 1, "Time (min)", "Power (MW)"),
    (1, 2, "Time (min)", "Reservoir Head (m)"),
    (2, 1, "Time (min)", "Power (MW)"),
    (2, 2, "Time (min)", "Gate / Norm. Flow"),
    (3, 1, "Time (min)", "Power (MW)"),
    (3, 2, "Time (hr)", "Reservoir Head (m)"),
]:
    fig.update_xaxes(title_text=xl, row=r, col=c)
    fig.update_yaxes(title_text=yl, row=r, col=c)

fig.update_layout(
    title="<b>EC128 Conventional Hydro Dam -- F2a Penstock + Governor ODE</b>",
    height=1000, template="plotly_white",
)

out_path = os.path.join(OUTPUT_DIR, "EC128_F2a_dynamic_report.html")
fig.write_html(out_path, include_plotlyjs="cdn")
print(f"Report saved: {out_path}")
