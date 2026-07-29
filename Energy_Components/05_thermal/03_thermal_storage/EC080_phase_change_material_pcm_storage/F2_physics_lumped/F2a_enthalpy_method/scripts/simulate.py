"""
EC080 -- Phase Change Material (PCM) Storage -- F2a Enthalpy Method
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

# --- Scenario 1: Full charge from cold ---
r1 = model.predict({
    "T_htf_K": 353.15,
    "m_dot_htf": 0.5,
    "T_init_K": 293.15,
    "dt": 30.0,
    "duration_s": 7200.0,
    "mode": "charge",
})

# --- Scenario 2: Discharge from hot ---
r2 = model.predict({
    "T_htf_K": 313.15,
    "m_dot_htf": 0.5,
    "T_init_K": 350.0,
    "dt": 30.0,
    "duration_s": 7200.0,
    "mode": "discharge",
})

# --- Scenario 3: Charge-discharge cycle ---
def cycle_T_htf(t):
    if t < 3600:
        return 353.15  # charge
    else:
        return 313.15  # discharge

r3 = model._model.simulate(cycle_T_htf, 0.5, 293.15, 30.0, 7200.0, mode="cycle")

fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=[
        "Charge: Mean Temperature",
        "Charge: Liquid Fraction",
        "Discharge: Mean Temperature",
        "Discharge: Liquid Fraction",
        "Cycle: Mean Temperature",
        "Cycle: Energy Stored",
    ],
    vertical_spacing=0.10,
    horizontal_spacing=0.10,
)

# Row 1: Charge
fig.add_trace(go.Scatter(x=r1["t"]/60, y=r1["T_mean"]-273.15, name="T_mean (charge)",
              line=dict(color="#d62728")), row=1, col=1)
for i in [0, 4, 9]:
    fig.add_trace(go.Scatter(x=r1["t"]/60, y=r1["T_nodes"][i]-273.15,
                  name=f"Node {i+1}", line=dict(dash="dot"), showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=r1["t"]/60, y=r1["lf_mean"], name="lf_mean (charge)",
              line=dict(color="#1f77b4")), row=1, col=2)

# Row 2: Discharge
fig.add_trace(go.Scatter(x=r2["t"]/60, y=r2["T_mean"]-273.15, name="T_mean (discharge)",
              line=dict(color="#2ca02c")), row=2, col=1)
fig.add_trace(go.Scatter(x=r2["t"]/60, y=r2["lf_mean"], name="lf_mean (discharge)",
              line=dict(color="#ff7f0e")), row=2, col=2)

# Row 3: Cycle
fig.add_trace(go.Scatter(x=r3["t"]/60, y=r3["T_mean"]-273.15, name="T_mean (cycle)",
              line=dict(color="#9467bd")), row=3, col=1)
fig.add_trace(go.Scatter(x=r3["t"]/60, y=r3["E_stored_J"]/1e6, name="E_stored (MJ)",
              line=dict(color="#8c564b")), row=3, col=2)

for r, c, xl, yl in [
    (1, 1, "Time (min)", "Temperature (C)"),
    (1, 2, "Time (min)", "Liquid Fraction (-)"),
    (2, 1, "Time (min)", "Temperature (C)"),
    (2, 2, "Time (min)", "Liquid Fraction (-)"),
    (3, 1, "Time (min)", "Temperature (C)"),
    (3, 2, "Time (min)", "Energy Stored (MJ)"),
]:
    fig.update_xaxes(title_text=xl, row=r, col=c)
    fig.update_yaxes(title_text=yl, row=r, col=c)

fig.update_layout(
    title="<b>EC080 PCM Storage -- F2a Enthalpy Method (10-Node ODE)</b>",
    height=1000, template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

out_path = os.path.join(OUTPUT_DIR, "EC080_F2a_enthalpy_method_report.html")
fig.write_html(out_path, include_plotlyjs="cdn")
print(f"Report saved: {out_path}")
