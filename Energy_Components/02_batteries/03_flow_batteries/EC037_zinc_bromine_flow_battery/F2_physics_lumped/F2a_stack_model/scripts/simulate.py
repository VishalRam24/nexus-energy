"""
EC037 -- Zinc-Bromine Flow Battery (ZBFB) -- F2a Physics-Lumped Stack Model
Plotly HTML simulation report generator (optional; plotly wrapped in try/except).
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
    print("plotly not installed -- skipping HTML report. (pip install plotly)")
    sys.exit(0)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "model_files")
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = ComponentModel()
m = model._model

# Scenario 1: charge then rest then discharge
def cycle_current(t):
    if t < 3600:
        return -100.0      # charge
    if t < 5400:
        return 0.0         # rest (self-discharge)
    return 120.0           # discharge

r1 = model._model.simulate(cycle_current, soc0=0.2, T0=298.15, flow_Lpm=2.0,
                           dt=10.0, duration_s=9000.0)

# Scenario 2: OCV vs SOC
soc_sweep = np.linspace(0.05, 0.95, 100)
ocv = [float(m.e_nernst(s, s * m.c_Br2_max, 298.15)) for s in soc_sweep]

# Scenario 3: polarization V vs I at SOC=0.5
I_sweep = np.linspace(-250, 250, 100)
V_pol = [m.cell_voltage(0.5, 0.5 * m.c_Br2_max, I, 298.15, 2.0) for I in I_sweep]

fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=[
        "Cycle: charge/rest/discharge -- SOC(t)",
        "Cycle -- stack voltage & OCV",
        "Temperature transient",
        "Coulombic efficiency (t)",
        "OCV vs SOC (per cell)",
        "Polarization V-I (per cell, SOC=0.5)",
    ],
    vertical_spacing=0.10, horizontal_spacing=0.10,
)

fig.add_trace(go.Scatter(x=r1["t"], y=r1["soc"], name="SOC"), row=1, col=1)
fig.add_trace(go.Scatter(x=r1["t"], y=r1["voltage"], name="V"), row=1, col=2)
fig.add_trace(go.Scatter(x=r1["t"], y=r1["ocv"], name="OCV"), row=1, col=2)
fig.add_trace(go.Scatter(x=r1["t"], y=r1["temperature"], name="T",
              line=dict(color="#d62728"), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=r1["t"], y=r1["coulombic_efficiency"], name="eta_C",
              line=dict(color="#2ca02c"), showlegend=False), row=2, col=2)
fig.add_trace(go.Scatter(x=soc_sweep, y=ocv, name="OCV(SOC)",
              line=dict(color="#1f77b4"), showlegend=False), row=3, col=1)
fig.add_trace(go.Scatter(x=I_sweep, y=V_pol, name="V-I",
              line=dict(color="#9C27B0"), showlegend=False), row=3, col=2)

for r, c, xl, yl in [
    (1, 1, "Time (s)", "SOC (-)"),
    (1, 2, "Time (s)", "Stack Voltage (V)"),
    (2, 1, "Time (s)", "Temperature (K)"),
    (2, 2, "Time (s)", "Coulombic Eff. (-)"),
    (3, 1, "SOC (-)", "Cell OCV (V)"),
    (3, 2, "Current (A, +disch)", "Cell Voltage (V)"),
]:
    fig.update_xaxes(title_text=xl, row=r, col=c)
    fig.update_yaxes(title_text=yl, row=r, col=c)

fig.update_layout(
    title="<b>EC037 Zn-Br Flow Battery -- F2a Physics-Lumped Stack (SOC/Br2/Thermal ODE)</b>",
    height=1000, template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

out_path = os.path.join(OUTPUT_DIR, "EC037_F2a_stack_model_report.html")
fig.write_html(out_path, include_plotlyjs="cdn")
print(f"Report saved: {out_path}")
