"""
EC001 -- PEM Fuel Cell (PEMFC) -- F2a Electrochemical
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

# --- Scenario 1: Steady current, thermal transient ---
r1 = model.predict({
    "current_density_A_cm2": 0.6,
    "T_cell_K": 300.0,
    "P_h2_atm": 1.0,
    "P_o2_atm": 0.21,
    "dt": 0.5,
    "duration_s": 300.0,
})

# --- Scenario 2: Current step ---
def step_current(t):
    return 0.3 if t < 60 else 0.8

r2 = model._model.simulate(step_current, 343.15, 1.0, 0.21, 0.5, 200.0)

# --- Scenario 3: Polarization at different temperatures ---
j_sweep = np.linspace(0.01, 1.4, 200)
temps = [313.15, 333.15, 353.15, 373.15]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=[
        "Thermal Transient (j=0.6 A/cm2, cold start)",
        "Voltage Response (j=0.6, cold start)",
        "Current Step Response (0.3->0.8 at t=60s)",
        "Power Density Step Response",
        "Polarization Curves at Various T",
        "Overpotential Breakdown (T=353K)",
    ],
    vertical_spacing=0.10,
    horizontal_spacing=0.10,
)

# Row 1: thermal + voltage transient
fig.add_trace(go.Scatter(x=r1["t"], y=r1["temperature"], name="T(t)",
              line=dict(color="#d62728")), row=1, col=1)
fig.add_trace(go.Scatter(x=r1["t"], y=r1["voltage"], name="V(t)",
              line=dict(color="#1f77b4")), row=1, col=2)

# Row 2: step response
fig.add_trace(go.Scatter(x=r2["t"], y=r2["voltage"], name="V step",
              line=dict(color="#2ca02c"), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=r2["t"], y=r2["power_density"], name="P step",
              line=dict(color="#ff7f0e"), showlegend=False), row=2, col=2)

# Row 3: polarization curves
m = model._model
for i, T in enumerate(temps):
    V_arr = [m.cell_voltage(j, T, 1.0, 0.21) for j in j_sweep]
    label = f"T={T-273.15:.0f}C"
    fig.add_trace(go.Scatter(x=j_sweep, y=V_arr, name=label,
                  line=dict(color=colors[i])), row=3, col=1)

# Overpotential breakdown at T=353K
V_list, act_list, ohm_list, conc_list = [], [], [], []
for j in j_sweep:
    V_list.append(m.cell_voltage(j, 353.15, 1.0, 0.21))
    act_list.append(m.activation_overpotential(j, 353.15))
    ohm_list.append(m.ohmic_overpotential(j, 353.15))
    conc_list.append(m.concentration_overpotential(j))

fig.add_trace(go.Scatter(x=j_sweep, y=act_list, name="eta_act",
              line=dict(color="#FF5722"), showlegend=False), row=3, col=2)
fig.add_trace(go.Scatter(x=j_sweep, y=ohm_list, name="eta_ohm",
              line=dict(color="#2196F3"), showlegend=False), row=3, col=2)
fig.add_trace(go.Scatter(x=j_sweep, y=conc_list, name="eta_conc",
              line=dict(color="#9C27B0"), showlegend=False), row=3, col=2)

for r, c, xl, yl in [
    (1, 1, "Time (s)", "Temperature (K)"),
    (1, 2, "Time (s)", "Cell Voltage (V)"),
    (2, 1, "Time (s)", "Cell Voltage (V)"),
    (2, 2, "Time (s)", "Power Density (W/cm2)"),
    (3, 1, "j (A/cm2)", "Cell Voltage (V)"),
    (3, 2, "j (A/cm2)", "Overpotential (V)"),
]:
    fig.update_xaxes(title_text=xl, row=r, col=c)
    fig.update_yaxes(title_text=yl, row=r, col=c)

fig.update_layout(
    title="<b>EC001 PEMFC -- F2a Full Electrochemical + Thermal ODE</b>",
    height=1000, template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

out_path = os.path.join(OUTPUT_DIR, "EC001_F2a_electrochemical_report.html")
fig.write_html(out_path, include_plotlyjs="cdn")
print(f"Report saved: {out_path}")
