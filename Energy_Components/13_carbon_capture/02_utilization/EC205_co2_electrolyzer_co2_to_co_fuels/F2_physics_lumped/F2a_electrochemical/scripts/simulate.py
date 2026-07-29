"""
EC205 -- CO2 Electrolyzer (CO2 -> CO/Fuels) -- F2a Electrochemical
Plotly HTML simulation report generator (optional; plotly import guarded).
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
    print("plotly not installed -- skipping HTML report. Run: pip install plotly")
    sys.exit(0)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "model_files")
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = ComponentModel()
m = model._model

# --- Scenario 1: steady current, thermal transient from cold start ---
r1 = model.predict({"current_density_A_cm2": 0.4, "T_cell_K": 300.0,
                    "dt": 0.5, "duration_s": 300.0})

# --- Scenario 2: current step ---
def step_current(t):
    return 0.15 if t < 60 else 0.45

r2 = m.simulate(step_current, 313.15, 0.5, 200.0)

# --- Scenario 3: polarization + FE + SEC sweep over j ---
j_sweep = np.linspace(0.005, 0.595, 200)
V_arr = [m.cell_voltage(j, 333.15) for j in j_sweep]
fe_arr = [m.faradaic_efficiency(j) for j in j_sweep]
sec_arr = [m.energy_per_kg_CO_kWh(j, 333.15) for j in j_sweep]
act_c = [m.activation_cathode(j, 333.15) for j in j_sweep]
act_a = [m.activation_anode(j, 333.15) for j in j_sweep]
ohm = [m.ohmic_overpotential(j, 333.15) for j in j_sweep]
conc = [m.concentration_overpotential(j, T=333.15) for j in j_sweep]

fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=[
        "Thermal Transient (j=0.4, cold start)",
        "Cell Voltage Transient",
        "Current Step Response: V & FE_CO",
        "CO Production Rate (step)",
        "Polarization V(j) & Faradaic Efficiency FE_CO(j)",
        "Overpotential Breakdown (T=60C)",
    ],
    specs=[[{}, {}], [{"secondary_y": True}, {}], [{"secondary_y": True}, {}]],
    vertical_spacing=0.10, horizontal_spacing=0.10,
)

fig.add_trace(go.Scatter(x=r1["t"], y=r1["temperature"], name="T(t)",
              line=dict(color="#d62728")), row=1, col=1)
fig.add_trace(go.Scatter(x=r1["t"], y=r1["voltage"], name="V(t)",
              line=dict(color="#1f77b4")), row=1, col=2)

fig.add_trace(go.Scatter(x=r2["t"], y=r2["voltage"], name="V step",
              line=dict(color="#2ca02c")), row=2, col=1, secondary_y=False)
fig.add_trace(go.Scatter(x=r2["t"], y=r2["faradaic_efficiency"], name="FE_CO step",
              line=dict(color="#9467bd", dash="dot")), row=2, col=1, secondary_y=True)
fig.add_trace(go.Scatter(x=r2["t"], y=r2["co_rate_mol_s"], name="CO rate",
              line=dict(color="#ff7f0e"), showlegend=False), row=2, col=2)

fig.add_trace(go.Scatter(x=j_sweep, y=V_arr, name="V(j)",
              line=dict(color="#1f77b4")), row=3, col=1, secondary_y=False)
fig.add_trace(go.Scatter(x=j_sweep, y=fe_arr, name="FE_CO(j)",
              line=dict(color="#9467bd")), row=3, col=1, secondary_y=True)
fig.add_trace(go.Scatter(x=j_sweep, y=act_c, name="eta_cathode",
              line=dict(color="#FF5722"), showlegend=True), row=3, col=2)
fig.add_trace(go.Scatter(x=j_sweep, y=act_a, name="eta_anode",
              line=dict(color="#795548"), showlegend=True), row=3, col=2)
fig.add_trace(go.Scatter(x=j_sweep, y=ohm, name="eta_ohm",
              line=dict(color="#2196F3"), showlegend=True), row=3, col=2)
fig.add_trace(go.Scatter(x=j_sweep, y=conc, name="eta_conc",
              line=dict(color="#9C27B0"), showlegend=True), row=3, col=2)

fig.update_yaxes(range=[0, 5], row=3, col=2)
fig.update_layout(
    title="<b>EC205 CO2 Electrolyzer -- F2a Electrochemical + Thermal ODE</b>",
    height=1000, template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

out_path = os.path.join(OUTPUT_DIR, "EC205_F2a_electrochemical_report.html")
fig.write_html(out_path, include_plotlyjs="cdn")
print(f"Report saved: {out_path}")
