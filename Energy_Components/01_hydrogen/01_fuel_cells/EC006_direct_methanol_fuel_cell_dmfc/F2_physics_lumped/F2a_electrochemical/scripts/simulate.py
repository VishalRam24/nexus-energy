"""
EC006 -- Direct Methanol Fuel Cell (DMFC) -- F2a Electrochemical
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
    print("plotly not installed -- skipping HTML report. (pip install plotly)")
    sys.exit(0)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "model_files")
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = ComponentModel()
m = model._model

# --- Scenario 1: cold-start thermal transient ---
r1 = model.predict({
    "current_density_A_cm2": 0.25,
    "T_cell_K": 318.15,
    "c_MeOH_molar": 1.0,
    "dt": 1.0,
    "duration_s": 400.0,
})

# --- Scenario 2: current step ---
def step_current(t):
    return 0.10 if t < 80 else 0.30

r2 = m.simulate(step_current, 353.15, 1.0, 1.0, 200.0)

# --- Scenario 3: polarization at various [MeOH] ---
j_sweep = np.linspace(0.002, 0.39, 200)
concs = [0.5, 1.0, 2.0]
colors = ["#1f77b4", "#2ca02c", "#d62728"]

fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=[
        "Thermal Transient (j=0.25, cold start)",
        "Voltage Response (cold start)",
        "Current Step (0.10->0.30 at t=80s)",
        "Fuel Efficiency vs Step",
        "Polarization at Various [MeOH]",
        "Overpotential Breakdown (353K, 1M)",
    ],
    vertical_spacing=0.10, horizontal_spacing=0.10,
)

fig.add_trace(go.Scatter(x=r1["t"], y=r1["temperature"], name="T(t)",
              line=dict(color="#d62728")), row=1, col=1)
fig.add_trace(go.Scatter(x=r1["t"], y=r1["voltage"], name="V(t)",
              line=dict(color="#1f77b4")), row=1, col=2)
fig.add_trace(go.Scatter(x=r2["t"], y=r2["voltage"], name="V step",
              line=dict(color="#2ca02c"), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=r2["t"], y=r2["fuel_efficiency"], name="fuel eff",
              line=dict(color="#ff7f0e"), showlegend=False), row=2, col=2)

for i, c in enumerate(concs):
    V_arr = [m.cell_voltage(j, 353.15, c) for j in j_sweep]
    fig.add_trace(go.Scatter(x=j_sweep, y=V_arr, name=f"[MeOH]={c}M",
                  line=dict(color=colors[i])), row=3, col=1)

eta_a = [m.anode_overpotential(j, 353.15) for j in j_sweep]
jx = m.crossover_current(353.15, 1.0)
eta_c = [m.cathode_overpotential(j, 353.15, jx) for j in j_sweep]
eta_ohm = [m.ohmic_overpotential(j, 353.15) for j in j_sweep]
eta_conc = [m.concentration_overpotential(j, 353.15) for j in j_sweep]
fig.add_trace(go.Scatter(x=j_sweep, y=eta_a, name="eta_anode",
              line=dict(color="#FF5722"), showlegend=False), row=3, col=2)
fig.add_trace(go.Scatter(x=j_sweep, y=eta_c, name="eta_cathode",
              line=dict(color="#2196F3"), showlegend=False), row=3, col=2)
fig.add_trace(go.Scatter(x=j_sweep, y=eta_ohm, name="eta_ohm",
              line=dict(color="#9C27B0"), showlegend=False), row=3, col=2)
fig.add_trace(go.Scatter(x=j_sweep, y=eta_conc, name="eta_conc",
              line=dict(color="#607D8B"), showlegend=False), row=3, col=2)

for r, c, xl, yl in [
    (1, 1, "Time (s)", "Temperature (K)"),
    (1, 2, "Time (s)", "Cell Voltage (V)"),
    (2, 1, "Time (s)", "Cell Voltage (V)"),
    (2, 2, "Time (s)", "Fuel Efficiency (-)"),
    (3, 1, "j (A/cm2)", "Cell Voltage (V)"),
    (3, 2, "j (A/cm2)", "Overpotential (V)"),
]:
    fig.update_xaxes(title_text=xl, row=r, col=c)
    fig.update_yaxes(title_text=yl, row=r, col=c)

fig.update_layout(
    title="<b>EC006 DMFC -- F2a Electrochemical + Methanol Crossover + Thermal ODE</b>",
    height=1000, template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

out_path = os.path.join(OUTPUT_DIR, "EC006_F2a_electrochemical_report.html")
fig.write_html(out_path, include_plotlyjs="cdn")
print(f"Report saved: {out_path}")
