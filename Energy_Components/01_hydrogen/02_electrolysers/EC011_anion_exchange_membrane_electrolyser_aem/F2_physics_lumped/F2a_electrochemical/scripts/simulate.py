"""
EC011 -- AEM Electrolyser -- F2a Electrochemical
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

# --- Scenario 1: constant load, thermal transient from cold start ---
r1 = model.predict({"current_density_A_cm2": 1.5, "T_cell_K": 300.0,
                    "P_h2_bar": 1.0, "P_o2_bar": 1.0, "dt": 2.0, "duration_s": 600.0})

# --- Scenario 2: current step ---
def step_j(t):
    return 0.5 if t < 120 else 2.0

r2 = m.simulate(step_j, 313.15, 1.0, 1.0, 2.0, 400.0)

# --- Scenario 3: polarization curves at several temperatures ---
j_sweep = np.linspace(0.02, 2.9, 200)
temps = [303.15, 323.15, 343.15, 353.15]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=[
        "Thermal Transient (j=1.5 A/cm2, cold start)",
        "Cell Voltage Transient",
        "Current Step Response (0.5->2.0 at t=120s)",
        "H2 Production Rate (step)",
        "Polarization Curves V(j) at various T",
        "Overpotential Breakdown (T=333K)",
    ],
    vertical_spacing=0.10, horizontal_spacing=0.10,
)

fig.add_trace(go.Scatter(x=r1["t"], y=r1["temperature"], name="T(t)",
              line=dict(color="#d62728")), row=1, col=1)
fig.add_trace(go.Scatter(x=r1["t"], y=r1["voltage"], name="V(t)",
              line=dict(color="#1f77b4")), row=1, col=2)
fig.add_trace(go.Scatter(x=r2["t"], y=r2["voltage"], name="V step",
              line=dict(color="#2ca02c"), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=r2["t"], y=np.array(r2["h2_rate_mol_s"]) * 1000.0,
              name="H2", line=dict(color="#ff7f0e"), showlegend=False), row=2, col=2)

for i, T in enumerate(temps):
    V_arr = [m.cell_voltage(j, T, 1.0, 1.0) for j in j_sweep]
    fig.add_trace(go.Scatter(x=j_sweep, y=V_arr, name=f"T={T-273.15:.0f}C",
                  line=dict(color=colors[i])), row=3, col=1)

E_arr = [m.reversible_voltage(333.15, 1.0, 1.0) for _ in j_sweep]
a_arr = [m.activation_anode(j, 333.15) for j in j_sweep]
c_arr = [m.activation_cathode(j, 333.15) for j in j_sweep]
o_arr = [m.ohmic_overpotential(j, 333.15) for j in j_sweep]
k_arr = [m.concentration_overpotential(j, 333.15) for j in j_sweep]
for arr, nm, col in [(E_arr, "E_rev", "#555"), (a_arr, "anode OER", "#FF5722"),
                     (c_arr, "cathode HER", "#795548"), (o_arr, "ohmic AEM", "#2196F3"),
                     (k_arr, "conc", "#9C27B0")]:
    fig.add_trace(go.Scatter(x=j_sweep, y=arr, name=nm,
                  line=dict(color=col), showlegend=False), row=3, col=2)

for r, c, xl, yl in [
    (1, 1, "Time (s)", "Temperature (K)"),
    (1, 2, "Time (s)", "Cell Voltage (V)"),
    (2, 1, "Time (s)", "Cell Voltage (V)"),
    (2, 2, "Time (s)", "H2 (mmol/s)"),
    (3, 1, "j (A/cm2)", "Cell Voltage (V)"),
    (3, 2, "j (A/cm2)", "Overpotential (V)"),
]:
    fig.update_xaxes(title_text=xl, row=r, col=c)
    fig.update_yaxes(title_text=yl, row=r, col=c)

fig.update_layout(
    title="<b>EC011 AEM Electrolyser -- F2a Full Electrochemical + Thermal ODE</b>",
    height=1000, template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

out_path = os.path.join(OUTPUT_DIR, "EC011_F2a_electrochemical_report.html")
fig.write_html(out_path, include_plotlyjs="cdn")
print(f"Report saved: {out_path}")
