"""
EC106 -- Fuel Cell CHP (SOFC-Based) -- F2a SOFC Cogeneration
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
    print("ERROR: plotly not installed. Run: pip install plotly")
    sys.exit(0)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "model_files")
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = ComponentModel()
m = model._model

# --- Scenario 1: cold(ish)-start thermal + CHP transient ---
r1 = model.predict({"current_density_A_cm2": 0.6, "T_cell_K": 1000.0,
                    "dt": 10.0, "duration_s": 3000.0})

# --- Scenario 2: load step ---
def step_j(t):
    return 0.3 if t < 1000 else 0.8

r2 = m.simulate(step_j, 1073.15, dt=10.0, duration_s=2000.0)

# --- Scenario 3: CHP efficiency vs load (at T=1073 K) ---
j_sweep = np.linspace(0.05, 1.8, 120)
eta_e = [m.power_and_heat(j, 1073.15)["eta_electrical"] for j in j_sweep]
eta_th = [m.power_and_heat(j, 1073.15)["eta_thermal"] for j in j_sweep]
eta_tot = [m.power_and_heat(j, 1073.15)["eta_total"] for j in j_sweep]
V_pol = [m.cell_voltage(j, 1073.15) for j in j_sweep]
p2h = [m.power_and_heat(j, 1073.15)["power_to_heat"] for j in j_sweep]

fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=[
        "Stack Thermal Transient (j=0.6, start 1000 K)",
        "Electrical & Useful-Thermal Power (transient)",
        "Load-Step Response: Cell Voltage",
        "Load-Step Response: Power-to-Heat",
        "Polarization Curve (T=800 C)",
        "CHP Efficiencies vs Load",
    ],
    vertical_spacing=0.10, horizontal_spacing=0.10,
)

fig.add_trace(go.Scatter(x=r1["t"], y=r1["temperature"], name="T(t)",
              line=dict(color="#d62728")), row=1, col=1)
fig.add_trace(go.Scatter(x=r1["t"], y=r1["P_e_kW"], name="P_e [kW]",
              line=dict(color="#1f77b4")), row=1, col=2)
fig.add_trace(go.Scatter(x=r1["t"], y=r1["Q_useful_thermal_kW"], name="Q_th [kW]",
              line=dict(color="#ff7f0e")), row=1, col=2)
fig.add_trace(go.Scatter(x=r2["t"], y=r2["voltage"], name="V step",
              line=dict(color="#2ca02c"), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=r2["t"], y=r2["power_to_heat"], name="P/H step",
              line=dict(color="#9467bd"), showlegend=False), row=2, col=2)
fig.add_trace(go.Scatter(x=j_sweep, y=V_pol, name="V_cell",
              line=dict(color="#17becf"), showlegend=False), row=3, col=1)
fig.add_trace(go.Scatter(x=j_sweep, y=eta_e, name="eta_e",
              line=dict(color="#1f77b4")), row=3, col=2)
fig.add_trace(go.Scatter(x=j_sweep, y=eta_th, name="eta_th",
              line=dict(color="#ff7f0e")), row=3, col=2)
fig.add_trace(go.Scatter(x=j_sweep, y=eta_tot, name="eta_total",
              line=dict(color="#2ca02c")), row=3, col=2)

for r, c, xl, yl in [
    (1, 1, "Time (s)", "Temperature (K)"),
    (1, 2, "Time (s)", "Power (kW)"),
    (2, 1, "Time (s)", "Cell Voltage (V)"),
    (2, 2, "Time (s)", "Power-to-Heat (-)"),
    (3, 1, "j (A/cm2)", "Cell Voltage (V)"),
    (3, 2, "j (A/cm2)", "Efficiency (-)"),
]:
    fig.update_xaxes(title_text=xl, row=r, col=c)
    fig.update_yaxes(title_text=yl, row=r, col=c)

fig.update_layout(
    title="<b>EC106 SOFC-CHP -- F2a Electrochemical Stack + Cogeneration + Thermal ODE</b>",
    height=1000, template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

out_path = os.path.join(OUTPUT_DIR, "EC106_F2a_sofc_cogen_report.html")
fig.write_html(out_path, include_plotlyjs="cdn")
print(f"Report saved: {out_path}")
