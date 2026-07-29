"""
EC098 -- Organic Rankine Cycle (ORC) -- F2a Thermo Cycle Steady-State
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

# --- Scenario 1: Part-load sweep ---
loads = np.linspace(0.1, 1.0, 50)
W_net = []
eta_th = []
m_dots = []
for lf in loads:
    r = model.predict({"load_fraction": lf})
    W_net.append(r["W_net"] / 1000.0)
    eta_th.append(r["eta_thermal"])
    m_dots.append(r["m_dot"])

# --- Scenario 2: Evaporator pressure sweep ---
P_evaps = np.linspace(800000, 3000000, 50)
eta_P = []
W_P = []
for P in P_evaps:
    r = model.predict({"P_evap": P})
    eta_P.append(r["eta_thermal"])
    W_P.append(r["W_net"] / 1000.0)

# --- Scenario 3: T-s diagram at design point ---
r_design = model.predict({"load_fraction": 1.0})
sp = r_design["state_points"]
T_cycle = sp["T"] + [sp["T"][0]]
s_cycle = sp["s"] + [sp["s"][0]]

# --- Scenario 4: Dynamic load step ---
def step_load(t):
    if t < 600:
        return 1.0
    elif t < 1200:
        return 0.5
    else:
        return 0.8

r_dyn = model.predict({
    "mode": "dynamic",
    "load_fraction": step_load,
    "T_ambient_K": 293.15,
    "dt": 5.0,
    "duration_s": 1800.0,
})

fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=[
        "Part-Load: Net Power Output",
        "Part-Load: Thermal Efficiency",
        "Evaporator Pressure: Efficiency",
        "T-s Diagram (Design Point)",
        "Dynamic: Net Power (Load Step)",
        "Dynamic: Thermal Efficiency (Load Step)",
    ],
    vertical_spacing=0.10,
    horizontal_spacing=0.10,
)

# Row 1: Part-load
fig.add_trace(go.Scatter(x=loads * 100, y=W_net, name="W_net",
              line=dict(color="#1f77b4")), row=1, col=1)
fig.add_trace(go.Scatter(x=loads * 100, y=eta_th, name="eta_th",
              line=dict(color="#ff7f0e")), row=1, col=2)

# Row 2: Pressure sweep + T-s
fig.add_trace(go.Scatter(x=P_evaps / 1e6, y=eta_P, name="eta vs P_evap",
              line=dict(color="#2ca02c")), row=2, col=1)
fig.add_trace(go.Scatter(x=s_cycle, y=T_cycle, name="Cycle",
              mode="lines+markers", line=dict(color="#d62728")), row=2, col=2)

# Row 3: Dynamic
fig.add_trace(go.Scatter(x=r_dyn["t"] / 60, y=r_dyn["W_net"] / 1000, name="W_net(t)",
              line=dict(color="#9467bd")), row=3, col=1)
fig.add_trace(go.Scatter(x=r_dyn["t"] / 60, y=r_dyn["eta_thermal"], name="eta(t)",
              line=dict(color="#8c564b")), row=3, col=2)

for r, c, xl, yl in [
    (1, 1, "Load (%)", "Net Power (kW)"),
    (1, 2, "Load (%)", "Thermal Efficiency (-)"),
    (2, 1, "P_evap (MPa)", "Thermal Efficiency (-)"),
    (2, 2, "Entropy (J/kg.K)", "Temperature (K)"),
    (3, 1, "Time (min)", "Net Power (kW)"),
    (3, 2, "Time (min)", "Thermal Efficiency (-)"),
]:
    fig.update_xaxes(title_text=xl, row=r, col=c)
    fig.update_yaxes(title_text=yl, row=r, col=c)

fig.update_layout(
    title="<b>EC098 ORC -- F2a Thermodynamic Cycle Steady-State + Part-Load</b>",
    height=1000, template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

out_path = os.path.join(OUTPUT_DIR, "EC098_F2a_thermo_cycle_ss_report.html")
fig.write_html(out_path, include_plotlyjs="cdn")
print(f"Report saved: {out_path}")
