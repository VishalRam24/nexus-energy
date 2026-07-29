"""
EC191 -- Gas Compressor Station -- F2a Centrifugal Polytropic
Plotly HTML simulation report (optional; plotly import guarded).
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
    print("plotly not installed; skipping HTML report. (pip install plotly)")
    sys.exit(0)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "model_files")
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = ComponentModel()
m = model._model

# --- Scenario 1: compressor map (psi vs phi) with surge/choke band ---
phi = np.linspace(m.phi_surge * 0.7, m.phi_choke * 1.1, 200)
psi = np.array([m.head_coefficient(p) for p in phi])

# --- Scenario 2: pressure-ratio / power / T_disch vs mass flow ---
m_dot = np.linspace(40, 160, 60)
ops = [m.operating_point(md) for md in m_dot]
PR = [o["pressure_ratio"] for o in ops]
W = [o["shaft_power_MW"] for o in ops]
Td = [o["T_discharge_K"] for o in ops]

# --- Scenario 3: discharge pressure / temperature transient ---
r = model.predict({"mass_flow_kg_s": 100.0, "dt": 0.2, "duration_s": 180.0})

# --- Scenario 4: flow step (load change) ---
def step_flow(t):
    return 70.0 if t < 60.0 else 110.0

r2 = m.simulate(step_flow, 1.0, None, None, 0.2, 180.0)

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=("Compressor map psi(phi)", "PR & shaft power vs flow",
                    "Discharge P & T transient", "Flow-step response"),
    specs=[[{"secondary_y": False}, {"secondary_y": True}],
           [{"secondary_y": True}, {"secondary_y": True}]],
)

fig.add_trace(go.Scatter(x=phi, y=psi, name="psi(phi)"), row=1, col=1)
fig.add_vline(x=m.phi_surge, line_dash="dash", line_color="red", row=1, col=1)
fig.add_vline(x=m.phi_choke, line_dash="dash", line_color="orange", row=1, col=1)

fig.add_trace(go.Scatter(x=m_dot, y=PR, name="PR"), row=1, col=2)
fig.add_trace(go.Scatter(x=m_dot, y=W, name="Shaft MW"), row=1, col=2, secondary_y=True)

fig.add_trace(go.Scatter(x=r["t"], y=r["P_discharge_bar"], name="P_disch bar"), row=2, col=1)
fig.add_trace(go.Scatter(x=r["t"], y=r["T_discharge_K"], name="T_disch K"),
              row=2, col=1, secondary_y=True)

fig.add_trace(go.Scatter(x=r2["t"], y=r2["P_discharge_bar"], name="P (step)"), row=2, col=2)
fig.add_trace(go.Scatter(x=r2["t"], y=r2["mass_flow_kg_s"], name="m_dot (step)"),
              row=2, col=2, secondary_y=True)

fig.update_layout(title="EC191 Gas Compressor Station — F2a Centrifugal Polytropic",
                  height=800, showlegend=True)

out = os.path.join(OUTPUT_DIR, "simulation_report.html")
fig.write_html(out)
print(f"Wrote {out}")
