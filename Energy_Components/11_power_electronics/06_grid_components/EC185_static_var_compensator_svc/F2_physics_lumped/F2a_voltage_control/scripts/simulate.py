"""
EC185 -- Static VAR Compensator (SVC) -- F2a Physics-Lumped Voltage Control
Plotly HTML simulation report generator (optional; safe if plotly absent).
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
    print("plotly not installed -- skipping HTML report (pip install plotly).")
    sys.exit(0)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "model_files")
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = ComponentModel()
m = model._model

# --- Scenario 1: over-voltage step (E=1.05) regulation transient ---
r_ov = model.predict({"E_thev": 1.05, "X_thev": 0.10, "dt": 0.001, "duration_s": 0.6})
# --- Scenario 2: under-voltage step (E=0.95) regulation transient ---
r_uv = model.predict({"E_thev": 0.95, "X_thev": 0.10, "dt": 0.001, "duration_s": 0.6})

# --- Scenario 3: B_L(alpha) characteristic ---
alpha = np.linspace(90.0, 180.0, 200)
B_L = m.tcr_susceptance(np.radians(alpha))
B_svc = m.net_susceptance(np.radians(alpha))

# --- Scenario 4: V-Q droop characteristic (steady state vs source voltage) ---
E_sweep = np.linspace(0.90, 1.10, 60)
V_ss, Q_ss = [], []
for E in E_sweep:
    B, V, _ = m.steady_state_susceptance(E, 0.10)
    V_ss.append(V)
    Q_ss.append(B * V ** 2 * m.S_base)

fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=[
        "Over-voltage Regulation: V_bus(t)  (E=1.05)",
        "Reactive Output Q(t)  (E=1.05, inductive)",
        "Under-voltage Regulation: V_bus(t)  (E=0.95)",
        "Reactive Output Q(t)  (E=0.95, capacitive)",
        "TCR Susceptance B_L(alpha) & Net B_svc(alpha)",
        "V-Q Droop Steady-State Characteristic",
    ],
    vertical_spacing=0.10, horizontal_spacing=0.10,
)

fig.add_trace(go.Scatter(x=r_ov["t"], y=r_ov["V_bus"], name="V_bus OV",
              line=dict(color="#d62728")), row=1, col=1)
fig.add_trace(go.Scatter(x=r_ov["t"], y=r_ov["Q_MVAR"], name="Q OV",
              line=dict(color="#1f77b4")), row=1, col=2)
fig.add_trace(go.Scatter(x=r_uv["t"], y=r_uv["V_bus"], name="V_bus UV",
              line=dict(color="#2ca02c")), row=2, col=1)
fig.add_trace(go.Scatter(x=r_uv["t"], y=r_uv["Q_MVAR"], name="Q UV",
              line=dict(color="#ff7f0e")), row=2, col=2)
fig.add_trace(go.Scatter(x=alpha, y=B_L, name="B_L(alpha)",
              line=dict(color="#9467bd")), row=3, col=1)
fig.add_trace(go.Scatter(x=alpha, y=B_svc, name="B_svc(alpha)",
              line=dict(color="#17becf")), row=3, col=1)
fig.add_trace(go.Scatter(x=Q_ss, y=V_ss, name="V-Q droop",
              line=dict(color="#8c564b")), row=3, col=2)

for r, c, xl, yl in [
    (1, 1, "Time (s)", "V_bus (pu)"),
    (1, 2, "Time (s)", "Q (MVAR)"),
    (2, 1, "Time (s)", "V_bus (pu)"),
    (2, 2, "Time (s)", "Q (MVAR)"),
    (3, 1, "Firing angle alpha (deg)", "Susceptance (pu)"),
    (3, 2, "Q (MVAR)", "V_bus (pu)"),
]:
    fig.update_xaxes(title_text=xl, row=r, col=c)
    fig.update_yaxes(title_text=yl, row=r, col=c)

fig.update_layout(
    title="<b>EC185 SVC -- F2a Physics-Lumped TCR/TSC Voltage-Control (V-Q Droop ODE)</b>",
    height=1000, template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

out_path = os.path.join(OUTPUT_DIR, "EC185_F2a_voltage_control_report.html")
fig.write_html(out_path, include_plotlyjs="cdn")
print(f"Report saved: {out_path}")
