"""
EC187 -- HVDC Converter Station -- F2a Physics-Lumped (LCC + DC-link ODE)
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
    print("plotly not installed -- skipping HTML report. Run: pip install plotly")
    sys.exit(0)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "model_files")
os.makedirs(OUTPUT_DIR, exist_ok=True)

cm = ComponentModel()
m = cm._model

# --- Scenario 1: cold-start energisation to rated power ---
r1 = cm.predict({"P_order_MW": 1000.0, "Id0_kA": 0.0, "dt": 1e-3, "duration_s": 0.5})

# --- Scenario 2: power-order step (500 -> 1000 MW) via firing-angle schedule ---
a_lo = m.alpha_for_power(500e6)
a_hi = m.alpha_for_power(1000e6)
def alpha_step(t):
    return a_lo if t < 0.25 else a_hi
r2 = m.simulate(alpha_step, Id0=m.steady_state_current(alpha=a_lo),
                dt=1e-3, duration_s=0.5)

# --- Scenario 3: steady-state sweeps over firing angle ---
alphas = np.deg2rad(np.linspace(5, 60, 120))
P_sw, eta_sw, Q_sw, Id_sw = [], [], [], []
for a in alphas:
    Id = m.steady_state_current(alpha=a)
    pb = m.power_balance(Id, alpha=a)
    P_sw.append(pb["P_dc_rect_W"] / 1e6)
    eta_sw.append(pb["efficiency"])
    Q_sw.append(pb["Q_rect_VAR"] / 1e6)
    Id_sw.append(Id / 1e3)
adeg = np.rad2deg(alphas)

fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=[
        "DC-link current energisation (P=1000 MW order)",
        "Rectifier / inverter DC voltage (energisation)",
        "Power-order step 500->1000 MW at t=0.25 s",
        "Efficiency during step",
        "Steady-state power & current vs firing angle",
        "Reactive consumption & efficiency vs firing angle",
    ],
    vertical_spacing=0.10, horizontal_spacing=0.10,
)

fig.add_trace(go.Scatter(x=r1["t"], y=r1["Id_kA"], name="Id(t)",
              line=dict(color="#d62728")), row=1, col=1)
fig.add_trace(go.Scatter(x=r1["t"], y=r1["Vd_rect_kV"], name="Vd_rect",
              line=dict(color="#1f77b4")), row=1, col=2)
fig.add_trace(go.Scatter(x=r1["t"], y=r1["Vd_inv_kV"], name="Vd_inv",
              line=dict(color="#ff7f0e")), row=1, col=2)

fig.add_trace(go.Scatter(x=r2["t"], y=r2["Id_A"] / 1e3, name="Id step",
              line=dict(color="#2ca02c"), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=r2["t"], y=r2["efficiency"], name="eta step",
              line=dict(color="#9467bd"), showlegend=False), row=2, col=2)

fig.add_trace(go.Scatter(x=adeg, y=P_sw, name="P (MW)",
              line=dict(color="#1f77b4")), row=3, col=1)
fig.add_trace(go.Scatter(x=adeg, y=Id_sw, name="Id (kA)",
              line=dict(color="#d62728"), yaxis="y2"), row=3, col=1)
fig.add_trace(go.Scatter(x=adeg, y=Q_sw, name="Q_rect (MVAR)",
              line=dict(color="#ff7f0e")), row=3, col=2)
fig.add_trace(go.Scatter(x=adeg, y=eta_sw, name="efficiency",
              line=dict(color="#2ca02c"), showlegend=False), row=3, col=2)

for r, c, xl, yl in [
    (1, 1, "Time (s)", "Id (kA)"),
    (1, 2, "Time (s)", "Vd (kV)"),
    (2, 1, "Time (s)", "Id (kA)"),
    (2, 2, "Time (s)", "Efficiency (-)"),
    (3, 1, "Firing angle alpha (deg)", "P (MW) / Id (kA)"),
    (3, 2, "Firing angle alpha (deg)", "Q (MVAR) / eta"),
]:
    fig.update_xaxes(title_text=xl, row=r, col=c)
    fig.update_yaxes(title_text=yl, row=r, col=c)

fig.update_layout(
    title="<b>EC187 HVDC Converter Station -- F2a LCC 12-pulse + DC-link ODE</b>",
    height=1000, template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

out_path = os.path.join(OUTPUT_DIR, "EC187_F2a_lcc_dc_link_report.html")
fig.write_html(out_path, include_plotlyjs="cdn")
print(f"Report saved: {out_path}")
