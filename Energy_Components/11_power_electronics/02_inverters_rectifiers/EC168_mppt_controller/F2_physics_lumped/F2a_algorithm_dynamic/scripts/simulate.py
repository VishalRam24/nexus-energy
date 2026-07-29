"""
EC168 -- MPPT Controller -- F2a Algorithm Dynamic
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
m = model._model

# --- Scenario 1: Steady 1000 W/m2 -- MPPT convergence ---
r1 = model.predict({
    "irradiance": 1000.0,
    "T_cell": 298.15,
    "dt": 0.001,
    "duration_s": 1.0,
})

# --- Scenario 2: Irradiance step 1000 -> 500 -> 1000 ---
def irr_step(t):
    if t < 0.5:
        return 1000.0
    elif t < 1.0:
        return 500.0
    else:
        return 1000.0

r2 = m.simulate(irr_step, 298.15, 0.001, 1.5)

# --- Scenario 3: Slow irradiance ramp ---
def irr_ramp(t):
    return 200.0 + 800.0 * min(t / 2.0, 1.0)

r3 = m.simulate(irr_ramp, 298.15, 0.001, 2.0)

# --- Scenario 4: PV I-V and P-V curves ---
pv = m.pv
V_sweep = np.linspace(0.1, 38.0, 300)
I_1000 = [pv.current(v, 1000.0) for v in V_sweep]
I_500 = [pv.current(v, 500.0) for v in V_sweep]
P_1000 = [v * pv.current(v, 1000.0) for v in V_sweep]
P_500 = [v * pv.current(v, 500.0) for v in V_sweep]

fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=[
        "MPPT Convergence (G=1000 W/m2)",
        "Tracking Efficiency",
        "Irradiance Step Response",
        "Power Tracking -- Step Response",
        "PV I-V Curves",
        "PV P-V Curves with MPP",
    ],
    vertical_spacing=0.10,
    horizontal_spacing=0.10,
)

# Row 1
fig.add_trace(go.Scatter(x=r1["t"], y=r1["V_ref"], name="V_ref",
              line=dict(color="#1f77b4")), row=1, col=1)
fig.add_trace(go.Scatter(x=r1["t"], y=r1["P_pv"], name="P_pv",
              line=dict(color="#2ca02c")), row=1, col=1)
fig.add_trace(go.Scatter(x=r1["t"], y=r1["tracking_efficiency"],
              name="eta_track", line=dict(color="#d62728")), row=1, col=2)

# Row 2
fig.add_trace(go.Scatter(x=r2["t"], y=r2["irradiance"], name="G",
              line=dict(color="#ff7f0e")), row=2, col=1)
fig.add_trace(go.Scatter(x=r2["t"], y=r2["V_ref"], name="V_ref step",
              line=dict(color="#1f77b4"), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=r2["t"], y=r2["P_pv"], name="P_pv step",
              line=dict(color="#2ca02c"), showlegend=False), row=2, col=2)
fig.add_trace(go.Scatter(x=r2["t"], y=r2["duty_cycle"], name="D step",
              line=dict(color="#9467bd")), row=2, col=2)

# Row 3
fig.add_trace(go.Scatter(x=V_sweep, y=I_1000, name="I-V 1000 W/m2",
              line=dict(color="#1f77b4")), row=3, col=1)
fig.add_trace(go.Scatter(x=V_sweep, y=I_500, name="I-V 500 W/m2",
              line=dict(color="#ff7f0e")), row=3, col=1)
fig.add_trace(go.Scatter(x=V_sweep, y=P_1000, name="P-V 1000",
              line=dict(color="#1f77b4"), showlegend=False), row=3, col=2)
fig.add_trace(go.Scatter(x=V_sweep, y=P_500, name="P-V 500",
              line=dict(color="#ff7f0e"), showlegend=False), row=3, col=2)

# Mark MPPs
V_m1, I_m1, P_m1 = pv.mpp(1000.0)
V_m5, I_m5, P_m5 = pv.mpp(500.0)
fig.add_trace(go.Scatter(x=[V_m1], y=[P_m1], mode="markers",
              marker=dict(size=12, color="red"), name="MPP 1000",
              showlegend=False), row=3, col=2)
fig.add_trace(go.Scatter(x=[V_m5], y=[P_m5], mode="markers",
              marker=dict(size=12, color="red"), name="MPP 500",
              showlegend=False), row=3, col=2)

for r, c, xl, yl in [
    (1, 1, "Time (s)", "V_ref (V) / P_pv (W)"),
    (1, 2, "Time (s)", "Tracking Efficiency (-)"),
    (2, 1, "Time (s)", "G (W/m2) / V_ref (V)"),
    (2, 2, "Time (s)", "P_pv (W) / Duty Cycle"),
    (3, 1, "Voltage (V)", "Current (A)"),
    (3, 2, "Voltage (V)", "Power (W)"),
]:
    fig.update_xaxes(title_text=xl, row=r, col=c)
    fig.update_yaxes(title_text=yl, row=r, col=c)

fig.update_layout(
    title="<b>EC168 MPPT Controller -- F2a P&O + Buck Converter</b>",
    height=1000, template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

out_path = os.path.join(OUTPUT_DIR, "EC168_F2a_algorithm_dynamic_report.html")
fig.write_html(out_path, include_plotlyjs="cdn")
print(f"Report saved: {out_path}")
