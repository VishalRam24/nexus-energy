"""
EC067 -- Airborne Wind Energy (AWE) -- F2a Crosswind Pumping-Cycle
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
    print("plotly not installed -- skipping HTML report (pip install plotly).")
    sys.exit(0)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "model_files")
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = ComponentModel()
m = model._model

# Scenario 1: one cycle at 12 m/s
r = model.predict({"v_wind": 12.0})

# Scenario 2: power & Loyd limit vs wind speed
v_sweep = np.linspace(m.v_cut_in, m.v_cut_out, 60)
P_avg = np.array([m.simulate(v)["P_avg"] for v in v_sweep]) / 1000.0
P_loyd = np.array([m.loyd_power_limit(v) for v in v_sweep]) / 1000.0
P_out_avg = np.array([m.simulate(v)["P_out_avg"] for v in v_sweep]) / 1000.0

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        "Pumping cycle: tether length L(t) (12 m/s)",
        "Pumping cycle: electrical power P(t) (12 m/s)",
        "Cycle-avg power vs wind speed (with Loyd bound)",
        "Traction force F(t) over cycle (12 m/s)",
    ],
    vertical_spacing=0.12, horizontal_spacing=0.10,
)

fig.add_trace(go.Scatter(x=r["t"], y=r["L"], name="L(t)",
              line=dict(color="#1f77b4")), row=1, col=1)
fig.add_trace(go.Scatter(x=r["t"], y=r["P_elec"] / 1000.0, name="P(t)",
              line=dict(color="#2ca02c")), row=1, col=2)

fig.add_trace(go.Scatter(x=v_sweep, y=P_loyd, name="Loyd ideal limit",
              line=dict(color="#d62728", dash="dash")), row=2, col=1)
fig.add_trace(go.Scatter(x=v_sweep, y=P_out_avg, name="Reel-out avg",
              line=dict(color="#ff7f0e")), row=2, col=1)
fig.add_trace(go.Scatter(x=v_sweep, y=P_avg, name="Net cycle avg",
              line=dict(color="#2ca02c")), row=2, col=1)

fig.add_trace(go.Scatter(x=r["t"], y=r["F_traction"] / 1000.0, name="F(t)",
              line=dict(color="#9467bd"), showlegend=False), row=2, col=2)

for rr, cc, xl, yl in [
    (1, 1, "Time (s)", "Tether length (m)"),
    (1, 2, "Time (s)", "Electrical power (kW)"),
    (2, 1, "Wind speed (m/s)", "Power (kW)"),
    (2, 2, "Time (s)", "Traction force (kN)"),
]:
    fig.update_xaxes(title_text=xl, row=rr, col=cc)
    fig.update_yaxes(title_text=yl, row=rr, col=cc)

fig.update_layout(
    title="<b>EC067 AWE -- F2a Crosswind Pumping-Cycle (Loyd 1980 + tether ODE)</b>",
    height=850, template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

out_path = os.path.join(OUTPUT_DIR, "EC067_F2a_pumping_cycle_report.html")
fig.write_html(out_path, include_plotlyjs="cdn")
print(f"Report saved: {out_path}")
