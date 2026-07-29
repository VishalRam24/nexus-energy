"""
EC074 -- Plate Heat Exchanger -- F2a Lumped Dynamic
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

# --- Scenario 1: Cold start with constant flows ---
r1 = model.predict({
    "m_dot_hot": 1.2, "m_dot_cold": 1.0,
    "T_hot_in": 353.15, "T_cold_in": 293.15,
    "T_hot_init": 293.15, "T_cold_init": 293.15,
    "dt": 1.0, "duration_s": 600.0,
})

# --- Scenario 2: Step change in hot flow rate ---
def step_flow(t):
    return 0.5 if t < 200 else 1.5

r2 = model._model.simulate(step_flow, 1.0, 353.15, 293.15, 330.0, 310.0, 1.0, 600.0)

# --- Scenario 3: Step change in hot inlet temperature ---
def step_temp(t):
    return 353.15 if t < 200 else 373.15

r3 = model._model.simulate(1.2, 1.0, step_temp, 293.15, 340.0, 310.0, 1.0, 600.0)

fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=[
        "Cold Start: Temperature Response",
        "Cold Start: Heat Transfer Rate & Effectiveness",
        "Flow Step (0.5->1.5 kg/s): Temperatures",
        "Flow Step: Q_transfer & Effectiveness",
        "Temperature Step (80->100 C): Temperatures",
        "Temperature Step: Q_transfer & Effectiveness",
    ],
    vertical_spacing=0.10,
    horizontal_spacing=0.10,
)

# Row 1: Cold start
fig.add_trace(go.Scatter(x=r1["t"], y=r1["T_hot_out"] - 273.15, name="T_hot_out",
              line=dict(color="#d62728")), row=1, col=1)
fig.add_trace(go.Scatter(x=r1["t"], y=r1["T_cold_out"] - 273.15, name="T_cold_out",
              line=dict(color="#1f77b4")), row=1, col=1)
fig.add_trace(go.Scatter(x=r1["t"], y=r1["Q_transfer"] / 1000, name="Q (kW)",
              line=dict(color="#ff7f0e")), row=1, col=2)
fig.add_trace(go.Scatter(x=r1["t"], y=r1["effectiveness"], name="epsilon",
              line=dict(color="#2ca02c"), yaxis="y2"), row=1, col=2)

# Row 2: Flow step
fig.add_trace(go.Scatter(x=r2["t"], y=r2["T_hot_out"] - 273.15, name="T_hot (flow step)",
              line=dict(color="#d62728"), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=r2["t"], y=r2["T_cold_out"] - 273.15, name="T_cold (flow step)",
              line=dict(color="#1f77b4"), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=r2["t"], y=r2["Q_transfer"] / 1000, name="Q (flow step)",
              line=dict(color="#ff7f0e"), showlegend=False), row=2, col=2)
fig.add_trace(go.Scatter(x=r2["t"], y=r2["effectiveness"], name="eps (flow step)",
              line=dict(color="#2ca02c"), showlegend=False), row=2, col=2)

# Row 3: Temperature step
fig.add_trace(go.Scatter(x=r3["t"], y=r3["T_hot_out"] - 273.15, name="T_hot (T step)",
              line=dict(color="#d62728"), showlegend=False), row=3, col=1)
fig.add_trace(go.Scatter(x=r3["t"], y=r3["T_cold_out"] - 273.15, name="T_cold (T step)",
              line=dict(color="#1f77b4"), showlegend=False), row=3, col=1)
fig.add_trace(go.Scatter(x=r3["t"], y=r3["Q_transfer"] / 1000, name="Q (T step)",
              line=dict(color="#ff7f0e"), showlegend=False), row=3, col=2)
fig.add_trace(go.Scatter(x=r3["t"], y=r3["effectiveness"], name="eps (T step)",
              line=dict(color="#2ca02c"), showlegend=False), row=3, col=2)

for r in range(1, 4):
    fig.update_xaxes(title_text="Time (s)", row=r, col=1)
    fig.update_xaxes(title_text="Time (s)", row=r, col=2)
    fig.update_yaxes(title_text="Temperature (C)", row=r, col=1)
    fig.update_yaxes(title_text="Q (kW) / Effectiveness (-)", row=r, col=2)

fig.update_layout(
    title="<b>EC074 Plate Heat Exchanger -- F2a Lumped Dynamic Model</b>",
    height=1000, template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

out_path = os.path.join(OUTPUT_DIR, "EC074_F2a_lumped_dynamic_report.html")
fig.write_html(out_path, include_plotlyjs="cdn")
print(f"Report saved: {out_path}")
