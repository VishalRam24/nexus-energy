"""
EC222 — Betavoltaic Cell — F2a Single-Diode I-V with Beta-Flux Photocurrent and Decay ODE
Plotly HTML simulation report generator (optional — plotly import is guarded).
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
    print("plotly not installed — skipping HTML report. (pip install plotly)")
    sys.exit(0)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "model_files")
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = ComponentModel()

# Scenario 1: I-V and P-V curve at t=0
iv = model.iv_curve(0.0)

# Scenario 2: life-time decay of output power (over ~1.5 half-lives)
life = model.predict({"t_years": 150.0, "n_eval": 120})

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "I-V curve (t=0)", "P-V curve & MPP (t=0)",
        "Output power over isotope life", "Activity / fraction remaining",
    ),
)

fig.add_trace(go.Scatter(x=iv["V"], y=iv["I"] * 1e6, name="I-V"), row=1, col=1)
fig.update_xaxes(title_text="V [V]", row=1, col=1)
fig.update_yaxes(title_text="I [uA]", row=1, col=1)

fig.add_trace(go.Scatter(x=iv["V"], y=iv["P"] * 1e6, name="P-V"), row=1, col=2)
fig.add_trace(go.Scatter(x=[iv["V_mpp_V"]], y=[iv["P_mpp_W"] * 1e6],
                         mode="markers", name="MPP", marker=dict(size=10)),
              row=1, col=2)
fig.update_xaxes(title_text="V [V]", row=1, col=2)
fig.update_yaxes(title_text="P [uW]", row=1, col=2)

fig.add_trace(go.Scatter(x=life["t_years"], y=life["P_out_uW"], name="P_out"),
              row=2, col=1)
fig.update_xaxes(title_text="t [years]", row=2, col=1)
fig.update_yaxes(title_text="P_out [uW]", row=2, col=1)

fig.add_trace(go.Scatter(x=life["t_years"], y=life["fraction_remaining"],
                         name="fraction remaining"), row=2, col=2)
fig.update_xaxes(title_text="t [years]", row=2, col=2)
fig.update_yaxes(title_text="A(t)/A0 [-]", row=2, col=2)

fig.update_layout(title="EC222 Betavoltaic Cell — F2a Single-Diode + Decay",
                  height=800, showlegend=True)

out = os.path.join(OUTPUT_DIR, "simulation_report.html")
fig.write_html(out)
print(f"Report written to {out}")
