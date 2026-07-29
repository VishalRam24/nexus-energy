"""
EC013 -- LH2 Storage -- F1b Boil-Off Thermal
4-panel Plotly HTML simulation report.
"""

import sys
import os
import numpy as np
import json

sys.path.insert(0, os.path.dirname(__file__))

from predict import ComponentModel
from model import LH2ThermalModel

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("ERROR: plotly not installed. Run: pip install plotly")
    sys.exit(1)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..")
os.makedirs(OUTPUT_DIR, exist_ok=True)

cm = ComponentModel()
with open(os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")) as fh:
    raw = json.load(fh)
physics = LH2ThermalModel(raw)

# --- BOR vs T_amb at different fill fractions ---
T_amb_arr = np.linspace(233.15, 333.15, 200)
fill_fractions = [0.3, 0.5, 0.8, 0.95]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

# --- BOR vs fill at fixed T_amb ---
f_arr = np.linspace(0.1, 0.95, 200)

# --- Pressurization transient ---
t_pres, P_pres, m_pres, bor_pres = physics.pressurization_transient(
    fill_fraction=0.9, T_amb_K=298.15, P0_bar=1.0,
    t_span=(0, 10 * 86400), n_steps=400
)

# --- U_eff vs T_amb ---
U_arr = [float(physics.u_eff(T)) for T in T_amb_arr]

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        "Boil-Off Rate vs Ambient Temperature",
        "BOR vs Fill Fraction at T_amb = 298 K",
        "Closed-Vent Pressurization (10 days, f=0.9)",
        "Effective U vs Ambient Temperature (MLI T-dependence)",
    ],
    vertical_spacing=0.14,
    horizontal_spacing=0.12,
)

for i, f in enumerate(fill_fractions):
    bor_arr = [float(physics.boiloff_rate_percent_day(f, T)) for T in T_amb_arr]
    label = f"fill = {f:.0%}"
    fig.add_trace(
        go.Scatter(x=T_amb_arr.tolist(), y=bor_arr, mode="lines",
                   line=dict(color=colors[i]), name=label),
        row=1, col=1
    )

bor_vs_fill = [float(physics.boiloff_rate_percent_day(f, 298.15)) for f in f_arr]
fig.add_trace(
    go.Scatter(x=f_arr.tolist(), y=bor_vs_fill, mode="lines",
               line=dict(color="#2ca02c", width=2), name="T_amb=298K", showlegend=False),
    row=1, col=2
)

# Pressurization
fig.add_trace(
    go.Scatter(x=(t_pres / 86400).tolist(), y=P_pres.tolist(), mode="lines",
               line=dict(color="#d62728", width=2), name="Pressure", showlegend=False),
    row=2, col=1
)
fig.add_hline(y=float(physics.P_vent), row=2, col=1,
              line=dict(color="grey", dash="dash"), annotation_text="Vent P")

fig.add_trace(
    go.Scatter(x=T_amb_arr.tolist(), y=U_arr, mode="lines",
               line=dict(color="#9467bd", width=2), name="U_eff", showlegend=False),
    row=2, col=2
)

fig.update_xaxes(title_text="T_ambient (K)", row=1, col=1)
fig.update_yaxes(title_text="BOR (%/day)", row=1, col=1)
fig.update_xaxes(title_text="Fill Fraction (-)", row=1, col=2)
fig.update_yaxes(title_text="BOR (%/day)", row=1, col=2)
fig.update_xaxes(title_text="Time (days)", row=2, col=1)
fig.update_yaxes(title_text="Tank Pressure (bar)", row=2, col=1)
fig.update_xaxes(title_text="T_ambient (K)", row=2, col=2)
fig.update_yaxes(title_text="U_eff W/(m2.K)", row=2, col=2)

fig.update_layout(
    title=dict(
        text=(
            "<b>EC013 -- LH2 Storage -- F1b Boil-Off Thermal Model</b><br>"
            "<sup>MLI k(T), T_amb variation, closed-vent pressurization dynamics</sup>"
        ),
        x=0.5,
    ),
    height=800,
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

out_path = os.path.join(OUTPUT_DIR, "simulation_report.html")
fig.write_html(out_path, include_plotlyjs="cdn")
print(f"Report saved: {out_path}")
