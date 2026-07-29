"""
EC016 -- H2 Compressor -- F1b Polytropic Thermal
4-panel Plotly HTML simulation report.
"""

import sys
import os
import numpy as np
import json

sys.path.insert(0, os.path.dirname(__file__))

from predict import ComponentModel
from model import H2CompressorThermalModel

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("ERROR: plotly not installed. Run: pip install plotly")
    sys.exit(1)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..")
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")) as fh:
    raw = json.load(fh)
physics = H2CompressorThermalModel(raw)
cm = ComponentModel()

# --- SEC vs T_inlet at different outlet pressures ---
T_arr = np.linspace(263.15, 333.15, 100)
P_outs = [200.0, 500.0, 900.0]
colors_p = ["#1f77b4", "#ff7f0e", "#2ca02c"]

# --- Stage discharge temperatures ---
T_inlets = [273.15, 298.15, 323.15]
colors_t = ["#1f77b4", "#d62728", "#2ca02c"]

# --- SEC vs intercooler effectiveness ---
eps_arr = np.linspace(0.0, 1.0, 100)

# --- Heat rejected vs T_inlet ---
Q_arr = [physics.heat_rejected_kw(0.014, 20.0, 900.0, T_inlet=float(T)) for T in T_arr]

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        "SEC vs Inlet Temperature at Different Outlet Pressures",
        "Stage Temperature Profile (P_in=20, P_out=900 bar)",
        "SEC vs Intercooler Effectiveness (T_inlet=298 K)",
        "Heat Rejected in Intercoolers vs Inlet Temperature",
    ],
    vertical_spacing=0.14,
    horizontal_spacing=0.12,
)

# Panel 1
for i, P_out in enumerate(P_outs):
    sec_arr = [physics.sec_kwh_per_kg(20.0, P_out, T_inlet=float(T)) for T in T_arr]
    fig.add_trace(
        go.Scatter(x=T_arr.tolist(), y=sec_arr, mode="lines",
                   line=dict(color=colors_p[i]), name=f"P_out={P_out:.0f} bar"),
        row=1, col=1
    )

# Panel 2: bar chart of stage temperatures for 3 inlet temperatures
stage_nums = list(range(1, physics.N + 1))
for i, T_in in enumerate(T_inlets):
    prof = physics.stage_temperature_profile(20.0, 900.0, T_inlet=T_in)
    T_disc = prof["T_discharge"].tolist()
    T_ic   = prof["T_after_ic"].tolist()
    label = f"T_in={T_in-273.15:.0f} C"
    fig.add_trace(
        go.Bar(x=stage_nums, y=T_disc, name=label + " T_disc",
               marker_color=colors_t[i], opacity=0.7, showlegend=True,
               legendgroup=label),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=stage_nums, y=T_ic, mode="markers+lines",
                   marker=dict(color=colors_t[i], size=8, symbol="circle-open"),
                   line=dict(color=colors_t[i], dash="dot"),
                   name=label + " after IC", showlegend=False, legendgroup=label),
        row=1, col=2
    )

# Panel 3
sec_vs_eps = [physics.sec_kwh_per_kg(20.0, 900.0, eps_ic=float(e)) for e in eps_arr]
fig.add_trace(
    go.Scatter(x=eps_arr.tolist(), y=sec_vs_eps, mode="lines",
               line=dict(color="#9467bd", width=2), name="SEC vs eps", showlegend=False),
    row=2, col=1
)

# Panel 4
fig.add_trace(
    go.Scatter(x=T_arr.tolist(), y=Q_arr, mode="lines",
               line=dict(color="#8c564b", width=2), name="Q_rejected", showlegend=False),
    row=2, col=2
)

fig.update_xaxes(title_text="T_inlet (K)", row=1, col=1)
fig.update_yaxes(title_text="SEC (kWh/kg)", row=1, col=1)
fig.update_xaxes(title_text="Stage Number", row=1, col=2)
fig.update_yaxes(title_text="Temperature (K)", row=1, col=2)
fig.update_xaxes(title_text="Intercooler Effectiveness (-)", row=2, col=1)
fig.update_yaxes(title_text="SEC (kWh/kg)", row=2, col=1)
fig.update_xaxes(title_text="T_inlet (K)", row=2, col=2)
fig.update_yaxes(title_text="Heat Rejected (kW)", row=2, col=2)

fig.update_layout(
    title=dict(
        text=(
            "<b>EC016 -- H2 Compressor -- F1b Polytropic Thermal Model</b><br>"
            "<sup>T_inlet variation, intercooler effectiveness, per-stage discharge temperature</sup>"
        ),
        x=0.5,
    ),
    height=800,
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    barmode="group",
)

out_path = os.path.join(OUTPUT_DIR, "simulation_report.html")
fig.write_html(out_path, include_plotlyjs="cdn")
print(f"Report saved: {out_path}")
