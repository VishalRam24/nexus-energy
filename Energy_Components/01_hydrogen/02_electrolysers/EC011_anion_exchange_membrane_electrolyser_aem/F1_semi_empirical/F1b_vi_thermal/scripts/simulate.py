"""
EC011 -- AEM Electrolyser -- F1b V-I Thermal
4-panel Plotly HTML simulation report.
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

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..")
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = ComponentModel()

j_vals = np.linspace(100.0, 18000.0, 300)   # A/m2
temperatures = [313.15, 323.15, 333.15, 343.15, 353.15]  # K
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

# Collect V-I and efficiency data at fixed T
data_by_T = {}
for T in temperatures:
    rows = []
    for j in j_vals:
        try:
            out = model.predict({"current_density": float(j), "temperature": T})
            rows.append(out)
        except Exception:
            rows.append(None)
    data_by_T[T] = rows

# Steady-state temperature vs current density
T_ss_list = []
for j in j_vals:
    try:
        out = model.predict({"current_density": float(j), "solve_thermal": True})
        T_ss_list.append(out["temperature_K"])
    except Exception:
        T_ss_list.append(None)

# Transient temperature at nominal operating point
from model import AEMThermalModel
import json
with open(os.path.join(os.path.dirname(__file__), "..", "data", "parameters.json")) as fh:
    raw = json.load(fh)
physics = AEMThermalModel(raw)
t_trans, T_trans = physics.transient_temperature(
    8000.0, 333.15, 333.15, (0, 3600), n_steps=500
)

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        "Polarization Curves (V-I) at Different Temperatures",
        "LHV Efficiency vs Current Density",
        "Steady-State Stack Temperature vs Load",
        "Transient Warm-Up: j = 8000 A/m2, T_cool = 333.15 K",
    ],
    vertical_spacing=0.14,
    horizontal_spacing=0.12,
)

for i, T in enumerate(temperatures):
    rows = data_by_T[T]
    valid_j = [j_vals[k] / 1e4 for k, r in enumerate(rows) if r is not None]  # convert to A/cm2
    v_cell = [r["cell_voltage_V"] for r in rows if r is not None]
    eta = [r["efficiency_lhv"] for r in rows if r is not None]

    label = f"T = {T-273.15:.0f} C"
    kw = dict(line=dict(color=colors[i]), name=label, legendgroup=label, showlegend=True)

    fig.add_trace(go.Scatter(x=valid_j, y=v_cell, **kw), row=1, col=1)
    fig.add_trace(go.Scatter(x=valid_j, y=eta, **{**kw, "showlegend": False}), row=1, col=2)

# Steady-state T
j_cm2_arr = [j / 1e4 for j in j_vals]
T_ss_clean = [t for t in T_ss_list if t is not None]
j_ss_clean = [j_cm2_arr[k] for k, t in enumerate(T_ss_list) if t is not None]

fig.add_trace(
    go.Scatter(x=j_ss_clean, y=T_ss_clean, mode="lines",
               line=dict(color="#e377c2", width=2), name="T_stack SS", showlegend=False),
    row=2, col=1,
)

# Transient
fig.add_trace(
    go.Scatter(x=(t_trans / 60).tolist(), y=T_trans.tolist(), mode="lines",
               line=dict(color="#8c564b", width=2), name="T_stack transient", showlegend=False),
    row=2, col=2,
)

fig.update_xaxes(title_text="Current Density (A/cm2)", row=1, col=1)
fig.update_yaxes(title_text="Cell Voltage (V)", row=1, col=1)
fig.update_xaxes(title_text="Current Density (A/cm2)", row=1, col=2)
fig.update_yaxes(title_text="LHV Efficiency (-)", row=1, col=2)
fig.update_xaxes(title_text="Current Density (A/cm2)", row=2, col=1)
fig.update_yaxes(title_text="Stack Temperature (K)", row=2, col=1)
fig.update_xaxes(title_text="Time (min)", row=2, col=2)
fig.update_yaxes(title_text="Stack Temperature (K)", row=2, col=2)

fig.update_layout(
    title=dict(
        text=(
            "<b>EC011 -- AEM Electrolyser -- F1b V-I Thermal Model</b><br>"
            "<sup>Arrhenius j0(T), ASR(T), E_rev(T), thermal balance with coolant</sup>"
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
