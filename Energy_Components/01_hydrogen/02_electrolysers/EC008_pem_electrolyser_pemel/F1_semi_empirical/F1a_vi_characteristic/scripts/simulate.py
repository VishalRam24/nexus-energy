"""
EC008 — PEM Electrolyser (PEMEL) — F1a V-I Characteristic
Plotly HTML simulation report generator.
"""

import sys
import os
import numpy as np
import json

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

# ---------------------------------------------------------------------------
# Build sweep data
# ---------------------------------------------------------------------------

model = ComponentModel()
info = model.get_info()

j_vals = np.linspace(0.01, 2.0, 300)
temperatures = [50.0, 60.0, 70.0, 80.0]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

data_by_T = {}
for T_C in temperatures:
    rows = []
    for j in j_vals:
        out = model.predict({"current_density": j, "temperature": T_C})
        rows.append(out)
    data_by_T[T_C] = rows

# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------

fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=[
        "V-I Characteristic (Cell Voltage)",
        "Stack Voltage vs Current Density",
        "Power Input vs Current Density",
        "Hydrogen Production Rate",
        "Efficiency vs Current Density",
        "Voltage Breakdown at 80 °C",
    ],
    vertical_spacing=0.12,
    horizontal_spacing=0.10,
)

for i, T_C in enumerate(temperatures):
    rows = data_by_T[T_C]
    j_arr    = j_vals
    v_cell   = [r["cell_voltage_V"]      for r in rows]
    v_stack  = [r["stack_voltage_V"]     for r in rows]
    power    = [r["power_W"]             for r in rows]
    h2_rate  = [r["hydrogen_rate_mol_s"] for r in rows]
    eta      = [r["efficiency"]          for r in rows]

    label = f"T = {T_C:.0f} °C"
    kw = dict(line=dict(color=colors[i]), name=label, legendgroup=label,
               showlegend=(True if i < 4 else False))

    fig.add_trace(go.Scatter(x=list(j_arr), y=v_cell,  **kw), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(j_arr), y=v_stack, **{**kw, "showlegend": False}), row=1, col=2)
    fig.add_trace(go.Scatter(x=list(j_arr), y=power,   **{**kw, "showlegend": False}), row=2, col=1)
    fig.add_trace(go.Scatter(x=list(j_arr), y=h2_rate, **{**kw, "showlegend": False}), row=2, col=2)
    fig.add_trace(go.Scatter(x=list(j_arr), y=eta,     **{**kw, "showlegend": False}), row=3, col=1)

# Voltage breakdown stacked bar at T=80°C, selected j points
j_bar = [0.25, 0.5, 1.0, 1.5, 2.0]
rows_80 = data_by_T[80.0]
idx = [np.argmin(np.abs(j_vals - jb)) for jb in j_bar]

e_rev_vals = [rows_80[k]["E_rev_V"] for k in idx]
v_act_vals = [rows_80[k]["V_act_V"] for k in idx]
v_ohm_vals = [rows_80[k]["V_ohm_V"] for k in idx]

fig.add_trace(go.Bar(name="E_rev", x=[str(j) for j in j_bar], y=e_rev_vals,
                     marker_color="#2196F3", showlegend=False), row=3, col=2)
fig.add_trace(go.Bar(name="V_act", x=[str(j) for j in j_bar], y=v_act_vals,
                     marker_color="#FF5722", showlegend=False), row=3, col=2)
fig.add_trace(go.Bar(name="V_ohm", x=[str(j) for j in j_bar], y=v_ohm_vals,
                     marker_color="#4CAF50", showlegend=False), row=3, col=2)

fig.update_layout(barmode="stack")

# Axis labels
axis_labels = [
    (1, 1, "Current Density (A/cm²)", "Cell Voltage (V)"),
    (1, 2, "Current Density (A/cm²)", "Stack Voltage (V)"),
    (2, 1, "Current Density (A/cm²)", "Power Input (W)"),
    (2, 2, "Current Density (A/cm²)", "H₂ Rate (mol/s)"),
    (3, 1, "Current Density (A/cm²)", "Efficiency (-)"),
    (3, 2, "Current Density (A/cm²)", "Voltage (V)"),
]
for row, col, xl, yl in axis_labels:
    fig.update_xaxes(title_text=xl, row=row, col=col)
    fig.update_yaxes(title_text=yl, row=row, col=col)

params = info["active_parameters"]
fig.update_layout(
    title=dict(
        text=(
            f"<b>EC008 — PEM Electrolyser (PEMEL) — F1a V-I Characteristic</b><br>"
            f"<sup>N_cells={params['N_cells']}, A={params['electrode_area']} cm², "
            f"R_mem={params['R_membrane']} Ω·cm², Nafion 117</sup>"
        ),
        x=0.5,
    ),
    height=1000,
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

out_path = os.path.join(OUTPUT_DIR, "EC008_vi_characteristic_report.html")
fig.write_html(out_path, include_plotlyjs="cdn")
print(f"Report saved: {out_path}")
