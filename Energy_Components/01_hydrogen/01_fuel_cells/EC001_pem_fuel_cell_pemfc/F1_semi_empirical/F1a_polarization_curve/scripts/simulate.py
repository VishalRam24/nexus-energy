"""
EC001 — PEM Fuel Cell (PEMFC) — F1a Polarization Curve
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
info  = model.get_info()
j_L   = info["inputs"]["current_density"]["range"][1]

j_vals      = np.linspace(0.001, j_L * 0.98, 400)
temperatures = [50.0, 60.0, 70.0, 80.0]
colors       = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

data_by_T = {}
for T_C in temperatures:
    rows = []
    for j in j_vals:
        try:
            out = model.predict({"current_density": float(j), "temperature": T_C})
            rows.append(out)
        except Exception:
            rows.append(None)
    data_by_T[T_C] = rows

fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=[
        "Polarization Curve (Cell Voltage)",
        "Power Density Curve",
        "Stack Voltage",
        "Stack Power",
        "Voltage Efficiency",
        "Voltage Breakdown at 70 °C",
    ],
    vertical_spacing=0.12,
    horizontal_spacing=0.10,
)

for i, T_C in enumerate(temperatures):
    rows = data_by_T[T_C]
    valid_j   = [j_vals[k] for k, r in enumerate(rows) if r is not None]
    v_cell    = [r["cell_voltage_V"]      for r in rows if r]
    p_density = [r["power_density_W_cm2"] for r in rows if r]
    v_stack   = [r["stack_voltage_V"]     for r in rows if r]
    p_stack   = [r["stack_power_W"]       for r in rows if r]
    eta       = [r["efficiency"]          for r in rows if r]

    label = f"T = {T_C:.0f} °C"
    kw = dict(line=dict(color=colors[i]), name=label,
               legendgroup=label, showlegend=True)

    fig.add_trace(go.Scatter(x=valid_j, y=v_cell,    **kw), row=1, col=1)
    fig.add_trace(go.Scatter(x=valid_j, y=p_density, **{**kw, "showlegend": False}), row=1, col=2)
    fig.add_trace(go.Scatter(x=valid_j, y=v_stack,   **{**kw, "showlegend": False}), row=2, col=1)
    fig.add_trace(go.Scatter(x=valid_j, y=p_stack,   **{**kw, "showlegend": False}), row=2, col=2)
    fig.add_trace(go.Scatter(x=valid_j, y=eta,       **{**kw, "showlegend": False}), row=3, col=1)

# Voltage breakdown at 70 °C
j_bar  = [0.1, 0.3, 0.5, 0.8, 1.0, 1.2]
rows70 = data_by_T[70.0]
idx    = [np.argmin(np.abs(j_vals - jb)) for jb in j_bar]

valid_idx = [k for k in idx if rows70[k] is not None]
j_labels  = [str(round(j_vals[k], 2)) for k in valid_idx]

e_nernst = [rows70[k]["E_Nernst_V"] for k in valid_idx]
v_act    = [rows70[k]["V_act_V"]    for k in valid_idx]
v_ohm    = [rows70[k]["V_ohm_V"]    for k in valid_idx]
v_conc   = [rows70[k]["V_conc_V"]   for k in valid_idx]
v_cell_b = [rows70[k]["cell_voltage_V"] for k in valid_idx]

fig.add_trace(go.Bar(name="V_cell",   x=j_labels, y=v_cell_b, marker_color="#4CAF50",  showlegend=False), row=3, col=2)
fig.add_trace(go.Bar(name="V_act",    x=j_labels, y=v_act,    marker_color="#FF5722",  showlegend=False), row=3, col=2)
fig.add_trace(go.Bar(name="V_ohm",    x=j_labels, y=v_ohm,    marker_color="#2196F3",  showlegend=False), row=3, col=2)
fig.add_trace(go.Bar(name="V_conc",   x=j_labels, y=v_conc,   marker_color="#9C27B0",  showlegend=False), row=3, col=2)

fig.update_layout(barmode="stack")

axis_labels = [
    (1, 1, "Current Density (A/cm²)", "Cell Voltage (V)"),
    (1, 2, "Current Density (A/cm²)", "Power Density (W/cm²)"),
    (2, 1, "Current Density (A/cm²)", "Stack Voltage (V)"),
    (2, 2, "Current Density (A/cm²)", "Stack Power (W)"),
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
            f"<b>EC001 — PEM Fuel Cell (PEMFC) — F1a Polarization Curve</b><br>"
            f"<sup>N_cells={params['N_cells']}, T_ref=70°C, "
            f"pH2={params['pH2']} atm, pO2={params['pO2']} atm, "
            f"j_L={params['j_L']} A/cm²</sup>"
        ),
        x=0.5,
    ),
    height=1000,
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

out_path = os.path.join(OUTPUT_DIR, "EC001_polarization_curve_report.html")
fig.write_html(out_path, include_plotlyjs="cdn")
print(f"Report saved: {out_path}")
