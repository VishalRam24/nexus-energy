"""
EC009 -- Alkaline Electrolyser (AEL) -- F1b Thermal
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

j_vals = np.linspace(100, 4000, 300)  # A/m2
temperatures = [333.0, 343.0, 353.0, 363.0, 373.0]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

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

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        "Polarization Curves at Different Temperatures",
        "Power Consumption vs Current Density",
        "Efficiency vs Current Density",
        "KOH Conductivity vs Temperature",
    ],
    vertical_spacing=0.14,
    horizontal_spacing=0.12,
)

for i, T in enumerate(temperatures):
    rows = data_by_T[T]
    valid_j = [j_vals[k] for k, r in enumerate(rows) if r is not None]
    v_cell = [r["cell_voltage_V"] for r in rows if r is not None]
    p_cons = [r["power_consumption_kW"] for r in rows if r is not None]
    eta = [r["efficiency"] for r in rows if r is not None]

    label = f"T = {T:.0f} K ({T-273.15:.0f} C)"
    kw = dict(line=dict(color=colors[i]), name=label, legendgroup=label, showlegend=True)

    fig.add_trace(go.Scatter(x=valid_j, y=v_cell, **kw), row=1, col=1)
    fig.add_trace(go.Scatter(x=valid_j, y=p_cons, **{**kw, "showlegend": False}), row=1, col=2)
    fig.add_trace(go.Scatter(x=valid_j, y=eta, **{**kw, "showlegend": False}), row=2, col=1)

# Panel 4: KOH conductivity vs T
T_sweep = np.linspace(333, 373, 100)
sigma_vals = [model.predict({"current_density": 1000.0, "temperature": float(t)})["koh_conductivity_S_cm"]
              for t in T_sweep]

fig.add_trace(
    go.Scatter(x=T_sweep.tolist(), y=sigma_vals, mode="lines",
               line=dict(color="#9467bd", width=2), name="sigma_KOH", showlegend=False),
    row=2, col=2,
)

fig.update_xaxes(title_text="Current Density (A/m2)", row=1, col=1)
fig.update_yaxes(title_text="Cell Voltage (V)", row=1, col=1)
fig.update_xaxes(title_text="Current Density (A/m2)", row=1, col=2)
fig.update_yaxes(title_text="Power Consumption (kW)", row=1, col=2)
fig.update_xaxes(title_text="Current Density (A/m2)", row=2, col=1)
fig.update_yaxes(title_text="Efficiency (-)", row=2, col=1)
fig.update_xaxes(title_text="Temperature (K)", row=2, col=2)
fig.update_yaxes(title_text="KOH Conductivity (S/cm)", row=2, col=2)

fig.update_layout(
    title=dict(
        text=(
            "<b>EC009 -- Alkaline Electrolyser -- F1b Thermal Model</b><br>"
            "<sup>KOH conductivity, bubble coverage, Arrhenius electrode kinetics</sup>"
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
