"""
EC002 -- Solid Oxide Fuel Cell (SOFC) -- F1b Thermal
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
info = model.get_info()
j_max = info["inputs"]["current_density"]["range"][1]

j_vals = np.linspace(0.01, j_max * 0.90, 300)
temperatures = [973.0, 1023.0, 1073.0, 1173.0, 1273.0]
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
        "Power Density vs Current Density",
        "Efficiency vs Current Density",
        "ASR Components vs Temperature",
    ],
    vertical_spacing=0.14,
    horizontal_spacing=0.12,
)

for i, T in enumerate(temperatures):
    rows = data_by_T[T]
    valid_j = [j_vals[k] for k, r in enumerate(rows) if r is not None]
    v_cell = [r["cell_voltage_V"] for r in rows if r is not None]
    p_dens = [r["power_density_W_cm2"] for r in rows if r is not None]
    eta = [r["efficiency"] for r in rows if r is not None]

    label = f"T = {T:.0f} K ({T-273.15:.0f} C)"
    kw = dict(line=dict(color=colors[i]), name=label, legendgroup=label, showlegend=True)

    fig.add_trace(go.Scatter(x=valid_j, y=v_cell, **kw), row=1, col=1)
    fig.add_trace(go.Scatter(x=valid_j, y=p_dens, **{**kw, "showlegend": False}), row=1, col=2)
    fig.add_trace(go.Scatter(x=valid_j, y=eta, **{**kw, "showlegend": False}), row=2, col=1)

# Panel 4: ASR vs temperature
T_sweep = np.linspace(973, 1273, 100)
ohmic_asr = []
total_asr = []
for t in T_sweep:
    out = model.predict({"current_density": 0.5, "temperature": float(t)})
    ohmic_asr.append(out["ohmic_asr_ohm_cm2"])
    total_asr.append(out["asr_ohm_cm2"])

fig.add_trace(
    go.Scatter(x=T_sweep.tolist(), y=total_asr, mode="lines",
               line=dict(color="#d62728", width=2), name="Total ASR", showlegend=False),
    row=2, col=2,
)
fig.add_trace(
    go.Scatter(x=T_sweep.tolist(), y=ohmic_asr, mode="lines",
               line=dict(color="#1f77b4", width=2, dash="dash"), name="Ohmic ASR", showlegend=False),
    row=2, col=2,
)

fig.update_xaxes(title_text="Current Density (A/cm2)", row=1, col=1)
fig.update_yaxes(title_text="Cell Voltage (V)", row=1, col=1)
fig.update_xaxes(title_text="Current Density (A/cm2)", row=1, col=2)
fig.update_yaxes(title_text="Power Density (W/cm2)", row=1, col=2)
fig.update_xaxes(title_text="Current Density (A/cm2)", row=2, col=1)
fig.update_yaxes(title_text="Efficiency (-)", row=2, col=1)
fig.update_xaxes(title_text="Temperature (K)", row=2, col=2)
fig.update_yaxes(title_text="ASR (ohm cm2)", row=2, col=2)

fig.update_layout(
    title=dict(
        text=(
            "<b>EC002 -- Solid Oxide Fuel Cell -- F1b Thermal Model</b><br>"
            "<sup>YSZ Arrhenius conductivity, temperature-dependent electrode kinetics</sup>"
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
