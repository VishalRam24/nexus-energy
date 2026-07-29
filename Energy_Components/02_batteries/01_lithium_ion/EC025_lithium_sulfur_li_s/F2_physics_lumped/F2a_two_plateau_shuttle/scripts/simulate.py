"""
EC025 -- Lithium-Sulfur Battery (Li-S) -- F2a Two-Plateau + Shuttle + Thermal
Optional Plotly HTML simulation report. Plotly import wrapped so absence is harmless.
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
    print("plotly not installed; skipping HTML report. (pip install plotly)")
    sys.exit(0)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "model_files")
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = ComponentModel()
m = model._model

# --- Scenario 1: 1A discharge from full ---
r1 = model.predict({"current_A": 1.0, "soc0": 1.0, "T0": 298.15, "dt": 10.0, "duration_s": 7200.0})

# --- Scenario 2: static OCV(SOC) two-plateau shape ---
soc = np.linspace(0.0, 1.0, 300)
ocv = m.ocv(soc)
i_sh = np.array([m.shuttle_current(s, 298.15) for s in soc])

# --- Scenario 3: rest self-discharge ---
r3 = model.predict({"current_A": 0.0, "soc0": 0.9, "dt": 60.0, "duration_s": 7200.0})

fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=[
        "Two-Plateau OCV vs SOC",
        "Polysulfide Shuttle Current vs SOC",
        "Discharge Voltage vs SOC (1A)",
        "Coulombic Efficiency vs time (1A)",
        "Temperature transient (1A)",
        "Self-discharge at rest (I=0)",
    ],
    vertical_spacing=0.10, horizontal_spacing=0.10,
)

fig.add_trace(go.Scatter(x=soc, y=ocv, name="OCV", line=dict(color="#1f77b4")), row=1, col=1)
fig.add_trace(go.Scatter(x=soc, y=i_sh, name="I_shuttle", line=dict(color="#d62728")), row=1, col=2)
fig.add_trace(go.Scatter(x=r1["soc"], y=r1["voltage"], name="V discharge",
              line=dict(color="#2ca02c")), row=2, col=1)
fig.add_trace(go.Scatter(x=r1["t"], y=r1["coulombic_efficiency"], name="eta_C",
              line=dict(color="#9467bd")), row=2, col=2)
fig.add_trace(go.Scatter(x=r1["t"], y=r1["temperature"], name="T(t)",
              line=dict(color="#ff7f0e")), row=3, col=1)
fig.add_trace(go.Scatter(x=r3["t"], y=r3["soc"], name="SOC rest",
              line=dict(color="#8c564b")), row=3, col=2)

for r, c, xl, yl in [
    (1, 1, "SOC", "OCV (V)"),
    (1, 2, "SOC", "Shuttle current (A)"),
    (2, 1, "SOC", "Voltage (V)"),
    (2, 2, "Time (s)", "Coulombic efficiency"),
    (3, 1, "Time (s)", "Temperature (K)"),
    (3, 2, "Time (s)", "SOC"),
]:
    fig.update_xaxes(title_text=xl, row=r, col=c)
    fig.update_yaxes(title_text=yl, row=r, col=c)

fig.update_layout(
    title="<b>EC025 Li-S -- F2a Two-Plateau + Polysulfide Shuttle + Thermal ODE</b>",
    height=1000, template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

out_path = os.path.join(OUTPUT_DIR, "EC025_F2a_two_plateau_shuttle_report.html")
fig.write_html(out_path, include_plotlyjs="cdn")
print(f"Report saved: {out_path}")
