"""
EC014 -- Metal Hydride H2 Storage -- F2a Kinetics + Thermal
Plotly HTML simulation report generator (optional; plotly wrapped in try/except).
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
    sys.exit(0)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "model_files")
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = ComponentModel()
m = model._model

# --- Scenario 1: Absorption (charging) from empty, cold bed ---
r1 = model.predict({"P_supply_bar": 20.0, "T_bed_K": 293.15, "X0": 0.0,
                    "dt": 5.0, "duration_s": 1800.0})

# --- Scenario 2: Desorption (discharge) from full bed ---
r2 = model.predict({"P_supply_bar": 0.2, "T_bed_K": 313.15, "X0": 6.0,
                    "dt": 5.0, "duration_s": 1800.0})

# --- Scenario 3: van't Hoff plateau pressure vs temperature ---
T_sweep = np.linspace(273.15, 363.15, 100)
P_abs = [m.plateau_pressure(T, "absorption") for T in T_sweep]
P_des = [m.plateau_pressure(T, "desorption") for T in T_sweep]

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        "Absorption: H/M ratio & SOC (P=20 bar, cold start)",
        "Absorption: Bed Temperature (exothermic)",
        "Desorption: H/M ratio (P=0.2 bar, full)",
        "van't Hoff Plateau Pressure vs T",
    ],
    vertical_spacing=0.13, horizontal_spacing=0.10,
    specs=[[{"secondary_y": True}, {}], [{}, {}]],
)

fig.add_trace(go.Scatter(x=r1["t"], y=r1["HM_ratio"], name="H/M",
              line=dict(color="#1f77b4")), row=1, col=1)
fig.add_trace(go.Scatter(x=r1["t"], y=r1["soc"], name="SOC",
              line=dict(color="#2ca02c", dash="dot")), row=1, col=1, secondary_y=True)
fig.add_trace(go.Scatter(x=r1["t"], y=r1["temperature"], name="T(t)",
              line=dict(color="#d62728"), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=r2["t"], y=r2["HM_ratio"], name="H/M discharge",
              line=dict(color="#ff7f0e"), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=T_sweep - 273.15, y=P_abs, name="P_abs",
              line=dict(color="#9467bd")), row=2, col=2)
fig.add_trace(go.Scatter(x=T_sweep - 273.15, y=P_des, name="P_des",
              line=dict(color="#8c564b")), row=2, col=2)

fig.update_xaxes(title_text="Time (s)", row=1, col=1)
fig.update_xaxes(title_text="Time (s)", row=1, col=2)
fig.update_xaxes(title_text="Time (s)", row=2, col=1)
fig.update_xaxes(title_text="Temperature (C)", row=2, col=2)
fig.update_yaxes(title_text="H/M ratio", row=1, col=1)
fig.update_yaxes(title_text="SOC", row=1, col=1, secondary_y=True)
fig.update_yaxes(title_text="Temperature (K)", row=1, col=2)
fig.update_yaxes(title_text="H/M ratio", row=2, col=1)
fig.update_yaxes(title_text="Plateau Pressure (bar)", type="log", row=2, col=2)

fig.update_layout(
    title="<b>EC014 Metal Hydride H2 Storage -- F2a Kinetics + Thermal ODE</b>",
    height=850, template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

out_path = os.path.join(OUTPUT_DIR, "EC014_F2a_kinetics_thermal_report.html")
fig.write_html(out_path, include_plotlyjs="cdn")
print(f"Report saved: {out_path}")
