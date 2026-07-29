"""
EC193 -- Methanation Reactor (Power-to-Gas) -- F2a Kinetics + Equilibrium
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

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..")
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = ComponentModel()
m = model._model

# --- Scenario 1: Startup transient ---
r1 = model.predict({
    "T0_K": 523.15,
    "duration_s": 600.0,
    "dt": 1.0,
})

# --- Scenario 2: Higher initial temperature ---
r2 = model.predict({
    "T0_K": 573.15,
    "duration_s": 600.0,
    "dt": 1.0,
})

# --- Scenario 3: Conversion vs temperature (steady-state) ---
T_range, X_kin, X_eq = m.conversion_vs_temperature(
    T_range=np.linspace(473.15, 773.15, 30),
)
T_C = T_range - 273.15

# --- Scenario 4: Pressure effect ---
X_at_P = []
P_range = [1.0, 5.0, 10.0, 20.0, 30.0]
for P in P_range:
    r_p = m.simulate(T0=573.15, duration_s=2000.0, dt=50.0, P=P,
                     T_in=573.15, T_cool=573.15)
    X_at_P.append(r_p["X_CO2"][-1])

fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=[
        "Reactor Temperature Transient",
        "CO2 Conversion Transient",
        "Species Concentrations (T0=250C)",
        "CH4 Dry Mole Fraction",
        "Conversion vs Temperature (kinetic vs equilibrium)",
        "Effect of Pressure on Conversion (T=300C)",
    ],
    vertical_spacing=0.10,
    horizontal_spacing=0.10,
)

# Row 1: Temperature + conversion transients
fig.add_trace(go.Scatter(x=r1["t"], y=r1["T"] - 273.15, name="T (250C start)",
              line=dict(color="#d62728")), row=1, col=1)
fig.add_trace(go.Scatter(x=r2["t"], y=r2["T"] - 273.15, name="T (300C start)",
              line=dict(color="#ff7f0e", dash="dash")), row=1, col=1)

fig.add_trace(go.Scatter(x=r1["t"], y=r1["X_CO2"], name="X_CO2 (250C)",
              line=dict(color="#1f77b4")), row=1, col=2)
fig.add_trace(go.Scatter(x=r2["t"], y=r2["X_CO2"], name="X_CO2 (300C)",
              line=dict(color="#2ca02c", dash="dash")), row=1, col=2)

# Row 2: Species concentrations
fig.add_trace(go.Scatter(x=r1["t"], y=r1["C_CO2"], name="CO2",
              line=dict(color="#1f77b4"), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=r1["t"], y=r1["C_H2"], name="H2",
              line=dict(color="#2ca02c"), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=r1["t"], y=r1["C_CH4"], name="CH4",
              line=dict(color="#ff7f0e"), showlegend=False), row=2, col=1)

fig.add_trace(go.Scatter(x=r1["t"], y=r1["y_CH4_dry"], name="y_CH4_dry",
              line=dict(color="#9467bd"), showlegend=False), row=2, col=2)

# Row 3: Steady-state analysis
fig.add_trace(go.Scatter(x=T_C, y=X_kin, name="Kinetic",
              line=dict(color="#1f77b4"), showlegend=False), row=3, col=1)
fig.add_trace(go.Scatter(x=T_C, y=X_eq, name="Equilibrium",
              line=dict(color="#d62728", dash="dash"), showlegend=False), row=3, col=1)

fig.add_trace(go.Bar(x=[str(p) for p in P_range], y=X_at_P, name="X_CO2 vs P",
              marker_color="#2ca02c", showlegend=False), row=3, col=2)

fig.update_xaxes(title_text="Time (s)", row=1, col=1)
fig.update_xaxes(title_text="Time (s)", row=1, col=2)
fig.update_xaxes(title_text="Time (s)", row=2, col=1)
fig.update_xaxes(title_text="Time (s)", row=2, col=2)
fig.update_xaxes(title_text="Temperature (C)", row=3, col=1)
fig.update_xaxes(title_text="Pressure (bar)", row=3, col=2)
fig.update_yaxes(title_text="T (C)", row=1, col=1)
fig.update_yaxes(title_text="CO2 Conversion", row=1, col=2)
fig.update_yaxes(title_text="mol/m3", row=2, col=1)
fig.update_yaxes(title_text="y_CH4 (dry)", row=2, col=2)
fig.update_yaxes(title_text="Conversion", row=3, col=1)
fig.update_yaxes(title_text="Conversion", row=3, col=2)

fig.update_layout(
    title_text="EC193 Methanation Reactor -- F2a Kinetics + Equilibrium CSTR",
    height=1000,
    showlegend=True,
)

html_path = os.path.join(OUTPUT_DIR, "simulation_report.html")
fig.write_html(html_path)
print(f"Report saved to {html_path}")
