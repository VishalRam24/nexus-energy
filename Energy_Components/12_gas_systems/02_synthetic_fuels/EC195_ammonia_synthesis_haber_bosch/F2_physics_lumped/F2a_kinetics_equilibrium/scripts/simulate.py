"""
EC195 -- Ammonia Synthesis (Haber-Bosch) -- F2a Kinetics + Equilibrium
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

# --- Scenario 1: Reactor transient ---
r1 = model.predict({
    "T0_K": 673.15,
    "duration_s": 600.0,
    "dt": 1.0,
})

# --- Scenario 2: Higher temperature start ---
r2 = model.predict({
    "T0_K": 773.15,
    "duration_s": 600.0,
    "dt": 1.0,
})

# --- Scenario 3: Equilibrium conversion vs T at various P ---
T_range, X_eq_dict = m.conversion_vs_T_P(
    T_range=np.linspace(573.15, 873.15, 40),
    P_values=[100, 150, 200, 250, 300],
)
T_C = T_range - 273.15

# --- Scenario 4: Recycle loop ---
r_recycle = m.simulate_with_recycle(n_passes=15)

fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=[
        "Reactor Temperature Transient",
        "N2 Conversion Transient",
        "NH3 Mole Fraction",
        "Species Concentrations (T0=400C)",
        "Equilibrium Conversion vs T at Various P",
        "Recycle Loop: Per-Pass Conversion",
    ],
    vertical_spacing=0.10,
    horizontal_spacing=0.10,
)

# Row 1: Transients
fig.add_trace(go.Scatter(x=r1["t"], y=r1["T"] - 273.15, name="T (400C)",
              line=dict(color="#d62728")), row=1, col=1)
fig.add_trace(go.Scatter(x=r2["t"], y=r2["T"] - 273.15, name="T (500C)",
              line=dict(color="#ff7f0e", dash="dash")), row=1, col=1)

fig.add_trace(go.Scatter(x=r1["t"], y=r1["X_N2"], name="X_N2 (400C)",
              line=dict(color="#1f77b4")), row=1, col=2)
fig.add_trace(go.Scatter(x=r2["t"], y=r2["X_N2"], name="X_N2 (500C)",
              line=dict(color="#2ca02c", dash="dash")), row=1, col=2)

# Row 2: NH3 fraction + species
fig.add_trace(go.Scatter(x=r1["t"], y=r1["y_NH3"], name="y_NH3 (400C)",
              line=dict(color="#9467bd"), showlegend=False), row=2, col=1)

fig.add_trace(go.Scatter(x=r1["t"], y=r1["C_N2"], name="N2",
              line=dict(color="#1f77b4"), showlegend=False), row=2, col=2)
fig.add_trace(go.Scatter(x=r1["t"], y=r1["C_H2"], name="H2",
              line=dict(color="#2ca02c"), showlegend=False), row=2, col=2)
fig.add_trace(go.Scatter(x=r1["t"], y=r1["C_NH3"], name="NH3",
              line=dict(color="#ff7f0e"), showlegend=False), row=2, col=2)

# Row 3: Equilibrium + recycle
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
for i, (P, X_arr) in enumerate(X_eq_dict.items()):
    fig.add_trace(go.Scatter(x=T_C, y=X_arr, name=f"{P} atm",
                  line=dict(color=colors[i])), row=3, col=1)

passes = list(range(1, len(r_recycle["single_pass_conversions"]) + 1))
fig.add_trace(go.Bar(x=passes, y=r_recycle["single_pass_conversions"],
              name="Per-pass X", marker_color="#2ca02c", showlegend=False), row=3, col=2)

fig.update_xaxes(title_text="Time (s)", row=1, col=1)
fig.update_xaxes(title_text="Time (s)", row=1, col=2)
fig.update_xaxes(title_text="Time (s)", row=2, col=1)
fig.update_xaxes(title_text="Time (s)", row=2, col=2)
fig.update_xaxes(title_text="Temperature (C)", row=3, col=1)
fig.update_xaxes(title_text="Pass Number", row=3, col=2)
fig.update_yaxes(title_text="T (C)", row=1, col=1)
fig.update_yaxes(title_text="N2 Conversion", row=1, col=2)
fig.update_yaxes(title_text="y_NH3", row=2, col=1)
fig.update_yaxes(title_text="mol/m3", row=2, col=2)
fig.update_yaxes(title_text="Eq. Conversion", row=3, col=1)
fig.update_yaxes(title_text="Single-Pass X", row=3, col=2)

fig.update_layout(
    title_text="EC195 Ammonia Synthesis (Haber-Bosch) -- F2a Temkin-Pyzhev + CSTR",
    height=1000,
    showlegend=True,
)

html_path = os.path.join(OUTPUT_DIR, "simulation_report.html")
fig.write_html(html_path)
print(f"Report saved to {html_path}")
