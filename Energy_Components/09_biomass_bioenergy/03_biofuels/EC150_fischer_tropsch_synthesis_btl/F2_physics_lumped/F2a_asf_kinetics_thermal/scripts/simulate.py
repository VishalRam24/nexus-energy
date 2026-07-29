"""
EC150 -- Fischer-Tropsch Synthesis (BTL) -- F2a ASF Kinetics + Thermal ODE
Plotly HTML simulation report. Plotly import is guarded so absence won't crash.
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

# --- Scenario 1: cold start thermal + conversion transient ---
r1 = model.predict({"syngas_flow_mol_s": 120.0, "CO_fraction": 0.40,
                    "T0_K": 470.0, "duration_s": 4000.0, "dt": 40.0})

# --- Scenario 2: ASF product slate vs alpha ---
alphas = np.linspace(0.55, 0.93, 60)
wax = [m.product_cuts(a)["wax_C21plus"] for a in alphas]
diesel = [m.product_cuts(a)["diesel_C10_C20"] for a in alphas]
naphtha = [m.product_cuts(a)["naphtha_C5_C9"] for a in alphas]
light = [m.product_cuts(a)["light_gas_C1_C4"] for a in alphas]

# --- Scenario 3: conversion vs temperature ---
Tspan = np.linspace(465.0, 535.0, 40)
Xt = [m.co_conversion(T, 48.0, 96.0, m.P_nom) for T in Tspan]

# --- Scenario 4: heat generated vs removed during transient ---
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        "Cold-start Thermal Transient + CO Conversion",
        "ASF Product Slate vs Chain-growth alpha",
        "CO Conversion vs Reactor Temperature (Arrhenius/LHHW)",
        "Exothermic Heat: Generated vs Removed",
    ],
    specs=[[{"secondary_y": True}, {}], [{}, {}]],
    vertical_spacing=0.13, horizontal_spacing=0.10,
)

fig.add_trace(go.Scatter(x=r1["t"], y=r1["temperature"], name="T (K)",
              line=dict(color="#d62728")), row=1, col=1, secondary_y=False)
fig.add_trace(go.Scatter(x=r1["t"], y=r1["CO_conversion"], name="X_CO",
              line=dict(color="#1f77b4")), row=1, col=1, secondary_y=True)

for arr, nm, c in [(wax, "wax C21+", "#8c564b"), (diesel, "diesel C10-C20", "#2ca02c"),
                   (naphtha, "naphtha C5-C9", "#ff7f0e"), (light, "light gas C1-C4", "#9467bd")]:
    fig.add_trace(go.Scatter(x=alphas, y=arr, name=nm, line=dict(color=c)), row=1, col=2)

fig.add_trace(go.Scatter(x=Tspan, y=Xt, name="X(T)", line=dict(color="#17becf"),
              showlegend=False), row=2, col=1)

fig.add_trace(go.Scatter(x=r1["t"], y=r1["heat_generated_W"], name="Q_gen",
              line=dict(color="#e377c2")), row=2, col=2)
fig.add_trace(go.Scatter(x=r1["t"], y=r1["heat_removed_W"], name="Q_cool",
              line=dict(color="#1f77b4", dash="dash")), row=2, col=2)

fig.update_yaxes(title_text="Temperature (K)", row=1, col=1, secondary_y=False)
fig.update_yaxes(title_text="CO conversion", row=1, col=1, secondary_y=True)
fig.update_xaxes(title_text="Time (s)", row=1, col=1)
fig.update_xaxes(title_text="alpha", row=1, col=2)
fig.update_yaxes(title_text="weight fraction", row=1, col=2)
fig.update_xaxes(title_text="Temperature (K)", row=2, col=1)
fig.update_yaxes(title_text="CO conversion", row=2, col=1)
fig.update_xaxes(title_text="Time (s)", row=2, col=2)
fig.update_yaxes(title_text="Heat (W)", row=2, col=2)

fig.update_layout(
    title="<b>EC150 Fischer-Tropsch (BTL) -- F2a ASF Kinetics + Exothermic Thermal ODE</b>",
    height=900, template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

out_path = os.path.join(OUTPUT_DIR, "EC150_F2a_asf_kinetics_thermal_report.html")
fig.write_html(out_path, include_plotlyjs="cdn")
print(f"Report saved: {out_path}")
