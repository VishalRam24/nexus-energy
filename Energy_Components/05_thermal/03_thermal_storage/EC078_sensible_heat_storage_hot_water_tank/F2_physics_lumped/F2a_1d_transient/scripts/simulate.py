"""
EC078 -- Hot Water Tank TES -- F2a 1D Transient
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
m = model._model

# --- Scenario 1: Charge from uniform cold ---
T_uniform_cold = np.full(20, 293.15)
r1 = m.simulate(0.5, 353.15, 0.0, 288.15, T_uniform_cold, 30.0, 3600.0)

# --- Scenario 2: Discharge from fully charged ---
T_uniform_hot = np.full(20, 353.15)
r2 = m.simulate(0.0, 353.15, 0.5, 288.15, T_uniform_hot, 30.0, 3600.0)

# --- Scenario 3: Charge then discharge cycle ---
def cycle_charge(t):
    return 0.5 if t < 1800 else 0.0

def cycle_discharge(t):
    return 0.0 if t < 1800 else 0.5

r3 = m.simulate(cycle_charge, 353.15, cycle_discharge, 288.15, T_uniform_cold, 30.0, 3600.0)

fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=[
        "Charge: Temperature Profiles Over Time",
        "Charge: Energy Stored & Stratification",
        "Discharge: Temperature Profiles Over Time",
        "Discharge: Energy Stored & Stratification",
        "Charge/Discharge Cycle: Temperature Profiles",
        "Cycle: Energy Stored & Stratification",
    ],
    vertical_spacing=0.10,
    horizontal_spacing=0.10,
)

# Helper: plot temperature profiles at selected times
colors_profile = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

for scenario_idx, (r, row) in enumerate([(r1, 1), (r2, 2), (r3, 3)]):
    # Select ~6 time snapshots
    n_times = len(r["t"])
    indices = np.linspace(0, n_times - 1, 6, dtype=int)
    node_positions = np.linspace(0, 2.0, 20)  # height in meters

    for j, idx in enumerate(indices):
        t_val = r["t"][idx]
        T_profile = r["T_profiles"][:, idx] - 273.15
        fig.add_trace(go.Scatter(
            x=T_profile, y=node_positions,
            name=f"t={t_val:.0f}s" if row == 1 else None,
            showlegend=(row == 1),
            line=dict(color=colors_profile[j % len(colors_profile)]),
        ), row=row, col=1)

    # Energy and stratification
    fig.add_trace(go.Scatter(
        x=r["t"], y=r["E_stored_kWh"],
        name="E_stored" if row == 1 else None,
        showlegend=(row == 1),
        line=dict(color="#ff7f0e"),
    ), row=row, col=2)
    fig.add_trace(go.Scatter(
        x=r["t"], y=r["stratification_K"],
        name="Stratification" if row == 1 else None,
        showlegend=(row == 1),
        line=dict(color="#2ca02c", dash="dash"),
    ), row=row, col=2)

for r in range(1, 4):
    fig.update_xaxes(title_text="Temperature (C)", row=r, col=1)
    fig.update_yaxes(title_text="Height (m)", row=r, col=1)
    fig.update_xaxes(title_text="Time (s)", row=r, col=2)
    fig.update_yaxes(title_text="Energy (kWh) / Stratification (K)", row=r, col=2)

fig.update_layout(
    title="<b>EC078 Hot Water Tank TES -- F2a 1D Transient (20-node)</b>",
    height=1100, template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

out_path = os.path.join(OUTPUT_DIR, "EC078_F2a_1d_transient_report.html")
fig.write_html(out_path, include_plotlyjs="cdn")
print(f"Report saved: {out_path}")
