"""
EC079 -- Molten Salt TES -- F2a 1D Transient
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

# --- Scenario 1: Charge from cold tank ---
T_cold_init = m.initial_temperature_profile("cold")
r1 = m.simulate(100.0, 838.15, 0.0, 563.15, T_cold_init, 60.0, 14400.0)

# --- Scenario 2: Discharge from hot tank ---
T_hot_init = m.initial_temperature_profile("hot")
r2 = m.simulate(0.0, 838.15, 100.0, 563.15, T_hot_init, 60.0, 14400.0)

# --- Scenario 3: Daily charge/discharge cycle ---
def daily_charge(t):
    return 150.0 if t < 7200 else 0.0

def daily_discharge(t):
    return 0.0 if t < 7200 else 120.0

r3 = m.simulate(daily_charge, 838.15, daily_discharge, 563.15, T_cold_init, 60.0, 14400.0)

# --- Salt properties vs temperature ---
T_range = np.linspace(563.15, 838.15, 100)
rho_arr = [m.salt_density(T) for T in T_range]
cp_arr = [m.salt_cp(T) for T in T_range]
k_arr = [m.salt_conductivity(T) for T in T_range]

fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=[
        "Charge: Temperature Profiles",
        "Charge: Energy Stored",
        "Discharge: Temperature Profiles",
        "Discharge: Energy Stored",
        "Daily Cycle: Energy & Stratification",
        "Solar Salt Properties vs Temperature",
    ],
    vertical_spacing=0.10,
    horizontal_spacing=0.12,
)

colors_profile = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
node_positions = np.linspace(0, 12.0, 20)

# Rows 1-2: Charge and discharge profiles
for scenario_idx, (r, row) in enumerate([(r1, 1), (r2, 2)]):
    n_times = len(r["t"])
    indices = np.linspace(0, n_times - 1, 6, dtype=int)
    for j, idx in enumerate(indices):
        t_val = r["t"][idx]
        T_profile = r["T_profiles"][:, idx] - 273.15
        fig.add_trace(go.Scatter(
            x=T_profile, y=node_positions,
            name=f"t={t_val/3600:.1f}h" if row == 1 else None,
            showlegend=(row == 1),
            line=dict(color=colors_profile[j % len(colors_profile)]),
        ), row=row, col=1)

    fig.add_trace(go.Scatter(
        x=r["t"] / 3600, y=r["E_stored_MWh"],
        name="E_stored" if row == 1 else None,
        showlegend=(row == 1),
        line=dict(color="#ff7f0e"),
    ), row=row, col=2)

# Row 1-2 axes
for row in [1, 2]:
    fig.update_xaxes(title_text="Temperature (C)", row=row, col=1)
    fig.update_yaxes(title_text="Height (m)", row=row, col=1)
    fig.update_xaxes(title_text="Time (hours)", row=row, col=2)
    fig.update_yaxes(title_text="Energy (MWh)", row=row, col=2)

# Row 3 col 1: Daily cycle
fig.add_trace(go.Scatter(x=r3["t"]/3600, y=r3["E_stored_MWh"], name="E cycle",
              line=dict(color="#ff7f0e"), showlegend=False), row=3, col=1)
fig.add_trace(go.Scatter(x=r3["t"]/3600, y=r3["stratification_K"], name="Strat",
              line=dict(color="#2ca02c", dash="dash"), showlegend=False), row=3, col=1)
fig.update_xaxes(title_text="Time (hours)", row=3, col=1)
fig.update_yaxes(title_text="Energy (MWh) / Stratification (K)", row=3, col=1)

# Row 3 col 2: Salt properties
fig.add_trace(go.Scatter(x=T_range-273.15, y=rho_arr, name="rho",
              line=dict(color="#1f77b4"), showlegend=False), row=3, col=2)
fig.add_trace(go.Scatter(x=T_range-273.15, y=cp_arr, name="cp",
              line=dict(color="#d62728", dash="dash"), showlegend=False), row=3, col=2)
fig.update_xaxes(title_text="Temperature (C)", row=3, col=2)
fig.update_yaxes(title_text="rho (kg/m3) / cp (J/(kg.K))", row=3, col=2)

fig.update_layout(
    title="<b>EC079 Molten Salt TES -- F2a 1D Transient (20-node, T-dependent)</b>",
    height=1100, template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

out_path = os.path.join(OUTPUT_DIR, "EC079_F2a_1d_transient_report.html")
fig.write_html(out_path, include_plotlyjs="cdn")
print(f"Report saved: {out_path}")
