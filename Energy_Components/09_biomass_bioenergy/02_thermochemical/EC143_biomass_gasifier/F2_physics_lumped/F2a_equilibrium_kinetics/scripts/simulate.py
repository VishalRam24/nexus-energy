"""
EC143 -- Biomass Gasifier -- F2a Chemical Equilibrium
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

# --- Scenario 1: Temperature sweep at ER=0.30 ---
T_range = np.linspace(973.15, 1373.15, 50)
T_arr, results_T = m.temperature_sweep(T_range=T_range, ER=0.30)
T_C = T_range - 273.15

CO_T = [r["composition_dry_mol_pct"]["CO"] for r in results_T]
CO2_T = [r["composition_dry_mol_pct"]["CO2"] for r in results_T]
H2_T = [r["composition_dry_mol_pct"]["H2"] for r in results_T]
CH4_T = [r["composition_dry_mol_pct"]["CH4"] for r in results_T]
N2_T = [r["composition_dry_mol_pct"]["N2"] for r in results_T]
LHV_T = [r["LHV_syngas_MJ_Nm3"] for r in results_T]
CGE_T = [r["cold_gas_efficiency"] for r in results_T]

# --- Scenario 2: ER sweep at T=1073K ---
ER_range = np.linspace(0.15, 0.50, 50)
ER_arr, results_ER = m.er_sweep(ER_range=ER_range, T=1073.15)

CO_ER = [r["composition_dry_mol_pct"]["CO"] for r in results_ER]
CO2_ER = [r["composition_dry_mol_pct"]["CO2"] for r in results_ER]
H2_ER = [r["composition_dry_mol_pct"]["H2"] for r in results_ER]
CH4_ER = [r["composition_dry_mol_pct"]["CH4"] for r in results_ER]
LHV_ER = [r["LHV_syngas_MJ_Nm3"] for r in results_ER]
H2CO_ER = [r["H2_CO_ratio"] for r in results_ER]

# --- Scenario 3: Moisture sweep at T=1073K, ER=0.30 ---
moist_range = np.linspace(0.0, 0.40, 30)
H2_M = []
CO_M = []
LHV_M = []
for moist in moist_range:
    r = m.solve_equilibrium(T=1073.15, ER=0.30, moisture=moist)
    H2_M.append(r["composition_dry_mol_pct"]["H2"])
    CO_M.append(r["composition_dry_mol_pct"]["CO"])
    LHV_M.append(r["LHV_syngas_MJ_Nm3"])

fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=[
        "Syngas Composition vs Temperature (ER=0.30)",
        "LHV & Cold Gas Efficiency vs Temperature",
        "Syngas Composition vs ER (T=800C)",
        "LHV & H2/CO Ratio vs ER",
        "Effect of Moisture on H2 and CO",
        "LHV vs Moisture Content",
    ],
    vertical_spacing=0.10,
    horizontal_spacing=0.10,
)

# Row 1: Temperature sweep
for name, data, color in [("CO", CO_T, "#1f77b4"), ("CO2", CO2_T, "#ff7f0e"),
                            ("H2", H2_T, "#2ca02c"), ("CH4", CH4_T, "#d62728"),
                            ("N2", N2_T, "#9467bd")]:
    fig.add_trace(go.Scatter(x=T_C, y=data, name=name, line=dict(color=color)),
                  row=1, col=1)

fig.add_trace(go.Scatter(x=T_C, y=LHV_T, name="LHV", line=dict(color="#1f77b4"),
              showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=T_C, y=CGE_T, name="CGE", line=dict(color="#ff7f0e", dash="dash"),
              showlegend=False, yaxis="y2"), row=1, col=2)

# Row 2: ER sweep
for name, data, color in [("CO", CO_ER, "#1f77b4"), ("CO2", CO2_ER, "#ff7f0e"),
                            ("H2", H2_ER, "#2ca02c"), ("CH4", CH4_ER, "#d62728")]:
    fig.add_trace(go.Scatter(x=ER_range, y=data, name=f"{name} (ER)", line=dict(color=color),
                  showlegend=False), row=2, col=1)

fig.add_trace(go.Scatter(x=ER_range, y=LHV_ER, name="LHV (ER)", line=dict(color="#1f77b4"),
              showlegend=False), row=2, col=2)
fig.add_trace(go.Scatter(x=ER_range, y=H2CO_ER, name="H2/CO", line=dict(color="#ff7f0e", dash="dash"),
              showlegend=False), row=2, col=2)

# Row 3: Moisture sweep
fig.add_trace(go.Scatter(x=moist_range * 100, y=H2_M, name="H2 (moist)", line=dict(color="#2ca02c"),
              showlegend=False), row=3, col=1)
fig.add_trace(go.Scatter(x=moist_range * 100, y=CO_M, name="CO (moist)", line=dict(color="#1f77b4"),
              showlegend=False), row=3, col=1)
fig.add_trace(go.Scatter(x=moist_range * 100, y=LHV_M, name="LHV (moist)", line=dict(color="#d62728"),
              showlegend=False), row=3, col=2)

fig.update_xaxes(title_text="Temperature (C)", row=1, col=1)
fig.update_xaxes(title_text="Temperature (C)", row=1, col=2)
fig.update_xaxes(title_text="Equivalence Ratio", row=2, col=1)
fig.update_xaxes(title_text="Equivalence Ratio", row=2, col=2)
fig.update_xaxes(title_text="Moisture Content (%)", row=3, col=1)
fig.update_xaxes(title_text="Moisture Content (%)", row=3, col=2)
fig.update_yaxes(title_text="mol%", row=1, col=1)
fig.update_yaxes(title_text="MJ/Nm3 / CGE", row=1, col=2)
fig.update_yaxes(title_text="mol%", row=2, col=1)
fig.update_yaxes(title_text="MJ/Nm3 / H2:CO", row=2, col=2)
fig.update_yaxes(title_text="mol%", row=3, col=1)
fig.update_yaxes(title_text="MJ/Nm3", row=3, col=2)

fig.update_layout(
    title_text="EC143 Biomass Gasifier -- F2a Chemical Equilibrium Model",
    height=1000,
    showlegend=True,
)

html_path = os.path.join(OUTPUT_DIR, "simulation_report.html")
fig.write_html(html_path)
print(f"Report saved to {html_path}")
