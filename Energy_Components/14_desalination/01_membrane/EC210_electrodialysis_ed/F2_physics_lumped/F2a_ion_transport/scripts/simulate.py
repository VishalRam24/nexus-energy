"""
EC210 -- Electrodialysis (ED) -- F2a Ion-Transport Stack Model
Plotly HTML simulation report generator. Plotly import is guarded so its
absence does not crash the build.
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

# --- Scenario 1: concentration profiles along the stack ---
r1 = model.predict({
    "current_density_A_m2": 200.0,
    "feed_conc_mol_m3": 100.0,
    "flow_velocity_cm_s": 5.0,
    "stack_length_cm": 100.0,
})

# --- Scenario 2: SEC & removal vs current density ---
i_sweep = np.linspace(20.0, 450.0, 40)
sec_i = [model.predict({"current_density_A_m2": float(i)})["SEC_kWh_m3"] for i in i_sweep]
rem_i = [model.predict({"current_density_A_m2": float(i)})["salt_removed_fraction"] * 100
         for i in i_sweep]

# --- Scenario 3: removal vs flow velocity ---
v_sweep = np.linspace(2.0, 20.0, 30)
rem_v = [model.predict({"flow_velocity_cm_s": float(v)})["salt_removed_fraction"] * 100
         for v in v_sweep]

# --- Scenario 4: limiting current density vs concentration ---
c_sweep = np.linspace(10.0, 600.0, 60)
ilim = [m.limiting_current_density(c) * 1e4 for c in c_sweep]   # A/m2

fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=[
        "Diluate / Concentrate profiles along stack",
        "Cell-pair voltage along stack",
        "SEC vs applied current density",
        "Salt removal vs applied current density",
        "Salt removal vs flow velocity",
        "Limiting current density vs concentration",
    ],
    vertical_spacing=0.10, horizontal_spacing=0.10,
)

fig.add_trace(go.Scatter(x=r1["x"], y=r1["c_diluate"], name="diluate",
              line=dict(color="#1f77b4")), row=1, col=1)
fig.add_trace(go.Scatter(x=r1["x"], y=r1["c_concentrate"], name="concentrate",
              line=dict(color="#d62728")), row=1, col=1)
fig.add_trace(go.Scatter(x=r1["x"], y=r1["cell_pair_voltage"], name="U_pair",
              line=dict(color="#2ca02c"), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=i_sweep, y=sec_i, name="SEC",
              line=dict(color="#ff7f0e"), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=i_sweep, y=rem_i, name="removal",
              line=dict(color="#9467bd"), showlegend=False), row=2, col=2)
fig.add_trace(go.Scatter(x=v_sweep, y=rem_v, name="removal vs v",
              line=dict(color="#17becf"), showlegend=False), row=3, col=1)
fig.add_trace(go.Scatter(x=c_sweep, y=ilim, name="i_lim",
              line=dict(color="#8c564b"), showlegend=False), row=3, col=2)

for r, c, xl, yl in [
    (1, 1, "Position (cm)", "Concentration (mol/m3)"),
    (1, 2, "Position (cm)", "Cell-pair voltage (V)"),
    (2, 1, "Current density (A/m2)", "SEC (kWh/m3)"),
    (2, 2, "Current density (A/m2)", "Salt removed (%)"),
    (3, 1, "Flow velocity (cm/s)", "Salt removed (%)"),
    (3, 2, "Concentration (mol/m3)", "i_lim (A/m2)"),
]:
    fig.update_xaxes(title_text=xl, row=r, col=c)
    fig.update_yaxes(title_text=yl, row=r, col=c)

fig.update_layout(
    title="<b>EC210 Electrodialysis -- F2a Ion-Transport Stack Model</b>",
    height=1000, template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

out_path = os.path.join(OUTPUT_DIR, "EC210_F2a_ion_transport_report.html")
fig.write_html(out_path, include_plotlyjs="cdn")
print(f"Report saved: {out_path}")
