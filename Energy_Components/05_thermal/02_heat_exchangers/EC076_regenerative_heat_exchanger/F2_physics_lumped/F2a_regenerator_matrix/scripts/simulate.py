"""
EC076 -- Regenerative Heat Exchanger -- F2a Physics-Lumped Regenerator Matrix
Plotly HTML simulation report (optional; plotly import guarded).
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
    print("plotly not installed -- skipping HTML report. (pip install plotly)")
    sys.exit(0)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "model_files")
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = ComponentModel()
m = model._model

# Scenario 1: convergence history to periodic steady state
r1 = model.predict({"T_h_in_K": 573.15, "T_c_in_K": 293.15, "n_cycles": 60})

# Scenario 2: effectiveness vs matrix capacity ratio Cr* (sweep rpm)
rpms = np.linspace(1.0, 50.0, 40)
crs = [m.matrix_capacity_ratio(rpm=rp) for rp in rpms]
eps_corr = [m.effectiveness_correlation(rpm=rp) for rp in rpms]

# Scenario 3: effectiveness vs NTU_o (sweep hA)
hA_vals = np.linspace(1000.0, 30000.0, 30)
ntus, eps_ntu = [], []
for hA in hA_vals:
    m.hA_h = m.hA_c = hA
    ntus.append(m.ntu_overall())
    eps_ntu.append(m.effectiveness_correlation())
m.hA_h = m.hA_c = 6000.0  # restore

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        "Effectiveness convergence to periodic SS",
        "Final matrix axial temperature profile",
        "Effectiveness vs matrix capacity ratio Cr*",
        "Effectiveness vs overall NTU_o",
    ],
    vertical_spacing=0.14, horizontal_spacing=0.10,
)

fig.add_trace(go.Scatter(y=r1["eps_history"], mode="lines+markers",
              name="eps(cycle)", line=dict(color="#1f77b4")), row=1, col=1)
fig.add_trace(go.Scatter(y=r1["matrix_profile_final"], mode="lines+markers",
              name="T_w node", line=dict(color="#d62728")), row=1, col=2)
fig.add_trace(go.Scatter(x=crs, y=eps_corr, mode="lines",
              name="eps vs Cr*", line=dict(color="#2ca02c")), row=2, col=1)
fig.add_trace(go.Scatter(x=ntus, y=eps_ntu, mode="lines",
              name="eps vs NTU_o", line=dict(color="#ff7f0e")), row=2, col=2)

for r, c, xl, yl in [
    (1, 1, "Cycle #", "Effectiveness"),
    (1, 2, "Node index (hot->cold face)", "Matrix T (K)"),
    (2, 1, "Cr* (matrix capacity ratio)", "Effectiveness"),
    (2, 2, "NTU_o", "Effectiveness"),
]:
    fig.update_xaxes(title_text=xl, row=r, col=c)
    fig.update_yaxes(title_text=yl, row=r, col=c)

fig.update_layout(
    title="<b>EC076 Regenerative HX -- F2a Periodic-Flow Matrix (Coppage-London)</b>",
    height=850, template="plotly_white", showlegend=False,
)

out_path = os.path.join(OUTPUT_DIR, "EC076_F2a_regenerator_report.html")
fig.write_html(out_path, include_plotlyjs="cdn")
print(f"Report saved: {out_path}")
