"""
EC184 -- Power Factor Correction Unit -- F2a Physics-Lumped
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

# --- Scenario 1: PF correction sweep vs initial PF ---
pf1_sweep = np.linspace(0.60, 0.95, 60)
Qc_arr, pf_after, released = [], [], []
for pf1 in pf1_sweep:
    r = m.compensate(800.0, pf1, 0.95)
    Qc_arr.append(r["Q_compensated_kVAR"])
    pf_after.append(r["pf_achieved"])
    released.append(r["released_capacity_kVA"])

# --- Scenario 2: resonance order vs Qc ---
Qc_sweep = np.linspace(100.0, 1000.0, 60)
h_par = [m.resonance(q)["h_parallel"] for q in Qc_sweep]

# --- Scenario 3: energization inrush transient ---
tr = m.energize(Qc_kVAR=1000.0, switch_angle_deg=90.0, duration_s=0.06)
tr_det = m.energize(Qc_kVAR=1000.0, detuning_pct=7.0,
                    switch_angle_deg=90.0, duration_s=0.06)

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        "PF after correction vs initial PF (target 0.95)",
        "Released upstream capacity (kVA)",
        "Parallel resonance order vs bank size",
        "Energization inrush current (90deg close)",
    ],
)

fig.add_trace(go.Scatter(x=pf1_sweep, y=pf_after, name="pf_after",
              line=dict(color="#1f77b4")), row=1, col=1)
fig.add_trace(go.Scatter(x=pf1_sweep, y=released, name="released kVA",
              line=dict(color="#2ca02c"), showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=Qc_sweep, y=h_par, name="h_parallel",
              line=dict(color="#d62728"), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=tr["t"] * 1000, y=tr["i"] / 1000, name="undamped/no reactor",
              line=dict(color="#ff7f0e")), row=2, col=2)
fig.add_trace(go.Scatter(x=tr_det["t"] * 1000, y=tr_det["i"] / 1000,
              name="with 7% detuning reactor",
              line=dict(color="#9467bd")), row=2, col=2)

for r, c, xl, yl in [
    (1, 1, "Initial PF", "Achieved PF"),
    (1, 2, "Initial PF", "Released kVA"),
    (2, 1, "Qc (kVAR)", "Resonance order h"),
    (2, 2, "Time (ms)", "Inrush current (kA)"),
]:
    fig.update_xaxes(title_text=xl, row=r, col=c)
    fig.update_yaxes(title_text=yl, row=r, col=c)

fig.update_layout(
    title="<b>EC184 PFC Unit -- F2a Physics-Lumped (compensation + resonance + RLC inrush)</b>",
    height=820, template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1),
)

out_path = os.path.join(OUTPUT_DIR, "EC184_F2a_switching_transient_report.html")
fig.write_html(out_path, include_plotlyjs="cdn")
print(f"Report saved: {out_path}")
