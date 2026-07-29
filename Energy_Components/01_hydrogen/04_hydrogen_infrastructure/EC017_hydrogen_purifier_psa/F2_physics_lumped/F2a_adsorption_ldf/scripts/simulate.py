"""
EC017 -- Hydrogen Purifier (PSA) -- F2a Adsorption + LDF
Plotly HTML simulation report generator (optional; plotly import guarded).
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

# --- Scenario 1: full PSA cycle loading profile (adsorption->blowdown->purge) ---
r1 = m.simulate_cycle(q0=0.0, dt=1.0)

# --- Scenario 2: breakthrough -- purity vs adsorption time ---
t_ads_sweep = np.linspace(60, 800, 30)
purity_bt = [m.simulate_cycle(t_ads=ta, q0=0.0, dt=4.0)["purity"] for ta in t_ads_sweep]

# --- Scenario 3: recovery vs purge ratio ---
pr_sweep = np.linspace(0.05, 0.40, 25)
recovery_pr = [m.simulate_cycle(purge_ratio=pr, dt=4.0)["recovery"] for pr in pr_sweep]

# --- Scenario 4: Langmuir isotherm at several temperatures ---
p_sweep = np.linspace(0.0, 30.0, 100)
temps = [273.15, 298.15, 323.15, 373.15]
colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        "PSA Cycle: Bed Loading q(t) (ads -> blowdown -> purge)",
        "Breakthrough: Product Purity vs Adsorption Time",
        "Recovery vs Purge-to-Feed Ratio",
        "Langmuir Isotherm q*(p) at Various T",
    ],
    vertical_spacing=0.13, horizontal_spacing=0.10,
)

fig.add_trace(go.Scatter(x=r1["t"], y=r1["loading"], name="q(t)",
              line=dict(color="#1f77b4")), row=1, col=1)
fig.add_trace(go.Scatter(x=r1["t"], y=r1["loading_equilibrium"], name="q* (target)",
              line=dict(color="#888", dash="dash")), row=1, col=1)

fig.add_trace(go.Scatter(x=t_ads_sweep, y=purity_bt, name="purity",
              line=dict(color="#d62728"), showlegend=False), row=1, col=2)

fig.add_trace(go.Scatter(x=pr_sweep, y=recovery_pr, name="recovery",
              line=dict(color="#2ca02c"), showlegend=False), row=2, col=1)

for i, T in enumerate(temps):
    q = m.q_equilibrium(p_sweep, T)
    fig.add_trace(go.Scatter(x=p_sweep, y=q, name=f"T={T-273.15:.0f}C",
                  line=dict(color=colors[i])), row=2, col=2)

for rr, cc, xl, yl in [
    (1, 1, "Time (s)", "Loading q (mol/kg)"),
    (1, 2, "Adsorption time (s)", "Product purity (-)"),
    (2, 1, "Purge / feed ratio (-)", "H2 recovery (-)"),
    (2, 2, "Impurity partial pressure (bar)", "q* (mol/kg)"),
]:
    fig.update_xaxes(title_text=xl, row=rr, col=cc)
    fig.update_yaxes(title_text=yl, row=rr, col=cc)

fig.update_layout(
    title="<b>EC017 Hydrogen PSA -- F2a Lumped Adsorption (Langmuir + LDF)</b>",
    height=850, template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1),
)

out_path = os.path.join(OUTPUT_DIR, "EC017_F2a_adsorption_ldf_report.html")
fig.write_html(out_path, include_plotlyjs="cdn")
print(f"Report saved: {out_path}")
