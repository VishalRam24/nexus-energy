"""
EC101 -- Combined Cycle Gas Turbine (CCGT) -- F2a Thermo Cycle SS
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

# --- Scenario 1: Design point ---
r_design = m.combined_cycle()

# --- Scenario 2: PR sweep ---
PR_arr = np.arange(10, 32, 1)
PR_results = [m.combined_cycle(PR=pr) for pr in PR_arr]
eta_PR = [r["eta_combined"] for r in PR_results]
W_PR = [r["W_total_MW"] for r in PR_results]

# --- Scenario 3: TIT sweep ---
TIT_arr = np.linspace(1273.15, 1873.15, 25)
TIT_results = [m.combined_cycle(TIT=tit) for tit in TIT_arr]
eta_TIT = [r["eta_combined"] for r in TIT_results]
W_TIT = [r["W_total_MW"] for r in TIT_results]

# --- Scenario 4: Part-load sweep ---
load_arr = np.linspace(0.3, 1.0, 20)
load_results = [m.combined_cycle(load_fraction=lf) for lf in load_arr]
eta_load = [r["eta_combined"] for r in load_results]
W_load = [r["W_total_MW"] for r in load_results]

fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=[
        f"Design Point: eta={r_design['eta_combined']:.1%}, W={r_design['W_total_MW']:.0f} MW",
        "PR Sweep: Combined Efficiency",
        "TIT Sweep: Combined Efficiency",
        "TIT Sweep: Power Output",
        "Part-Load: Efficiency",
        "Part-Load: Power Output",
    ],
    vertical_spacing=0.10,
    horizontal_spacing=0.10,
)

# Row 1: Design point T-s diagram (approximate) + PR sweep
b = r_design["brayton"]
T_brayton = [b["T1"], b["T2"], b["T3"], b["T4"], b["T1"]]
stage_labels = ["1 (inlet)", "2 (comp out)", "3 (comb out)", "4 (turb out)", "1"]
fig.add_trace(go.Scatter(x=list(range(5)), y=[t-273.15 for t in T_brayton],
              mode="lines+markers", name="Brayton T",
              text=stage_labels, line=dict(color="#d62728")), row=1, col=1)
fig.add_trace(go.Scatter(x=PR_arr, y=[e*100 for e in eta_PR], name="eta_cc vs PR",
              line=dict(color="#1f77b4")), row=1, col=2)

# Row 2: TIT sweep
fig.add_trace(go.Scatter(x=TIT_arr-273.15, y=[e*100 for e in eta_TIT], name="eta_cc vs TIT",
              line=dict(color="#2ca02c")), row=2, col=1)
fig.add_trace(go.Scatter(x=TIT_arr-273.15, y=W_TIT, name="W_total vs TIT",
              line=dict(color="#ff7f0e")), row=2, col=2)

# Row 3: Part-load
fig.add_trace(go.Scatter(x=load_arr*100, y=[e*100 for e in eta_load], name="eta_cc vs load",
              line=dict(color="#9467bd")), row=3, col=1)
fig.add_trace(go.Scatter(x=load_arr*100, y=W_load, name="W_total vs load",
              line=dict(color="#8c564b")), row=3, col=2)

for r, c, xl, yl in [
    (1, 1, "Stage", "Temperature (C)"),
    (1, 2, "Pressure Ratio", "Combined Efficiency (%)"),
    (2, 1, "TIT (C)", "Combined Efficiency (%)"),
    (2, 2, "TIT (C)", "Total Power (MW)"),
    (3, 1, "Load Fraction (%)", "Combined Efficiency (%)"),
    (3, 2, "Load Fraction (%)", "Total Power (MW)"),
]:
    fig.update_xaxes(title_text=xl, row=r, col=c)
    fig.update_yaxes(title_text=yl, row=r, col=c)

fig.update_layout(
    title="<b>EC101 CCGT -- F2a Steady-State Thermodynamic Cycle (Brayton + Rankine)</b>",
    height=1000, template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

out_path = os.path.join(OUTPUT_DIR, "EC101_F2a_thermo_cycle_ss_report.html")
fig.write_html(out_path, include_plotlyjs="cdn")
print(f"Report saved: {out_path}")
