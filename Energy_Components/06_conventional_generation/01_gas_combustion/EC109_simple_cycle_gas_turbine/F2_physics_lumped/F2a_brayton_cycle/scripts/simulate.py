"""
EC109 -- Simple Cycle Gas Turbine -- F2a Brayton Cycle
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

# --- Design point ---
r_design = m.brayton_cycle()

# --- PR sweep ---
PR_arr = np.arange(6, 38, 1)
PR_results = [m.brayton_cycle(PR=pr) for pr in PR_arr]
eta_PR = [r["eta_electrical"] for r in PR_results]
W_PR = [r["W_elec_MW"] for r in PR_results]
T4_PR = [r["T4"] - 273.15 for r in PR_results]

# --- TIT sweep ---
TIT_arr = np.linspace(1073.15, 1873.15, 30)
TIT_results = [m.brayton_cycle(TIT=tit) for tit in TIT_arr]
eta_TIT = [r["eta_electrical"] for r in TIT_results]
W_TIT = [r["W_elec_MW"] for r in TIT_results]

# --- Part-load ---
load_arr = np.linspace(0.3, 1.0, 20)
load_results = [m.part_load(lf) for lf in load_arr]
eta_load = [r["eta_electrical"] for r in load_results]
W_load = [r["W_elec_MW"] for r in load_results]

# --- Ambient temperature ---
T_amb_arr = np.linspace(253.15, 323.15, 20)
T_amb_results = [m.brayton_cycle(T_amb=ta) for ta in T_amb_arr]
eta_amb = [r["eta_electrical"] for r in T_amb_results]
W_amb = [r["W_elec_MW"] for r in T_amb_results]

fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=[
        "PR Sweep: Electrical Efficiency",
        "PR Sweep: Power Output & Exhaust T",
        "TIT Sweep: Efficiency & Power",
        "Part-Load: Efficiency & Power",
        "Ambient T Effect: Efficiency",
        "Ambient T Effect: Power Output",
    ],
    vertical_spacing=0.10,
    horizontal_spacing=0.10,
    specs=[[{}, {}], [{}, {}], [{}, {}]],
)

# Row 1: PR sweep
fig.add_trace(go.Scatter(x=PR_arr, y=[e*100 for e in eta_PR], name="eta vs PR",
              line=dict(color="#1f77b4")), row=1, col=1)
fig.add_trace(go.Scatter(x=PR_arr, y=W_PR, name="W_elec vs PR",
              line=dict(color="#d62728")), row=1, col=2)
fig.add_trace(go.Scatter(x=PR_arr, y=T4_PR, name="T_exhaust vs PR",
              line=dict(color="#ff7f0e", dash="dot"), yaxis="y"), row=1, col=2)

# Row 2: TIT sweep + Part-load
fig.add_trace(go.Scatter(x=TIT_arr-273.15, y=[e*100 for e in eta_TIT], name="eta vs TIT",
              line=dict(color="#2ca02c")), row=2, col=1)
fig.add_trace(go.Scatter(x=load_arr*100, y=[e*100 for e in eta_load], name="eta vs load",
              line=dict(color="#9467bd")), row=2, col=2)

# Row 3: Ambient temperature
fig.add_trace(go.Scatter(x=T_amb_arr-273.15, y=[e*100 for e in eta_amb], name="eta vs T_amb",
              line=dict(color="#8c564b")), row=3, col=1)
fig.add_trace(go.Scatter(x=T_amb_arr-273.15, y=W_amb, name="W_elec vs T_amb",
              line=dict(color="#e377c2")), row=3, col=2)

for r, c, xl, yl in [
    (1, 1, "Pressure Ratio", "Electrical Efficiency (%)"),
    (1, 2, "Pressure Ratio", "Power (MW) / Exhaust T (C)"),
    (2, 1, "TIT (C)", "Electrical Efficiency (%)"),
    (2, 2, "Load Fraction (%)", "Electrical Efficiency (%)"),
    (3, 1, "Ambient Temperature (C)", "Electrical Efficiency (%)"),
    (3, 2, "Ambient Temperature (C)", "Power Output (MW)"),
]:
    fig.update_xaxes(title_text=xl, row=r, col=c)
    fig.update_yaxes(title_text=yl, row=r, col=c)

fig.update_layout(
    title=f"<b>EC109 Simple Cycle Gas Turbine -- F2a Brayton Cycle "
          f"(Design: {r_design['W_elec_MW']:.0f} MW, eta={r_design['eta_electrical']:.1%})</b>",
    height=1000, template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

out_path = os.path.join(OUTPUT_DIR, "EC109_F2a_brayton_cycle_report.html")
fig.write_html(out_path, include_plotlyjs="cdn")
print(f"Report saved: {out_path}")
