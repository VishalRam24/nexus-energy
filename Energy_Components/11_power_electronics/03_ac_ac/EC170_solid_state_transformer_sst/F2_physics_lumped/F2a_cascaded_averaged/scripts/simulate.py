"""
EC170 -- Solid State Transformer (SST) -- F2a Cascaded Averaged Model
Plotly HTML simulation report generator (optional; safe if plotly absent).
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _HAVE_PLOTLY = True
except ImportError:
    print("plotly not installed -- skipping HTML report (model still usable).")
    _HAVE_PLOTLY = False

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "model_files")
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = ComponentModel()
m = model._model

# --- Scenario 1: forward power step, DC-link transient ---
def cmd_step(t):
    return 0.0 if t < 0.02 else 8000.0

r1 = m.simulate(cmd_step, power_factor=1.0, dt=0.0005, duration_s=0.2)

# --- Scenario 2: efficiency vs load (cascade product) ---
loads = np.linspace(200.0, 12000.0, 80)
eta = np.array([float(m.total_efficiency(p)) for p in loads])
er = np.array([float(m.stage_efficiencies(p)[0]) for p in loads])
ed = np.array([float(m.stage_efficiencies(p)[1]) for p in loads])
ei = np.array([float(m.stage_efficiencies(p)[2]) for p in loads])

# --- Scenario 3: reverse power step ---
r3 = m.simulate(-7000.0, power_factor=1.0, dt=0.0005, duration_s=0.15)

if _HAVE_PLOTLY:
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Forward step: DC-link voltages",
            "Cascaded efficiency vs load (product of 3 stages)",
            "Forward step: DAB power tracking",
            "Reverse step: DAB power",
        ),
    )
    fig.add_trace(go.Scatter(x=r1["t"], y=r1["v_hv_dc"], name="V_hv_dc"), 1, 1)
    fig.add_trace(go.Scatter(x=r1["t"], y=r1["v_lv_dc"], name="V_lv_dc"), 1, 1)

    fig.add_trace(go.Scatter(x=loads, y=eta, name="eta_total"), 1, 2)
    fig.add_trace(go.Scatter(x=loads, y=er, name="eta_rect", line=dict(dash="dot")), 1, 2)
    fig.add_trace(go.Scatter(x=loads, y=ed, name="eta_dab", line=dict(dash="dot")), 1, 2)
    fig.add_trace(go.Scatter(x=loads, y=ei, name="eta_inv", line=dict(dash="dot")), 1, 2)

    fig.add_trace(go.Scatter(x=r1["t"], y=r1["p_dab_w"], name="P_dab fwd"), 2, 1)
    fig.add_trace(go.Scatter(x=r1["t"], y=r1["p_command_w"], name="P_cmd",
                             line=dict(dash="dash")), 2, 1)

    fig.add_trace(go.Scatter(x=r3["t"], y=r3["p_dab_w"], name="P_dab rev"), 2, 2)

    fig.update_layout(height=800, width=1200,
                      title_text="EC170 SST F2a -- Cascaded Averaged Three-Stage Model")
    out = os.path.join(OUTPUT_DIR, "simulation_report.html")
    fig.write_html(out)
    print(f"Report written to {out}")
