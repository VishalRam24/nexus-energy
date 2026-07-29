"""
EC130 -- Small/Micro Hydropower -- F2a Penstock Transient
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
    _HAVE_PLOTLY = True
except ImportError:
    _HAVE_PLOTLY = False

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "model_files")
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = ComponentModel()

# --- Scenario 1: cold start to full load (water-column acceleration) ---
r1 = model.predict({"gate_command": 1.0, "v0": 0.1, "dt": 0.2, "duration_s": 120.0})

# --- Scenario 2: gate step 0.5 -> 1.0 (load pickup transient) ---
def gate_step(t):
    return 0.5 if t < 20.0 else 1.0

r2 = model._model.simulate(gate_step, dt=0.1, duration_s=120.0)

# --- Scenario 3: load rejection with surge (gate cut, supply held) ---
def gate_reject(t):
    return 1.0 if t < 30.0 else 0.4

r3 = model._model.simulate(gate_reject, Q_in=model._model.Q_design,
                           dt=0.1, duration_s=250.0)

if not _HAVE_PLOTLY:
    print("plotly not installed; skipping HTML report.")
    print(f"S1 final P={r1['power_el'][-1]:.1f} kW; "
          f"S2 final Q={r2['flow'][-1]:.3f} m3/s; "
          f"S3 max surge={np.max(np.abs(r3['surge_level'])):.3f} m")
    sys.exit(0)

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "Cold start: penstock velocity & power",
        "Gate step 0.5->1.0: flow & power",
        "Load rejection: surge level",
        "Net head vs head loss (cold start)",
    ),
)

fig.add_trace(go.Scatter(x=r1["t"], y=r1["velocity"], name="v [m/s]"), 1, 1)
fig.add_trace(go.Scatter(x=r1["t"], y=r1["power_el"], name="P_el [kW]", yaxis="y2"), 1, 1)

fig.add_trace(go.Scatter(x=r2["t"], y=r2["flow"], name="Q [m3/s]"), 1, 2)
fig.add_trace(go.Scatter(x=r2["t"], y=r2["power_el"], name="P_el [kW]"), 1, 2)

fig.add_trace(go.Scatter(x=r3["t"], y=r3["surge_level"], name="z surge [m]"), 2, 1)

fig.add_trace(go.Scatter(x=r1["t"], y=r1["head_net"], name="H_net [m]"), 2, 2)
fig.add_trace(go.Scatter(x=r1["t"], y=r1["head_loss"], name="H_loss [m]"), 2, 2)

fig.update_layout(
    title="EC130 Micro-Hydro F2a -- Penstock Transient Report",
    height=800, showlegend=True,
)

out = os.path.join(OUTPUT_DIR, "simulation_report.html")
fig.write_html(out)
print(f"Report written: {out}")
