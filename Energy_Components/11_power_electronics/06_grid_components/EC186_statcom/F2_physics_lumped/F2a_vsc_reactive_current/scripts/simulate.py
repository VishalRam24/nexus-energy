"""
EC186 -- STATCOM -- F2a VSC Reactive-Current Control
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
    print("plotly not installed -- skipping HTML report. (pip install plotly)")
    sys.exit(0)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "model_files")
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = ComponentModel()

# Scenario 1: capacitive step +100 MVAR -- fast response
r1 = model.predict({"Q_ref_MVAR": 100.0, "V_bus_pu": 1.0, "dt": 5e-5, "duration_s": 0.03})

# Scenario 2: command step from inductive to capacitive
def q_step(t):
    return -80.0 if t < 0.02 else 80.0

r2 = model.predict({"Q_ref_MVAR": q_step, "V_bus_pu": 1.0, "dt": 5e-5, "duration_s": 0.05})

# Scenario 3: constant-current capability -- voltage sag during full command
def v_sag(t):
    return 1.0 if t < 0.02 else 0.4

r3 = model.predict({"Q_ref_MVAR": 100.0, "V_bus_pu": v_sag, "dt": 5e-5, "duration_s": 0.05})

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "Capacitive step: Q response (sub-cycle)",
        "Inductive->Capacitive command reversal",
        "Constant-current under voltage sag (vs SVC)",
        "DC-link voltage regulation",
    ),
)

fig.add_trace(go.Scatter(x=r1["t"] * 1e3, y=r1["Q_out_MVAR"], name="Q_out"), 1, 1)
fig.add_trace(go.Scatter(x=r2["t"] * 1e3, y=r2["Q_out_MVAR"], name="Q reversal"), 1, 2)
fig.add_trace(go.Scatter(x=r3["t"] * 1e3, y=r3["I_mag"], name="I_mag"), 2, 1)
fig.add_trace(go.Scatter(x=r3["t"] * 1e3, y=r3["V_bus_pu"] * 100, name="V_bus %",
                         yaxis="y2"), 2, 1)
fig.add_trace(go.Scatter(x=r1["t"] * 1e3, y=r1["Vdc"] / 1e3, name="Vdc"), 2, 2)

fig.update_xaxes(title_text="time [ms]")
fig.update_layout(title="EC186 STATCOM F2a -- VSC Reactive-Current Control",
                  height=800, showlegend=True)

out = os.path.join(OUTPUT_DIR, "simulation_report.html")
fig.write_html(out)
print(f"Report written: {out}")
