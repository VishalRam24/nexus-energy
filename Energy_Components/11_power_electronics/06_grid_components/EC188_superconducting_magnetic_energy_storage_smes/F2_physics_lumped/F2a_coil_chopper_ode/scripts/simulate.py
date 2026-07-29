"""
EC188 -- SMES -- F2a Physics-Lumped
Plotly HTML simulation report (optional; safe if plotly absent).
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "model_files")
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = ComponentModel()

# --- Scenario: charge from empty, hold, then discharge (voltage profile) ---
def v_profile(t):
    if t < 1.0:
        return 4000.0      # charge
    elif t < 2.0:
        return 0.0         # idle hold (supercon -> energy persists)
    else:
        return -4000.0     # discharge


r = model.predict({"I0_A": 0.0, "command": v_profile, "mode": "voltage",
                   "dt": 0.005, "duration_s": 3.0})

rt = model.round_trip(P_MW=1.0)
print(f"Round-trip efficiency @1 MW: {rt['eta_rt']*100:.2f} %")
print(f"Peak energy: {np.max(r['E_stored_MJ']):.3f} MJ, "
      f"peak current: {np.max(r['I_coil_A']):.1f} A")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("plotly not installed; skipping HTML report.")
    sys.exit(0)

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=("Coil current I(t)", "Stored energy E(t) = 0.5 L I^2",
                    "Chopper voltage V(t)", "Coil & grid power"),
)
fig.add_trace(go.Scatter(x=r["t"], y=r["I_coil_A"], name="I [A]"), 1, 1)
fig.add_trace(go.Scatter(x=r["t"], y=r["E_stored_MJ"], name="E [MJ]"), 1, 2)
fig.add_trace(go.Scatter(x=r["t"], y=r["V_chop_V"], name="V [V]"), 2, 1)
fig.add_trace(go.Scatter(x=r["t"], y=r["P_coil_MW"], name="P_coil [MW]"), 2, 2)
fig.add_trace(go.Scatter(x=r["t"], y=r["P_grid_MW"], name="P_grid [MW]"), 2, 2)
fig.update_layout(title="EC188 SMES F2a -- charge / hold / discharge cycle",
                  height=800, template="plotly_white")

out = os.path.join(OUTPUT_DIR, "simulation_report.html")
fig.write_html(out)
print(f"Report written to {out}")
