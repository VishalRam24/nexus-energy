"""
EC181 — Transmission Line — F2a Distributed-Parameter / Cascaded-Pi
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
    print("ERROR: plotly not installed. Run: pip install plotly")
    sys.exit(1)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "model_files")
os.makedirs(OUTPUT_DIR, exist_ok=True)

m = ComponentModel()

# --- Scenario A: voltage profile & efficiency vs load ---
loads = np.linspace(0.0, 2.0, 40)
Vr, eta, ploss = [], [], []
for P in loads:
    r = m.predict({"V_s_pu": 1.0, "P_load_pu": float(P), "Q_load_pu": 0.3 * float(P),
                   "length_km": 300.0})
    Vr.append(r["V_r_pu"]); eta.append(r["efficiency"]); ploss.append(r["P_loss_pu"])

# --- Scenario B: Ferranti vs line length ---
lengths = np.linspace(20.0, 600.0, 40)
rise = [m._model.ferranti_no_load(1.0, length_km=float(L))["rise_factor"] for L in lengths]

# --- Scenario C: dynamic cascaded-pi charging transient (open end) ---
sim = m.simulate({"V_s_pu": 1.0, "n_sections": 8, "length_km": 300.0,
                  "duration_s": 0.08, "open_end": True})

fig = make_subplots(rows=2, cols=2, subplot_titles=(
    "V_r & efficiency vs load (300 km)", "Line losses vs load",
    "Ferranti rise factor vs length (no load)", "Cascaded-pi open-end charging transient"))

fig.add_trace(go.Scatter(x=loads, y=Vr, name="V_r [pu]"), row=1, col=1)
fig.add_trace(go.Scatter(x=loads, y=eta, name="efficiency", yaxis="y2"), row=1, col=1)
fig.add_trace(go.Scatter(x=loads, y=ploss, name="P_loss [pu]"), row=1, col=2)
fig.add_trace(go.Scatter(x=lengths, y=rise, name="rise factor"), row=2, col=1)
fig.add_hline(y=1.0, line_dash="dash", row=2, col=1)
fig.add_trace(go.Scatter(x=sim["t"] * 1000, y=sim["v_s"], name="v_s(t)"), row=2, col=2)
fig.add_trace(go.Scatter(x=sim["t"] * 1000, y=sim["v_r"], name="v_r(t)"), row=2, col=2)

fig.update_layout(title="EC181 Transmission Line — F2a Distributed-Parameter Report",
                  height=800, showlegend=True)
out = os.path.join(OUTPUT_DIR, "simulation_report.html")
fig.write_html(out)
print(f"Report written: {out}")
