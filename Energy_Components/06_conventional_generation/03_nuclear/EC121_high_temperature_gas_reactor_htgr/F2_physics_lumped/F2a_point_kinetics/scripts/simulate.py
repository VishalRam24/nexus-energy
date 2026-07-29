"""
EC121 -- High Temperature Gas Reactor (HTGR) -- F2a Point Kinetics + Lumped Thermal
Plotly HTML simulation report generator (optional; gracefully skips if no plotly).
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel
from model import PCM

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

# --- Scenario 1: +150 pcm reactivity insertion (negative feedback arrests rise) ---
r1 = model.predict({"rho_ext_pcm": 150.0, "duration_s": 3000.0})

# --- Scenario 2: control-rod insertion, -300 pcm (power down-transient) ---
r2 = model.predict({"rho_ext_pcm": -300.0, "duration_s": 3000.0})

# --- Scenario 3: passive decay-heat removal after scram ---
r3 = m.decay_heat_simulate(duration_s=40000.0)

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "+150 pcm: power self-limits (negative feedback)",
        "Node temperatures (+150 pcm)",
        "-300 pcm rod insertion: power down-transient",
        "Passive decay-heat removal (post-scram core T)",
    ),
)

fig.add_trace(go.Scatter(x=r1["t"], y=r1["power_fraction"], name="P/P_rated"),
              row=1, col=1)
fig.add_trace(go.Scatter(x=r1["t"], y=r1["T_fuel_K"] - 273.15, name="Fuel"),
              row=1, col=2)
fig.add_trace(go.Scatter(x=r1["t"], y=r1["T_graphite_K"] - 273.15, name="Graphite"),
              row=1, col=2)
fig.add_trace(go.Scatter(x=r1["t"], y=r1["T_helium_outlet_C"], name="He outlet"),
              row=1, col=2)
fig.add_trace(go.Scatter(x=r2["t"], y=r2["power_fraction"], name="P/P_rated (-300pcm)"),
              row=2, col=1)
fig.add_trace(go.Scatter(x=r3["t"], y=r3["T_core_C"], name="Core T (decay)"),
              row=2, col=2)
fig.add_hline(y=1600, line_dash="dash", line_color="red", row=2, col=2,
              annotation_text="TRISO limit 1600C")

fig.update_layout(height=800, width=1100,
                  title_text="EC121 HTGR F2a — Point Kinetics + Lumped Thermal")
out = os.path.join(OUTPUT_DIR, "simulation_report.html")
fig.write_html(out)
print(f"Report written to {out}")
