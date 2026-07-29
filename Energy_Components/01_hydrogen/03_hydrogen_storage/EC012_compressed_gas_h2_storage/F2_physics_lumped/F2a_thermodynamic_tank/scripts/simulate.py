"""
EC012 -- Compressed Gas H2 Storage -- F2a Thermodynamic Tank
Plotly HTML simulation report (optional; degrades gracefully without plotly).
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

# Scenario 1: fast fill from 20 -> high pressure, pre-cooled inlet
r_fill = model.predict({
    "mdot_kg_s": 0.012, "T0_K": 298.15, "T_amb_K": 298.15,
    "T_in_K": 233.15, "P0_bar": 20.0, "dt": 1.0, "duration_s": 200.0,
})

# Scenario 2: post-fill cooldown (no flow)
r_cool = model.predict({
    "mdot_kg_s": 0.0, "T0_K": float(r_fill["temperature"][-1]),
    "T_amb_K": 298.15, "m0_kg": float(r_fill["mass"][-1]),
    "dt": 10.0, "duration_s": 3600.0,
})

# Scenario 3: discharge from full
m_full = float(model._model.mass_from_PT(700.0, 298.15))
r_dis = model.predict({
    "mdot_kg_s": -0.004, "T0_K": 298.15, "T_amb_K": 298.15,
    "m0_kg": m_full, "dt": 5.0, "duration_s": 600.0,
})

if not _HAVE_PLOTLY:
    print("plotly not installed; skipping HTML report. Summary:")
    print(f"  fill:  P {r_fill['pressure'][0]:.0f}->{r_fill['pressure'][-1]:.0f} bar, "
          f"T {r_fill['temperature'][0]:.0f}->{r_fill['temperature'][-1]:.0f} K, "
          f"SOC {r_fill['soc'][-1]:.2f}")
    print(f"  cool:  T {r_cool['temperature'][0]:.0f}->{r_cool['temperature'][-1]:.0f} K, "
          f"P {r_cool['pressure'][0]:.0f}->{r_cool['pressure'][-1]:.0f} bar")
    print(f"  disch: P {r_dis['pressure'][0]:.0f}->{r_dis['pressure'][-1]:.0f} bar, "
          f"SOC {r_dis['soc'][0]:.2f}->{r_dis['soc'][-1]:.2f}")
    sys.exit(0)

fig = make_subplots(rows=2, cols=2, subplot_titles=(
    "Fast fill: pressure & temperature", "Fast fill: mass & SOC",
    "Post-fill cooldown", "Discharge from full"))

fig.add_trace(go.Scatter(x=r_fill["t"], y=r_fill["pressure"], name="P [bar]"), 1, 1)
fig.add_trace(go.Scatter(x=r_fill["t"], y=r_fill["temperature"], name="T_gas [K]", yaxis="y2"), 1, 1)
fig.add_trace(go.Scatter(x=r_fill["t"], y=r_fill["mass"], name="mass [kg]"), 1, 2)
fig.add_trace(go.Scatter(x=r_fill["t"], y=r_fill["soc"], name="SOC"), 1, 2)
fig.add_trace(go.Scatter(x=r_cool["t"], y=r_cool["temperature"], name="T_gas cool [K]"), 2, 1)
fig.add_trace(go.Scatter(x=r_cool["t"], y=r_cool["T_wall"], name="T_wall [K]"), 2, 1)
fig.add_trace(go.Scatter(x=r_dis["t"], y=r_dis["pressure"], name="P discharge [bar]"), 2, 2)

fig.update_layout(title="EC012 F2a Lumped Thermodynamic Tank", height=800)
out = os.path.join(OUTPUT_DIR, "simulation_report.html")
fig.write_html(out)
print(f"Wrote {out}")
