"""
EC089 -- Hydrogen Boiler -- F2a Physics-Lumped
Optional Plotly HTML report. Plotly import is wrapped so absence won't crash.
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
except Exception:
    _HAVE_PLOTLY = False

OUTPUT = os.path.join(os.path.dirname(__file__), "..", "simulation_report.html")

model = ComponentModel()

# Cold-start transient at full fire.
r1 = model.predict({"firing_rate": 1.0, "T_water_K": 300.0, "dt": 5.0, "duration_s": 1200.0})

# Firing-rate step.
def step_phi(t):
    return 0.3 if t < 600 else 1.0

r2 = model._model.simulate(step_phi, 310.0, 5.0, 1200.0)

# Efficiency vs firing rate (condensing) and vs excess air sweep.
phis = np.linspace(0.1, 1.0, 30)
eta_hhv = [model._model.efficiency_hhv(p) for p in phis]
eta_lhv = [model._model.efficiency_lhv(p) for p in phis]

lams = np.linspace(1.05, 3.0, 30)
T_flame = [model._model.adiabatic_flame_temp(l) for l in lams]
nox = [model._model.nox_index(l) for l in lams]

print(f"Cold-start final water T: {r1['temperature'][-1]:.2f} K, "
      f"eta_HHV={r1['efficiency'][-1]:.4f}")
print(f"Adiabatic flame T (design): {r1['T_adiabatic_flame_K']:.0f} K")

if not _HAVE_PLOTLY:
    print("plotly not installed -- skipping HTML report (data computed OK).")
    sys.exit(0)

fig = make_subplots(rows=2, cols=2, subplot_titles=(
    "Cold-start water temperature", "Firing-rate step response",
    "Efficiency vs firing rate (condensing)", "Flame temp & NOx vs excess air"))

fig.add_trace(go.Scatter(x=r1["t"], y=r1["temperature"], name="T_water"), row=1, col=1)
fig.add_trace(go.Scatter(x=r2["t"], y=r2["temperature"], name="T_water (step)"), row=1, col=2)
fig.add_trace(go.Scatter(x=phis, y=eta_hhv, name="eta HHV"), row=2, col=1)
fig.add_trace(go.Scatter(x=phis, y=eta_lhv, name="eta LHV"), row=2, col=1)
fig.add_trace(go.Scatter(x=lams, y=T_flame, name="T_flame [K]"), row=2, col=2)
fig.add_trace(go.Scatter(x=lams, y=nox, name="NOx index", yaxis="y2"), row=2, col=2)

fig.update_layout(title="EC089 Hydrogen Boiler -- F2a Physics-Lumped", height=800)
fig.write_html(OUTPUT)
print(f"Report written to {OUTPUT}")
