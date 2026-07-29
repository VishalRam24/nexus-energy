"""
EC152 -- Flash Steam Geothermal Plant -- F2a Flash Thermodynamics
Optional Plotly HTML simulation report. Plotly import is guarded.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel
from model import FlashSteamGeothermalF2a

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAVE_PLOTLY = True
except Exception:
    HAVE_PLOTLY = False

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "model_files")
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = ComponentModel()
m = model._model

# Scenario 1: cold-start transient at design point
r1 = model.predict({"m_dot_brine_kgs": 100.0, "T_geo_C": 240.0, "T_reject_C": 50.0,
                    "dt": 0.5, "duration_s": 150.0})

# Scenario 2: brine-flow step
def step(t):
    return 60.0 if t < 60.0 else 120.0
r2 = m.simulate(step, 240.0, 50.0, dt=0.5, duration_s=160.0)

# Scenario 3: flash-temperature sweep (optimal flash hump)
T_geo, T_rej = 240.0, 50.0
Tfl = np.arange(90.0, 226.0, 4.0)
P_sweep = np.array([float(m.net_power_ss(T_geo, T_rej, 100.0, T)) for T in Tfl])
T_opt = float(m.optimal_flash_temperature(T_geo, T_rej))

print(f"Design steady power: {r1['P_steady_kW']:.0f} kW, "
      f"eta_util={r1['eta_utilization'][-1]:.3f}, Carnot={r1['eta_carnot'][-1]:.3f}")
print(f"Optimal flash T (analytic): {T_opt:.1f} C, "
      f"grid argmax: {Tfl[int(np.argmax(P_sweep))]:.1f} C")

if not HAVE_PLOTLY:
    print("plotly not installed; skipping HTML report.")
    sys.exit(0)

fig = make_subplots(rows=2, cols=2, subplot_titles=(
    "Cold-start transient (net power)", "Brine-flow step response",
    "Optimal-flash power hump", "Steam flow vs time (step)"))

fig.add_trace(go.Scatter(x=r1["t"], y=r1["net_power_kW"], name="P_net"), 1, 1)
fig.add_trace(go.Scatter(x=r2["t"], y=r2["net_power_kW"], name="P_net (step)"), 1, 2)
fig.add_trace(go.Scatter(x=Tfl, y=P_sweep, name="P vs T_flash"), 2, 1)
fig.add_vline(x=T_opt, line_dash="dash", row=2, col=1)
fig.add_trace(go.Scatter(x=r2["t"], y=r2["steam_flow_kgs"], name="steam flow"), 2, 2)

fig.update_layout(title="EC152 Flash Steam Geothermal -- F2a Flash Thermodynamics",
                  height=720, showlegend=False)
out = os.path.join(OUTPUT_DIR, "simulation_report.html")
fig.write_html(out)
print(f"Report written: {out}")
