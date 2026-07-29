"""
EC136 -- Overtopping Device WEC (Wave Dragon) -- F2a Physics-Lumped Reservoir Dynamics
Plotly HTML simulation report generator (optional; safe if plotly absent).
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "model_files")
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = ComponentModel()

# --- Scenario 1: steady design sea state, reservoir fill/drain transient ---
r1 = model.predict({"Hs_m": 3.0, "Tz_s": 7.0, "level0_m": 0.0, "dt": 5.0, "duration_s": 2400.0})

# --- Scenario 2: rising storm (time-varying Hm0) ---
def Hm0_storm(t):
    return 1.5 + 3.0 * (t / 3600.0)  # 1.5 m -> 4.5 m over an hour

r2 = model._model.simulate(Hm0_storm, 8.0, level0=0.5, dt=5.0, duration_s=3600.0)

# --- Scenario 3: power & efficiency vs sea state ---
Hs_sweep = np.linspace(0.5, 6.0, 16)
P_sweep, eta_sweep = [], []
for Hs in Hs_sweep:
    r = model.predict({"Hs_m": float(Hs), "Tz_s": 7.0, "dt": 15.0, "duration_s": 1800.0})
    P_sweep.append(r["P_mean_kW"])
    eta_sweep.append(r["eta_overall"] * 100.0)

print(f"Scenario 1: mean power {r1['P_mean_kW']:.1f} kW, eta {r1['eta_overall']*100:.1f} %")
print(f"Scenario 2 (storm): mean power {r2['P_mean_kW']:.1f} kW")
print(f"Scenario 3: P range {min(P_sweep):.1f}-{max(P_sweep):.1f} kW")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("plotly not installed -- skipping HTML report (computation OK).")
    sys.exit(0)

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "Reservoir level & flows (Hs=3 m)",
        "Electrical power (Hs=3 m)",
        "Rising storm: level vs time",
        "Mean power & efficiency vs Hs",
    ),
    specs=[[{"secondary_y": True}, {}], [{}, {"secondary_y": True}]],
)

fig.add_trace(go.Scatter(x=r1["t"], y=r1["level"], name="level [m]"), row=1, col=1)
fig.add_trace(go.Scatter(x=r1["t"], y=r1["Q_in"], name="Q_in [m3/s]"), row=1, col=1, secondary_y=True)
fig.add_trace(go.Scatter(x=r1["t"], y=r1["Q_out"], name="Q_out [m3/s]"), row=1, col=1, secondary_y=True)

fig.add_trace(go.Scatter(x=r1["t"], y=r1["power_elec_W"] / 1e3, name="P_elec [kW]"), row=1, col=2)

fig.add_trace(go.Scatter(x=r2["t"], y=r2["level"], name="storm level [m]"), row=2, col=1)

fig.add_trace(go.Scatter(x=Hs_sweep, y=P_sweep, name="P_mean [kW]"), row=2, col=2)
fig.add_trace(go.Scatter(x=Hs_sweep, y=eta_sweep, name="eta [%]"), row=2, col=2, secondary_y=True)

fig.update_layout(title="EC136 Overtopping WEC -- F2a Reservoir Dynamics", height=800)
out = os.path.join(OUTPUT_DIR, "simulation_report.html")
fig.write_html(out)
print(f"Report written: {out}")
