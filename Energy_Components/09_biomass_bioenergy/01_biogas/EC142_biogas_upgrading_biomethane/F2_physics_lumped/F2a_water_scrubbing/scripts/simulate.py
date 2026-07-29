"""
EC142 -- Biogas Upgrading to Biomethane -- F2a Water Scrubbing
Plotly HTML simulation report generator (optional; plotly wrapped in try/except).
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "model_files")
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = ComponentModel()

# Scenario 1: design-point transient to steady state
r1 = model.predict({"biogas_flow_Nm3_per_h": 500.0, "CH4_fraction_in": 0.60,
                    "dt": 2.0, "duration_s": 300.0})

# Scenario 2: sweep CO2 removal / purity vs biogas throughput
flows = np.linspace(100.0, 1500.0, 25)
pur, rec, slip, sec = [], [], [], []
for Q in flows:
    r = model.predict({"biogas_flow_Nm3_per_h": float(Q), "CH4_fraction_in": 0.60,
                       "dt": 10.0, "duration_s": 200.0})
    pur.append(r["purity_CH4_ss"])
    rec.append(r["CH4_recovery_ss"])
    slip.append(r["CH4_slip_ss"])
    sec.append(r["SEC_kWh_per_Nm3"])

print("Design point (500 Nm3/h, 60% CH4):")
print(f"  purity={r1['purity_CH4_ss']:.4f}  recovery={r1['CH4_recovery_ss']:.4f}  "
      f"slip={r1['CH4_slip_ss']:.4f}  CO2_removal={r1['CO2_removal_ss']:.4f}  "
      f"SEC={r1['SEC_kWh_per_Nm3']:.3f} kWh/Nm3")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("plotly not installed -- skipping HTML report (scenarios computed OK).")
    sys.exit(0)

fig = make_subplots(rows=2, cols=2, subplot_titles=(
    "CO2 removal transient", "Liquid dissolved conc.",
    "Purity & recovery vs biogas flow", "Methane slip & SEC vs flow"))

fig.add_trace(go.Scatter(x=r1["t"], y=r1["CO2_removal"], name="CO2 removal"), 1, 1)
fig.add_trace(go.Scatter(x=r1["t"], y=r1["purity_CH4"], name="CH4 purity"), 1, 1)
fig.add_trace(go.Scatter(x=r1["t"], y=r1["C_CO2_liquid"], name="C_CO2 [mol/m3]"), 1, 2)
fig.add_trace(go.Scatter(x=r1["t"], y=r1["C_CH4_liquid"], name="C_CH4 [mol/m3]"), 1, 2)
fig.add_trace(go.Scatter(x=flows, y=pur, name="purity"), 2, 1)
fig.add_trace(go.Scatter(x=flows, y=rec, name="recovery"), 2, 1)
fig.add_trace(go.Scatter(x=flows, y=slip, name="CH4 slip"), 2, 2)
fig.add_trace(go.Scatter(x=flows, y=sec, name="SEC [kWh/Nm3]"), 2, 2)

fig.update_layout(title="EC142 Biogas Upgrading F2a -- High-Pressure Water Scrubbing",
                  height=800, template="plotly_white")
out = os.path.join(OUTPUT_DIR, "simulation_report.html")
fig.write_html(out)
print(f"Report written: {out}")
