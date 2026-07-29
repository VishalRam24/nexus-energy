"""
EC200 -- Oxy-Fuel Combustion Capture -- F1a Capture & Energy Model
Plotly HTML report (optional; import wrapped so absence doesn't crash).
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
    print("plotly not installed -- skipping HTML report.")
    sys.exit(0)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "model_files")
os.makedirs(OUTPUT_DIR, exist_ok=True)

m = ComponentModel()

fuels = np.linspace(5.0, 90.0, 60)
r = m.predict({"fuel_rate": fuels})

loads = np.linspace(0.4, 1.0, 60)
eff_drop = np.array([float(m.predict({"fuel_rate": 50.0, "load": float(l)})["efficiency_drop_pts"])
                     for l in loads]) * 100.0

purities = np.linspace(0.90, 0.995, 60)
asu_e = np.array([float(m._model.asu_specific_energy(float(p))) for p in purities])

fig = make_subplots(rows=2, cols=2, subplot_titles=[
    "CO2 produced / captured vs fuel rate",
    "Parasitic power (ASU + compression) vs fuel rate",
    "Efficiency drop vs plant load",
    "ASU specific energy vs O2 purity",
])

fig.add_trace(go.Scatter(x=fuels, y=r["co2_produced_kgs"], name="CO2 produced"), row=1, col=1)
fig.add_trace(go.Scatter(x=fuels, y=r["co2_captured_kgs"], name="CO2 captured"), row=1, col=1)
fig.add_trace(go.Scatter(x=fuels, y=r["asu_power_mw"], name="ASU"), row=1, col=2)
fig.add_trace(go.Scatter(x=fuels, y=r["compression_power_mw"], name="Compression"), row=1, col=2)
fig.add_trace(go.Scatter(x=loads, y=eff_drop, name="eff drop (pts)", showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=purities, y=asu_e, name="ASU kWh/tO2", showlegend=False), row=2, col=2)

fig.update_xaxes(title_text="Fuel rate (kg/s)", row=1, col=1)
fig.update_yaxes(title_text="CO2 (kg/s)", row=1, col=1)
fig.update_xaxes(title_text="Fuel rate (kg/s)", row=1, col=2)
fig.update_yaxes(title_text="Power (MW)", row=1, col=2)
fig.update_xaxes(title_text="Load (-)", row=2, col=1)
fig.update_yaxes(title_text="Efficiency drop (pts)", row=2, col=1)
fig.update_xaxes(title_text="O2 purity (mol/mol)", row=2, col=2)
fig.update_yaxes(title_text="ASU energy (kWh/tO2)", row=2, col=2)

fig.update_layout(title="<b>EC200 Oxy-Fuel Combustion Capture -- F1a</b>",
                  height=800, template="plotly_white")

out = os.path.join(OUTPUT_DIR, "EC200_F1a_capture_energy_report.html")
fig.write_html(out, include_plotlyjs="cdn")
print(f"Report saved: {out}")
