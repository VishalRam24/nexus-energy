"""
EC200 -- Oxy-Fuel Combustion Capture -- F2a Combustion + Recycle Model
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

# Scenario 1: thermal transient at recycle = 0.6
r1 = m.predict({"mdot_fuel": 50.0, "recycle_ratio": 0.6, "T0": 1000.0,
                "dt": 0.5, "duration_s": 200.0})

# Scenario 2: adiabatic flame temp vs recycle ratio
Rs = np.linspace(0.0, 0.8, 40)
T_ad = np.array([m._model.adiabatic_flame_temp(50.0, float(R)) for R in Rs])
T_ss = np.array([m.predict({"mdot_fuel": 50.0, "recycle_ratio": float(R),
                            "duration_s": 250.0})["T_steady"] for R in Rs])

# Scenario 3: temperature transients at several recycle ratios
fig = make_subplots(rows=2, cols=2, subplot_titles=[
    "Furnace temperature transient (R=0.6)",
    "Adiabatic & steady T vs recycle ratio",
    "Temperature transients for several recycle ratios",
    "Flue-gas CO2 purity (dry vs wet)",
])

fig.add_trace(go.Scatter(x=r1["t"], y=r1["temperature"], name="T(t)"), row=1, col=1)
fig.add_trace(go.Scatter(x=Rs, y=T_ad, name="T_adiabatic"), row=1, col=2)
fig.add_trace(go.Scatter(x=Rs, y=T_ss, name="T_steady"), row=1, col=2)

for R in [0.0, 0.3, 0.6, 0.8]:
    rr = m.predict({"mdot_fuel": 50.0, "recycle_ratio": R, "T0": 1000.0,
                    "dt": 0.5, "duration_s": 200.0})
    fig.add_trace(go.Scatter(x=rr["t"], y=rr["temperature"],
                  name=f"R={R}"), row=2, col=1)

p_dry = m.predict({"mdot_fuel": 50.0})["co2_purity_dry"]
p_wet = m.predict({"mdot_fuel": 50.0})["co2_purity_wet"]
fig.add_trace(go.Bar(x=["wet (pre-knockout)", "dry (post-knockout)"],
              y=[p_wet, p_dry], showlegend=False), row=2, col=2)

fig.update_xaxes(title_text="Time (s)", row=1, col=1)
fig.update_yaxes(title_text="T (K)", row=1, col=1)
fig.update_xaxes(title_text="Recycle ratio (-)", row=1, col=2)
fig.update_yaxes(title_text="T (K)", row=1, col=2)
fig.update_xaxes(title_text="Time (s)", row=2, col=1)
fig.update_yaxes(title_text="T (K)", row=2, col=1)
fig.update_yaxes(title_text="CO2 mole fraction", row=2, col=2)

fig.update_layout(title="<b>EC200 Oxy-Fuel Combustion Capture -- F2a (lumped ODE)</b>",
                  height=850, template="plotly_white")

out = os.path.join(OUTPUT_DIR, "EC200_F2a_combustion_recycle_report.html")
fig.write_html(out, include_plotlyjs="cdn")
print(f"Report saved: {out}")
