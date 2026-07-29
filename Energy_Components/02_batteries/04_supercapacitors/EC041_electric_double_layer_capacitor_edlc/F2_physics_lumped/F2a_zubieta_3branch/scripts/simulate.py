"""
EC041 -- EDLC Supercapacitor -- F2a Zubieta 3-Branch ECM
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
    print("plotly not installed; skipping HTML report. Run: pip install plotly")
    sys.exit(0)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "model_files")
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = ComponentModel()
m = model._model

# Scenario 1: charge 10 s then open-circuit rest -> redistribution + self-discharge
def prof(t):
    return 100.0 if t < 10.0 else 0.0

r1 = model.predict({"current_A": prof, "v0_V": 0.0, "dt": 0.1, "duration_s": 120.0})

# Scenario 2: high-rate charge/discharge cycling -> thermal rise
def cyc(t):
    return 250.0 if int(t) % 2 == 0 else -250.0

r2 = model.predict({"current_A": cyc, "v0_V": 1.0, "T0_K": 298.15, "dt": 0.05, "duration_s": 200.0})

# Scenario 3: C(V) curve and round-trip efficiency vs current
v_sweep = np.linspace(0.0, 2.7, 100)
C_curve = m.C_imm(v_sweep)
I_sweep = np.linspace(10.0, 350.0, 25)
eff_curve = [m.round_trip_efficiency(I, v_top=2.5)[0] for I in I_sweep]

fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=[
        "Charge + Redistribution: branch voltages",
        "Terminal V & stored energy (charge/rest)",
        "High-rate cycling: temperature rise",
        "Joule heat dissipation (cycling)",
        "Voltage-dependent capacitance C(V)",
        "Round-trip efficiency vs current",
    ],
    vertical_spacing=0.10, horizontal_spacing=0.10,
)

fig.add_trace(go.Scatter(x=r1["t"], y=r1["v1"], name="v1 immediate"), row=1, col=1)
fig.add_trace(go.Scatter(x=r1["t"], y=r1["v2"], name="v2 delayed"), row=1, col=1)
fig.add_trace(go.Scatter(x=r1["t"], y=r1["v3"], name="v3 long-term"), row=1, col=1)

fig.add_trace(go.Scatter(x=r1["t"], y=r1["v_terminal"], name="V_term", showlegend=False), row=1, col=2)
fig.add_trace(go.Scatter(x=r1["t"], y=r1["energy_J"], name="E (J)", yaxis="y2", showlegend=False), row=1, col=2)

fig.add_trace(go.Scatter(x=r2["t"], y=r2["temperature"], name="T(t)", showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=r2["t"], y=r2["heat_W"], name="Q_gen", showlegend=False), row=2, col=2)

fig.add_trace(go.Scatter(x=v_sweep, y=C_curve, name="C(V)", showlegend=False), row=3, col=1)
fig.add_trace(go.Scatter(x=I_sweep, y=eff_curve, name="eff", showlegend=False), row=3, col=2)

for r, c, xl, yl in [
    (1, 1, "Time (s)", "Branch V (V)"),
    (1, 2, "Time (s)", "V_term (V) / E (J)"),
    (2, 1, "Time (s)", "Temperature (K)"),
    (2, 2, "Time (s)", "Heat (W)"),
    (3, 1, "Voltage (V)", "Capacitance (F)"),
    (3, 2, "Current (A)", "Round-trip eff (-)"),
]:
    fig.update_xaxes(title_text=xl, row=r, col=c)
    fig.update_yaxes(title_text=yl, row=r, col=c)

fig.update_layout(
    title="<b>EC041 EDLC -- F2a Zubieta 3-Branch ECM + Thermal ODE</b>",
    height=1000, template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

out_path = os.path.join(OUTPUT_DIR, "EC041_F2a_zubieta_3branch_report.html")
fig.write_html(out_path, include_plotlyjs="cdn")
print(f"Report saved: {out_path}")
