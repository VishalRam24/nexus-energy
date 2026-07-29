"""
EC056 -- Linear Fresnel CSP -- F2a Physics-Lumped Receiver Thermal ODE
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
    print("plotly not installed -- skipping HTML report. (pip install plotly)")
    sys.exit(0)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "model_files")
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = ComponentModel()
m = model._model

# Scenario 1: cold-start thermal transient
r1 = model.predict({"dni": 850.0, "theta_L_deg": 0.0, "theta_T_deg": 0.0,
                    "T_htf_in_C": 200.0, "T_wall0_C": 30.0,
                    "dt": 5.0, "duration_s": 2400.0})

# Scenario 2: passing cloud (DNI drop), wall response
def dni_cloud(t):
    return 200.0 if 900.0 < t < 1500.0 else 900.0
r2 = model.predict({"dni": dni_cloud, "theta_L_deg": 0.0, "theta_T_deg": 0.0,
                    "T_htf_in_C": 200.0, "T_wall0_C": 200.0,
                    "dt": 5.0, "duration_s": 3000.0})

# Scenario 3: IAM surface (optical efficiency vs angles)
tL = np.linspace(0, 70, 60)
tT = np.linspace(0, 55, 60)
eta_L = [m.optical_efficiency(a, 0.0) for a in tL]
eta_T = [m.optical_efficiency(0.0, a) for a in tT]

# Scenario 4: steady thermal efficiency vs DNI
dni_sweep = np.linspace(100, 1100, 40)
eta_th = []
for D in dni_sweep:
    _, d = m.steady_wall_temp(D, 0.0, 0.0, 298.15, 473.15)
    eta_th.append(d["eta_thermal"])

fig = make_subplots(rows=2, cols=2, subplot_titles=[
    "Cold-start wall + HTF outlet transient (DNI=850)",
    "Cloud transient (DNI 900->200->900)",
    "Optical efficiency vs incidence angle (IAM)",
    "Steady thermal efficiency vs DNI",
], vertical_spacing=0.12, horizontal_spacing=0.10)

fig.add_trace(go.Scatter(x=r1["t"], y=r1["T_wall_C"], name="T_wall"), row=1, col=1)
fig.add_trace(go.Scatter(x=r1["t"], y=r1["T_htf_out_C"], name="T_htf_out"), row=1, col=1)

fig.add_trace(go.Scatter(x=r2["t"], y=r2["T_wall_C"], name="T_wall (cloud)",
              showlegend=False), row=1, col=2)

fig.add_trace(go.Scatter(x=tL, y=eta_L, name="vs theta_L"), row=2, col=1)
fig.add_trace(go.Scatter(x=tT, y=eta_T, name="vs theta_T"), row=2, col=1)

fig.add_trace(go.Scatter(x=dni_sweep, y=eta_th, name="eta_thermal",
              showlegend=False), row=2, col=2)

for r, c, xl, yl in [(1, 1, "Time (s)", "Temp (C)"),
                     (1, 2, "Time (s)", "Wall Temp (C)"),
                     (2, 1, "Incidence angle (deg)", "Optical efficiency"),
                     (2, 2, "DNI (W/m2)", "Thermal efficiency")]:
    fig.update_xaxes(title_text=xl, row=r, col=c)
    fig.update_yaxes(title_text=yl, row=r, col=c)

fig.update_layout(title="<b>EC056 Linear Fresnel CSP -- F2a Lumped Receiver Thermal ODE</b>",
                  height=850, template="plotly_white")

out_path = os.path.join(OUTPUT_DIR, "EC056_F2a_receiver_thermal_ode_report.html")
fig.write_html(out_path, include_plotlyjs="cdn")
print(f"Report saved: {out_path}")
