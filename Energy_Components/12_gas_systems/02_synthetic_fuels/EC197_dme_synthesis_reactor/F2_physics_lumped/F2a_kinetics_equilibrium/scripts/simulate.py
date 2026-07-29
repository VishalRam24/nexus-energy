"""
EC197 -- DME Synthesis Reactor -- F2a Kinetics + Equilibrium
Plotly HTML simulation report generator (optional; safe import).
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

# --- Scenario 1: cooled reactor profile along residence time ---
r1 = model.predict({"T_in_K": 523.15, "P_bar": 40.0, "tau_max": 6.0, "n_eval": 150})

# --- Scenario 2: adiabatic hot-spot (runaway-prone) ---
r2 = model.predict({"T_in_K": 523.15, "P_bar": 40.0, "tau_max": 6.0,
                    "n_eval": 150, "adiabatic": True})

# --- Scenario 3: exit DME yield vs temperature (cooled) ---
T_sweep = np.linspace(480.0, 560.0, 30)
yield_T, sel_T, X_T = [], [], []
for T in T_sweep:
    rr = m.simulate(T, 40.0, tau_max=3.0, n_eval=40, adiabatic=False)
    X_T.append(rr["CO_conversion"][-1])
    yield_T.append(rr["DME_yield"][-1])
    sel_T.append(rr["DME_selectivity"][-1])

# --- Scenario 4: exit conversion vs pressure ---
P_sweep = np.linspace(15.0, 70.0, 25)
X_P, yield_P = [], []
for P in P_sweep:
    rr = m.simulate(523.15, P, tau_max=3.0, n_eval=40)
    X_P.append(rr["CO_conversion"][-1])
    yield_P.append(rr["DME_yield"][-1])

fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=[
        "Species molar flow along reactor (cooled, 250C/40bar)",
        "Temperature profile: cooled vs adiabatic",
        "CO conversion & DME selectivity along reactor (cooled)",
        "Heat release rate along reactor",
        "Exit DME yield / selectivity / X_CO vs T (40 bar)",
        "Exit X_CO & DME yield vs Pressure (250C)",
    ],
    vertical_spacing=0.10, horizontal_spacing=0.10,
)

# Row 1,1 species
for sp, col in [("n_CO", "#1f77b4"), ("n_H2", "#7f7f7f"), ("n_CH3OH", "#ff7f0e"),
                ("n_DME", "#2ca02c"), ("n_H2O", "#17becf")]:
    fig.add_trace(go.Scatter(x=r1["tau"], y=r1[sp], name=sp,
                  line=dict(color=col)), row=1, col=1)

# Row 1,2 temperature
fig.add_trace(go.Scatter(x=r1["tau"], y=r1["T"], name="T cooled",
              line=dict(color="#1f77b4")), row=1, col=2)
fig.add_trace(go.Scatter(x=r2["tau"], y=r2["T"], name="T adiabatic",
              line=dict(color="#d62728")), row=1, col=2)

# Row 2,1 conversion + selectivity
fig.add_trace(go.Scatter(x=r1["tau"], y=r1["CO_conversion"], name="X_CO",
              line=dict(color="#1f77b4"), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=r1["tau"], y=r1["DME_selectivity"], name="S_DME",
              line=dict(color="#2ca02c"), showlegend=False), row=2, col=1)

# Row 2,2 heat release
fig.add_trace(go.Scatter(x=r1["tau"], y=r1["heat_release_kW"], name="Q cooled",
              line=dict(color="#1f77b4"), showlegend=False), row=2, col=2)
fig.add_trace(go.Scatter(x=r2["tau"], y=r2["heat_release_kW"], name="Q adiabatic",
              line=dict(color="#d62728"), showlegend=False), row=2, col=2)

# Row 3,1 vs T
fig.add_trace(go.Scatter(x=T_sweep - 273.15, y=X_T, name="X_CO(T)",
              line=dict(color="#1f77b4"), showlegend=False), row=3, col=1)
fig.add_trace(go.Scatter(x=T_sweep - 273.15, y=yield_T, name="DME yield(T)",
              line=dict(color="#2ca02c"), showlegend=False), row=3, col=1)
fig.add_trace(go.Scatter(x=T_sweep - 273.15, y=sel_T, name="S_DME(T)",
              line=dict(color="#ff7f0e"), showlegend=False), row=3, col=1)

# Row 3,2 vs P
fig.add_trace(go.Scatter(x=P_sweep, y=X_P, name="X_CO(P)",
              line=dict(color="#1f77b4"), showlegend=False), row=3, col=2)
fig.add_trace(go.Scatter(x=P_sweep, y=yield_P, name="DME yield(P)",
              line=dict(color="#2ca02c"), showlegend=False), row=3, col=2)

for r, c, xl, yl in [
    (1, 1, "tau (s-equiv)", "molar flow (mol/s)"),
    (1, 2, "tau (s-equiv)", "Temperature (K)"),
    (2, 1, "tau (s-equiv)", "fraction (-)"),
    (2, 2, "tau (s-equiv)", "Heat release (kW/m3)"),
    (3, 1, "Temperature (C)", "fraction (-)"),
    (3, 2, "Pressure (bar)", "fraction (-)"),
]:
    fig.update_xaxes(title_text=xl, row=r, col=c)
    fig.update_yaxes(title_text=yl, row=r, col=c)

fig.update_layout(
    title="<b>EC197 DME Synthesis Reactor -- F2a Kinetics + Equilibrium (LHHW + Lumped Energy Balance)</b>",
    height=1050, template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

out_path = os.path.join(OUTPUT_DIR, "EC197_F2a_kinetics_equilibrium_report.html")
fig.write_html(out_path, include_plotlyjs="cdn")
print(f"Report saved: {out_path}")
