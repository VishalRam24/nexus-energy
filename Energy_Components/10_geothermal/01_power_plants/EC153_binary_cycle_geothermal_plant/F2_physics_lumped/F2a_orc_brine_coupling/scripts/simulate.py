"""
EC153 -- Binary Cycle Geothermal Plant -- F2a ORC-Brine Coupling
Plotly HTML simulation report generator.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from predict import ComponentModel
from model import BinaryCycleGeothermal_F2a, IsobutaneProperties

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("ERROR: plotly not installed. Run: pip install plotly")
    sys.exit(1)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "model_files")
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = ComponentModel()
m = model._model

# --- Scenario 1: Startup transient (cold evaporator) ---
r1 = model.predict({
    "T_brine_in": 443.15,
    "T_evap_init": 350.0,
    "dt": 5.0,
    "duration_s": 3000.0,
})

# --- Scenario 2: Brine temperature step change ---
def brine_step(t):
    return 443.15 if t < 1500 else 423.15

r2 = m.simulate(brine_step, 393.15, 5.0, 3000.0)

# --- Scenario 3: Parametric -- net power vs brine inlet temperature ---
T_brine_range = np.linspace(383.15, 493.15, 50)
W_net_param = []
eta_param = []
for Tb in T_brine_range:
    T_ev = m.effective_evap_temperature(Tb)
    cycle = m.orc_cycle(T_ev)
    W_net_param.append(max(cycle["W_net"], 0.0) / 1e6)
    eta_param.append(cycle["eta_thermal"])

# --- Scenario 4: Long-term brine decline (20 years compressed) ---
r4 = model.predict({
    "T_brine_in": 443.15,
    "T_evap_init": 393.15,
    "dt": 10.0,
    "duration_s": 6000.0,
    "brine_decline_years": 20.0,
})

fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=[
        "Startup Transient -- Evaporator Temperature",
        "Startup Transient -- Net Power Output",
        "Brine Step Response (170->150 C at t=1500s)",
        "Thermal Efficiency Response",
        "Net Power vs Brine Inlet Temperature",
        "20-Year Brine Decline -- Power Output",
    ],
    vertical_spacing=0.10,
    horizontal_spacing=0.10,
)

# Row 1: startup
fig.add_trace(go.Scatter(x=r1["t"], y=r1["T_evap"] - 273.15,
              name="T_evap", line=dict(color="#d62728")), row=1, col=1)
fig.add_trace(go.Scatter(x=r1["t"], y=r1["T_brine_out"] - 273.15,
              name="T_brine_out", line=dict(color="#1f77b4", dash="dash")),
              row=1, col=1)
fig.add_trace(go.Scatter(x=r1["t"], y=r1["W_net"] / 1e6,
              name="W_net", line=dict(color="#2ca02c")), row=1, col=2)

# Row 2: step response
fig.add_trace(go.Scatter(x=r2["t"], y=r2["T_brine_in"] - 273.15,
              name="T_brine_in", line=dict(color="#ff7f0e")),
              row=2, col=1)
fig.add_trace(go.Scatter(x=r2["t"], y=r2["W_net"] / 1e6,
              name="W_net step", line=dict(color="#2ca02c"),
              showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=r2["t"], y=r2["eta_thermal"],
              name="eta_th", line=dict(color="#9467bd")),
              row=2, col=2)

# Row 3: parametric & long-term
fig.add_trace(go.Scatter(x=T_brine_range - 273.15, y=W_net_param,
              name="W_net(T_brine)", line=dict(color="#1f77b4")),
              row=3, col=1)
fig.add_trace(go.Scatter(x=T_brine_range - 273.15, y=eta_param,
              name="eta_th(T_brine)", line=dict(color="#ff7f0e", dash="dot")),
              row=3, col=1)

# Long-term decline: convert time to years
t_years = r4["t"] / r4["t"][-1] * 20.0
fig.add_trace(go.Scatter(x=t_years, y=r4["W_net"] / 1e6,
              name="W_net decline", line=dict(color="#d62728")),
              row=3, col=2)
fig.add_trace(go.Scatter(x=t_years, y=r4["T_brine_in"] - 273.15,
              name="T_brine_in", line=dict(color="#1f77b4", dash="dash"),
              showlegend=False), row=3, col=2)

for r, c, xl, yl in [
    (1, 1, "Time (s)", "Temperature (C)"),
    (1, 2, "Time (s)", "Net Power (MW)"),
    (2, 1, "Time (s)", "Temperature (C) / Power (MW)"),
    (2, 2, "Time (s)", "Thermal Efficiency (-)"),
    (3, 1, "Brine Inlet Temperature (C)", "MW / Efficiency"),
    (3, 2, "Time (years)", "Net Power (MW)"),
]:
    fig.update_xaxes(title_text=xl, row=r, col=c)
    fig.update_yaxes(title_text=yl, row=r, col=c)

fig.update_layout(
    title="<b>EC153 Binary Cycle Geothermal -- F2a ORC-Brine Coupling</b>",
    height=1000, template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

out_path = os.path.join(OUTPUT_DIR, "EC153_F2a_orc_brine_coupling_report.html")
fig.write_html(out_path, include_plotlyjs="cdn")
print(f"Report saved: {out_path}")
