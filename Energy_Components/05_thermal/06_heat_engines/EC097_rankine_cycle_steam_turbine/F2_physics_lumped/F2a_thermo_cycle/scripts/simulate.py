"""
EC097 -- Rankine Cycle (Steam Turbine) -- F2a Physics-Lumped Thermo Cycle
Plotly HTML simulation report generator.
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
    print("ERROR: plotly not installed. Run: pip install plotly")
    sys.exit(1)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "model_files")
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = ComponentModel()
m = model._model

# --- Scenario 1: boiler-drum transient from cold start ---
r1 = model.predict({"mode": "transient", "duration_s": 6000.0, "dt": 20.0})
tr = r1["transient"]

# --- Scenario 2: fuel step (60% -> 100% at t=2000 s) ---
def step_fuel(t):
    return 0.6 * m.Q_fuel_design if t < 2000 else m.Q_fuel_design

tr2 = m.simulate(step_fuel, dt=20.0, duration_s=6000.0)

# --- Scenario 3: efficiency vs superheat temperature ---
T_sh = np.linspace(300.0, 600.0, 60)
eta_plain = [m.solve_cycle(T_superheat=T)["eta_thermal"] for T in T_sh]
eta_reheat = [m.solve_cycle(T_superheat=T, reheat=True)["eta_thermal"] for T in T_sh]
eta_carnot = [m.solve_cycle(T_superheat=T)["eta_carnot"] for T in T_sh]

# --- Scenario 4: efficiency vs condenser pressure ---
P_c = np.linspace(0.03, 0.5, 60)
eta_pc = [m.solve_cycle(P_condenser=P)["eta_thermal"] for P in P_c]
x_pc = [m.solve_cycle(P_condenser=P)["x_turbine_exit"] for P in P_c]

# --- Scenario 5: cycle configuration comparison ---
cfgs = [("Basic", {}), ("Reheat", {"reheat": True}),
        ("Regen", {"regeneration": True}),
        ("Reheat+Regen", {"reheat": True, "regeneration": True})]
labels = [c[0] for c in cfgs]
etas = [m.solve_cycle(**c[1])["eta_thermal"] for c in cfgs]

fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=[
        "Boiler-Drum Transient (cold start, 100% fuel)",
        "Instantaneous Efficiency During Transient",
        "Fuel Step Response (60%->100% at t=2000s)",
        "Efficiency vs Superheat Temperature",
        "Efficiency / Exit Quality vs Condenser Pressure",
        "Cycle Configuration Comparison (eta_thermal)",
    ],
    vertical_spacing=0.10,
    horizontal_spacing=0.12,
    specs=[[{}, {}], [{}, {}], [{"secondary_y": True}, {}]],
)

# Row 1: transient T_drum + eta
fig.add_trace(go.Scatter(x=tr["t"], y=tr["T_drum"] - 273.15, name="T_drum",
              line=dict(color="#d62728")), row=1, col=1)
fig.add_trace(go.Scatter(x=tr["t"], y=tr["eta_thermal"], name="eta(t)",
              line=dict(color="#1f77b4"), showlegend=False), row=1, col=2)

# Row 2: fuel step
fig.add_trace(go.Scatter(x=tr2["t"], y=tr2["T_drum"] - 273.15, name="T_drum step",
              line=dict(color="#2ca02c"), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=tr2["t"], y=tr2["Q_fuel_W"] / 1e6, name="Q_fuel",
              line=dict(color="#ff7f0e"), showlegend=False), row=2, col=1)

# Row 2 col 2: eta vs superheat
fig.add_trace(go.Scatter(x=T_sh, y=eta_plain, name="Basic",
              line=dict(color="#1f77b4")), row=2, col=2)
fig.add_trace(go.Scatter(x=T_sh, y=eta_reheat, name="Reheat",
              line=dict(color="#2ca02c")), row=2, col=2)
fig.add_trace(go.Scatter(x=T_sh, y=eta_carnot, name="Carnot bound",
              line=dict(color="#7f7f7f", dash="dash")), row=2, col=2)

# Row 3 col 1: eta + quality vs condenser P (twin axes)
fig.add_trace(go.Scatter(x=P_c, y=eta_pc, name="eta",
              line=dict(color="#9467bd"), showlegend=False),
              row=3, col=1, secondary_y=False)
fig.add_trace(go.Scatter(x=P_c, y=x_pc, name="x_exit",
              line=dict(color="#8c564b", dash="dot"), showlegend=False),
              row=3, col=1, secondary_y=True)

# Row 3 col 2: config bar chart
fig.add_trace(go.Bar(x=labels, y=etas, name="config eta",
              marker_color=["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"],
              showlegend=False), row=3, col=2)

fig.update_xaxes(title_text="Time (s)", row=1, col=1)
fig.update_yaxes(title_text="T_drum (degC)", row=1, col=1)
fig.update_xaxes(title_text="Time (s)", row=1, col=2)
fig.update_yaxes(title_text="eta_thermal (-)", row=1, col=2)
fig.update_xaxes(title_text="Time (s)", row=2, col=1)
fig.update_yaxes(title_text="T_drum (degC) / Q_fuel (MW)", row=2, col=1)
fig.update_xaxes(title_text="Superheat T (degC)", row=2, col=2)
fig.update_yaxes(title_text="efficiency (-)", row=2, col=2)
fig.update_xaxes(title_text="Condenser P (bar)", row=3, col=1)
fig.update_yaxes(title_text="eta_thermal (-)", row=3, col=1, secondary_y=False)
fig.update_yaxes(title_text="exit quality x (-)", row=3, col=1, secondary_y=True)
fig.update_yaxes(title_text="eta_thermal (-)", row=3, col=2)

fig.update_layout(
    title="<b>EC097 Rankine Cycle Steam Turbine -- F2a Thermodynamic Cycle + Boiler-Drum ODE</b>",
    height=1050, template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

out_path = os.path.join(OUTPUT_DIR, "EC097_F2a_thermo_cycle_report.html")
fig.write_html(out_path, include_plotlyjs="cdn")
print(f"Report saved: {out_path}")
