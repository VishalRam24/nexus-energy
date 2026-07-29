"""
EC189 -- Natural Gas Pipeline -- F2a Line-Pack Dynamics
Plotly HTML simulation report generator (optional; plotly import guarded).
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
    sys.exit(0)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "model_files")
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = ComponentModel()
m = model._model

# --- Scenario 1: line-pack charging (surplus supply) ---
r1 = model.predict({"P_avg0_bar": 55.0, "P_out_bar": 50.0,
                    "m_in_kg_s": 200.0, "dt": 60.0, "duration_s": 6 * 3600.0})

# --- Scenario 2: diurnal demand swing (time-varying inflow) ---
def diurnal_in(t):
    return 120.0 + 60.0 * np.sin(2 * np.pi * t / 86400.0)

r2 = m.simulate(60.0, 50.0, diurnal_in, dt=300.0, duration_s=86400.0)

# --- Scenario 3: driven-pressure approach to steady state ---
r3 = model.predict({"P_avg0_bar": 55.0, "P_out_bar": 50.0,
                    "P_in_bar": 70.0, "dt": 60.0, "duration_s": 6 * 3600.0})

# --- Scenario 4: steady flow-equation curve Q vs delivery pressure ---
P_out_sweep = np.linspace(30e5, 69e5, 100)
Q_curve = [m.flow_rate_std_m3_day(70e5, P2) / 1e6 for P2 in P_out_sweep]  # Mm3/d

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        "Line-pack charging (m_in=200 kg/s surplus)",
        "Diurnal demand swing -- line-pack buffering",
        "Driven-pressure approach to steady state",
        "Steady flow eq: Q vs delivery pressure (P1=70 bar)",
    ],
    specs=[[{"secondary_y": True}, {"secondary_y": True}],
           [{"secondary_y": True}, {}]],
    vertical_spacing=0.13, horizontal_spacing=0.10,
)

fig.add_trace(go.Scatter(x=r1["t"] / 3600, y=r1["P_avg"], name="P_avg",
              line=dict(color="#d62728")), row=1, col=1)
fig.add_trace(go.Scatter(x=r1["t"] / 3600, y=r1["linepack_mass"] / 1e3,
              name="line-pack (t)", line=dict(color="#1f77b4")),
              row=1, col=1, secondary_y=True)

fig.add_trace(go.Scatter(x=r2["t"] / 3600, y=r2["m_in"], name="m_in",
              line=dict(color="#2ca02c")), row=1, col=2)
fig.add_trace(go.Scatter(x=r2["t"] / 3600, y=r2["m_out"], name="m_out",
              line=dict(color="#ff7f0e")), row=1, col=2)
fig.add_trace(go.Scatter(x=r2["t"] / 3600, y=r2["P_avg"], name="P_avg",
              line=dict(color="#9467bd", dash="dot")),
              row=1, col=2, secondary_y=True)

fig.add_trace(go.Scatter(x=r3["t"] / 3600, y=r3["m_in"], name="m_in (drv)",
              line=dict(color="#17becf")), row=2, col=1)
fig.add_trace(go.Scatter(x=r3["t"] / 3600, y=r3["m_out"], name="m_out (drv)",
              line=dict(color="#bcbd22")), row=2, col=1)
fig.add_trace(go.Scatter(x=r3["t"] / 3600, y=r3["P_avg"], name="P_avg (drv)",
              line=dict(color="#e377c2", dash="dot")),
              row=2, col=1, secondary_y=True)

fig.add_trace(go.Scatter(x=P_out_sweep / 1e5, y=Q_curve, name="Q(P2)",
              line=dict(color="#8c564b")), row=2, col=2)

fig.update_xaxes(title_text="Time (h)", row=1, col=1)
fig.update_xaxes(title_text="Time (h)", row=1, col=2)
fig.update_xaxes(title_text="Time (h)", row=2, col=1)
fig.update_xaxes(title_text="Delivery pressure (bar)", row=2, col=2)
fig.update_yaxes(title_text="P_avg (bar)", row=1, col=1)
fig.update_yaxes(title_text="line-pack (t)", row=1, col=1, secondary_y=True)
fig.update_yaxes(title_text="Flow (kg/s)", row=1, col=2)
fig.update_yaxes(title_text="P_avg (bar)", row=1, col=2, secondary_y=True)
fig.update_yaxes(title_text="Flow (kg/s)", row=2, col=1)
fig.update_yaxes(title_text="P_avg (bar)", row=2, col=1, secondary_y=True)
fig.update_yaxes(title_text="Q (Mm3/day)", row=2, col=2)

fig.update_layout(
    title="<b>EC189 NG Pipeline -- F2a Isothermal Line-Pack Dynamics</b>",
    height=820, template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1),
)

out_path = os.path.join(OUTPUT_DIR, "EC189_F2a_linepack_dynamics_report.html")
fig.write_html(out_path, include_plotlyjs="cdn")
print(f"Report saved: {out_path}")
