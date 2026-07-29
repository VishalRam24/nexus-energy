"""
EC204 -- Calcium Looping -- F2a Carbonator/Calciner Coupled ODE
Plotly HTML simulation report generator (optional).
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

# Scenario 1: fresh-sorbent dynamic capture & thermal transient
r_fresh = model.predict({"cycle_number": 1, "T0_K": m.T_carb - 20.0,
                         "dt": 2.0, "duration_s": 300.0})
# Scenario 2: aged sorbent
r_aged = model.predict({"cycle_number": 200, "T0_K": m.T_carb - 20.0,
                        "dt": 2.0, "duration_s": 300.0})

# Scenario 3: Grasa-Abanades capacity vs cycle number
N_sweep = np.arange(1, 500)
X_N = m.carrying_capacity(N_sweep)

# Scenario 4: average population capacity vs make-up fraction
f_sweep = np.linspace(0.005, 0.30, 60)
X_ave = [m.average_capacity(f) for f in f_sweep]

fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=[
        "Capture efficiency transient (fresh vs aged)",
        "Carbonator temperature transient",
        "Sorbent conversion X(t)",
        "Grasa-Abanades capacity X_N vs cycle N",
        "Calciner duty (endothermic) X(t)",
        "Average capacity vs make-up fraction",
    ],
    vertical_spacing=0.10, horizontal_spacing=0.10,
)

fig.add_trace(go.Scatter(x=r_fresh["t"], y=r_fresh["capture_efficiency"],
              name="fresh N=1", line=dict(color="#2ca02c")), row=1, col=1)
fig.add_trace(go.Scatter(x=r_aged["t"], y=r_aged["capture_efficiency"],
              name="aged N=200", line=dict(color="#d62728")), row=1, col=1)

fig.add_trace(go.Scatter(x=r_fresh["t"], y=r_fresh["temperature"],
              name="T fresh", line=dict(color="#ff7f0e"), showlegend=False), row=1, col=2)

fig.add_trace(go.Scatter(x=r_fresh["t"], y=r_fresh["conversion"],
              name="X fresh", line=dict(color="#1f77b4"), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=r_aged["t"], y=r_aged["conversion"],
              name="X aged", line=dict(color="#9467bd"), showlegend=False), row=2, col=1)

fig.add_trace(go.Scatter(x=N_sweep, y=X_N, name="X_N",
              line=dict(color="#8c564b"), showlegend=False), row=2, col=2)

fig.add_trace(go.Scatter(x=r_fresh["t"], y=np.array(r_fresh["calciner_duty"]) / 1e3,
              name="duty kW", line=dict(color="#e377c2"), showlegend=False), row=3, col=1)

fig.add_trace(go.Scatter(x=f_sweep, y=X_ave, name="X_ave",
              line=dict(color="#17becf"), showlegend=False), row=3, col=2)

for r, c, xl, yl in [
    (1, 1, "Time (s)", "Capture efficiency (-)"),
    (1, 2, "Time (s)", "Temperature (K)"),
    (2, 1, "Time (s)", "Conversion X (-)"),
    (2, 2, "Cycle number N", "Capacity X_N (-)"),
    (3, 1, "Time (s)", "Calciner duty (kW)"),
    (3, 2, "Make-up fraction f0 (-)", "Avg capacity (-)"),
]:
    fig.update_xaxes(title_text=xl, row=r, col=c)
    fig.update_yaxes(title_text=yl, row=r, col=c)

fig.update_layout(
    title="<b>EC204 Calcium Looping -- F2a Coupled Carbonator/Calciner ODE</b>",
    height=1000, template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

out_path = os.path.join(OUTPUT_DIR, "EC204_F2a_calcium_looping_report.html")
fig.write_html(out_path, include_plotlyjs="cdn")
print(f"Report saved: {out_path}")
