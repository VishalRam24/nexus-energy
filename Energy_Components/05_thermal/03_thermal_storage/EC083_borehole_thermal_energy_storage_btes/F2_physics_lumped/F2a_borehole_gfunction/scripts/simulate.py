"""
EC083 -- BTES -- F2a Physics-Lumped
Plotly HTML simulation report generator (optional; import guarded).
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
    print("plotly not installed; skipping report. (pip install plotly)")
    sys.exit(0)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "model_files")
os.makedirs(OUTPUT_DIR, exist_ok=True)

model = ComponentModel()
m = model._model
DAY = 86400.0

# --- Scenario 1: 90-day charge from undisturbed ground ---
r1 = model.predict({"Q_fluid_W": 500000.0, "T_store0_C": 10.0,
                    "duration_s": 90 * DAY})

# --- Scenario 2: full seasonal cycle (charge then discharge) ---
def Q_season(t):
    return 500000.0 if t < 180 * DAY else -500000.0
r2 = m.simulate(Q_season, T_store0=10.0, T_amb=8.0,
                duration_s=360 * DAY, n_out=720)

# --- Scenario 3: g-function vs Fourier time ---
t_grid = np.logspace(3, 10, 200)
g_vals = m.g_function(t_grid)

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=[
        "90-day charge: store & fluid temperatures",
        "Seasonal cycle: store temperature & load",
        "Eskilson g-function (saturating ground response)",
        "Seasonal cycle: stored energy",
    ],
    vertical_spacing=0.13, horizontal_spacing=0.10,
)

fig.add_trace(go.Scatter(x=r1["t_days"], y=r1["T_store"], name="T_store",
              line=dict(color="#d62728")), row=1, col=1)
fig.add_trace(go.Scatter(x=r1["t_days"], y=r1["T_out"], name="T_out",
              line=dict(color="#1f77b4")), row=1, col=1)
fig.add_trace(go.Scatter(x=r1["t_days"], y=r1["T_wall"], name="T_wall",
              line=dict(color="#2ca02c", dash="dot")), row=1, col=1)

fig.add_trace(go.Scatter(x=r2["t_days"], y=r2["T_store"], name="T_store (season)",
              line=dict(color="#ff7f0e"), showlegend=False), row=1, col=2)

fig.add_trace(go.Scatter(x=t_grid, y=g_vals, name="g(t)",
              line=dict(color="#9467bd"), showlegend=False), row=2, col=1)

fig.add_trace(go.Scatter(x=r2["t_days"], y=r2["E_stored_MWh"], name="E_stored",
              line=dict(color="#8c564b"), showlegend=False), row=2, col=2)

fig.update_xaxes(title_text="time (days)", row=1, col=1)
fig.update_yaxes(title_text="Temperature (C)", row=1, col=1)
fig.update_xaxes(title_text="time (days)", row=1, col=2)
fig.update_yaxes(title_text="T_store (C)", row=1, col=2)
fig.update_xaxes(title_text="time (s)", type="log", row=2, col=1)
fig.update_yaxes(title_text="g", row=2, col=1)
fig.update_xaxes(title_text="time (days)", row=2, col=2)
fig.update_yaxes(title_text="E_stored (MWh)", row=2, col=2)

fig.update_layout(
    title="<b>EC083 BTES -- F2a Physics-Lumped (Borehole HX + g-function)</b>",
    height=850, template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

out_path = os.path.join(OUTPUT_DIR, "EC083_F2a_borehole_gfunction_report.html")
fig.write_html(out_path, include_plotlyjs="cdn")
print(f"Report saved: {out_path}")
